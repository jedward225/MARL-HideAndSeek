#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/render/api.hpp>

#include <array>
#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_MWGPU_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::py;

namespace GPUHideSeek {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};

static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg)
{
    if (mgr_cfg.extRenderDev || !mgr_cfg.enableBatchRenderer) {
        return Optional<RenderGPUState>::none();
    }

    auto render_api_lib = render::APIManager::loadDefaultLib();
    render::APIManager render_api_mgr(render_api_lib.lib());
    render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

    return RenderGPUState {
        .apiLib = std::move(render_api_lib),
        .apiMgr = std::move(render_api_mgr),
        .gpu = std::move(gpu),
    };
}

static inline Optional<render::RenderManager> initRenderManager(
    const Manager::Config &mgr_cfg,
    const Optional<RenderGPUState> &render_gpu_state)
{
    if (!mgr_cfg.extRenderDev && !mgr_cfg.enableBatchRenderer) {
        return Optional<render::RenderManager>::none();
    }

    render::APIBackend *render_api;
    render::GPUDevice *render_dev;

    if (render_gpu_state.has_value()) {
        render_api = render_gpu_state->apiMgr.backend();
        render_dev = render_gpu_state->gpu.device();
    } else {
        render_api = mgr_cfg.extRenderAPI;
        render_dev = mgr_cfg.extRenderDev;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = mgr_cfg.enableBatchRenderer,
        .renderMode = render::RenderManager::Config::RenderMode::RGBD,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = mgr_cfg.maxHiders + mgr_cfg.maxSeekers,
        .maxInstancesPerWorld = 1000,
        .execMode = mgr_cfg.execMode,
        .voxelCfg = {},
    });
}

struct Manager::Impl {
    Config cfg;
    int32_t maxAgentsPerWorld;
    PhysicsLoader physicsLoader;
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;
    WorldReset *resetsPointer;
    Action *actionsPointer;

    static inline Impl * make(const Config &cfg);


    template <EnumType EnumT>
    Tensor exportStateTensor(EnumT slot,
                             TensorElementType type,
                             Span<const int64_t> dimensions);

};

struct Manager::CPUImpl : Manager::Impl {
    using TaskGraphT =
        TaskGraphExecutor<Engine, Sim, GPUHideSeek::Config, WorldInit>;

    TaskGraphT cpuExec;

    inline void init();
    inline void step();

    inline void saveCheckpoints();
    inline void loadCheckpoints();

#ifdef MADRONA_MWGPU_SUPPORT
    inline void gpuStreamInit(cudaStream_t strm, void **buffers, Manager &mgr);
    inline void gpuStreamStep(cudaStream_t strm, void **buffers, Manager &mgr);
#endif
};

void Manager::CPUImpl::init()
{
    cpuExec.runTaskGraph(TaskGraphID::Init);
}

void Manager::CPUImpl::step()
{
    cpuExec.runTaskGraph(TaskGraphID::Step);
}

void Manager::CPUImpl::saveCheckpoints()
{
    cpuExec.runTaskGraph(TaskGraphID::SaveCheckpoints);
}

void Manager::CPUImpl::loadCheckpoints()
{
    cpuExec.runTaskGraph(TaskGraphID::LoadCheckpoints);
}

#ifdef MADRONA_MWGPU_SUPPORT
void Manager::CPUImpl::gpuStreamInit(cudaStream_t, void **, Manager &)
{
    assert(false);
}

void Manager::CPUImpl::gpuStreamStep(cudaStream_t, void **, Manager &)
{
    assert(false);
}
#endif

#ifdef MADRONA_MWGPU_SUPPORT

static inline uint64_t numTensorBytes(const Tensor &t)
{
    uint64_t num_items = 1;
    uint64_t num_dims = t.numDims();
    for (uint64_t i = 0; i < num_dims; i++) {
        num_items *= t.dims()[i];
    }

    return num_items * (uint64_t)t.numBytesPerItem();
}

struct JAXIOObservations {
    AgentPrepCounter *prepCounter;
    SelfObservation *selfObs;
    AgentType *selfType;
    AgentActiveMask *selfMask;
    Lidar *lidar;

    RelativeAgentObservations *agentData;
    RelativeBoxObservations *boxData;
    RelativeRampObservations *rampData;

    AgentVisibilityMasks *agentMasks;
    BoxVisibilityMasks *boxMasks;
    RampVisibilityMasks *rampMasks;

    inline void ** set(void **ptrs)
    {
      prepCounter = (AgentPrepCounter *)*ptrs++;
      selfObs = (SelfObservation *)*ptrs++;
      selfType = (AgentType *)*ptrs++;
      selfMask = (AgentActiveMask *)*ptrs++;
      lidar = (Lidar *)*ptrs++;

      agentData = (RelativeAgentObservations *)*ptrs++;
      boxData = (RelativeBoxObservations *)*ptrs++;
      rampData = (RelativeRampObservations *)*ptrs++;

      agentMasks = (AgentVisibilityMasks *)*ptrs++;
      boxMasks = (BoxVisibilityMasks *)*ptrs++;
      rampMasks = (RampVisibilityMasks *)*ptrs++;

      return ptrs;
    }
};

struct JAXIOSaveCheckpoint {
    CheckpointControl *ckptCtrl;
    Checkpoint *ckptData;

#ifdef MADRONA_CUDA_SUPPORT
    static inline JAXIOSaveCheckpoint gpuSetup(void **buffers)
    {
        JAXIOSaveCheckpoint io;
        buffers = io.setupInputs(buffers);
        buffers = io.setupOutputs(buffers);
        return io;
    }
#endif

    static inline JAXIOSaveCheckpoint cpuSetup(void **inputs, void **outputs)
    {
        JAXIOSaveCheckpoint io;
        io.setupInputs(inputs);
        io.setupOutputs(outputs);
        return io;
    }

private:
    inline void ** setupInputs(void **ptrs)
    {
        ckptCtrl = (CheckpointControl *)*ptrs++;

        return ptrs;
    }

    inline void ** setupOutputs(void **ptrs)
    {
        ckptData = (Checkpoint *)*ptrs++;

        return ptrs;
    }
};

struct JAXIORestoreCheckpoint {
    CheckpointControl *ckptCtrl;
    Checkpoint *ckptData;
    JAXIOObservations obs;

#ifdef MADRONA_CUDA_SUPPORT
    static inline JAXIORestoreCheckpoint gpuSetup(void **buffers)
    {
        JAXIORestoreCheckpoint io;
        buffers = io.setupInputs(buffers);
        buffers = io.setupOutputs(buffers);
        return io;
    }
#endif

    static inline JAXIORestoreCheckpoint cpuSetup(void **inputs, void **outputs)
    {
        JAXIORestoreCheckpoint io;
        io.setupInputs(inputs);
        io.setupOutputs(outputs);
        return io;
    }

private:
    inline void ** setupInputs(void **ptrs)
    {
        ckptCtrl = (CheckpointControl *)*ptrs++;
        ckptData = (Checkpoint *)*ptrs++;

        return ptrs;
    }

    inline void ** setupOutputs(void **ptrs)
    {
        return obs.set(ptrs);
    }
};


struct Manager::CUDAImpl : Manager::Impl {
    MWCudaExecutor mwGPU;
    MWCudaLaunchGraph stepGraph;
    MWCudaLaunchGraph saveCkptGraph;
    MWCudaLaunchGraph loadCkptGraph;

    inline void init();
    inline void step();

    inline void saveCheckpoints();
    inline void loadCheckpoints();

    inline void copyFromSim(cudaStream_t strm, void *dst, const Tensor &src);
    inline void copyToSim(cudaStream_t strm, const Tensor &dst, void *src);
    inline void copyOutObservations(
        cudaStream_t strm, JAXIOObservations &io, Manager &mgr);

    inline void gpuStreamInit(cudaStream_t strm, void **buffers, Manager &mgr);
    inline void gpuStreamStep(cudaStream_t strm, void **buffers, Manager &mgr);

    void gpuJAXSaveCheckpoints(cudaStream_t strm, void **buffers, Manager &mgr);
    void gpuJAXLoadCheckpoints(cudaStream_t strm, void **buffers, Manager &mgr);
};

void Manager::CUDAImpl::init()
{
    MWCudaLaunchGraph init_graph = mwGPU.buildLaunchGraph(TaskGraphID::Init);

    mwGPU.run(init_graph);
}

void Manager::CUDAImpl::step()
{
    mwGPU.run(stepGraph);
}

void Manager::CUDAImpl::saveCheckpoints()
{
    mwGPU.run(saveCkptGraph);
}

void Manager::CUDAImpl::loadCheckpoints()
{
    mwGPU.run(loadCkptGraph);
}

void Manager::CUDAImpl::copyFromSim(
    cudaStream_t strm, void *dst, const Tensor &src)
{
    uint64_t num_bytes = numTensorBytes(src);

    REQ_CUDA(cudaMemcpyAsync(dst, src.devicePtr(), num_bytes,
        cudaMemcpyDeviceToDevice, strm));
}

void Manager::CUDAImpl::copyToSim(
    cudaStream_t strm, const Tensor &dst, void *src)
{
    uint64_t num_bytes = numTensorBytes(dst);

    REQ_CUDA(cudaMemcpyAsync(dst.devicePtr(), src, num_bytes,
        cudaMemcpyDeviceToDevice, strm));
}

void Manager::CUDAImpl::copyOutObservations(
    cudaStream_t strm,
    JAXIOObservations &io,
    Manager &mgr)

{
    // Observations
    copyFromSim(strm, io.prepCounter, mgr.prepCounterTensor());
    copyFromSim(strm, io.selfObs, mgr.selfDataTensor());
    copyFromSim(strm, io.selfType, mgr.selfTypeTensor());
    copyFromSim(strm, io.selfMask, mgr.selfMaskTensor());
    copyFromSim(strm, io.lidar, mgr.lidarTensor());

    copyFromSim(strm, io.agentData, mgr.agentDataTensor());
    copyFromSim(strm, io.boxData, mgr.boxDataTensor());
    copyFromSim(strm, io.rampData, mgr.rampDataTensor());
    copyFromSim(strm, io.agentMasks, mgr.visibleAgentsMaskTensor());
    copyFromSim(strm, io.boxMasks, mgr.visibleBoxesMaskTensor());
    copyFromSim(strm, io.rampMasks, mgr.visibleRampsMaskTensor());
}

void Manager::CUDAImpl::gpuStreamInit(
    cudaStream_t strm, void **buffers, Manager &mgr)
{
    MWCudaLaunchGraph init_graph = mwGPU.buildLaunchGraph(TaskGraphID::Init);

    mwGPU.runAsync(init_graph, strm);
    
    JAXIOObservations io_obs;
    buffers = io_obs.set(buffers);
    copyOutObservations(strm, io_obs, mgr);

    REQ_CUDA(cudaStreamSynchronize(strm));
}

void Manager::CUDAImpl::gpuStreamStep(
    cudaStream_t strm, void **buffers, Manager &mgr)
{
    copyToSim(strm, mgr.actionTensor(), *buffers++);
    copyToSim(strm, mgr.resetTensor(), *buffers++);

    copyToSim(strm, mgr.policyAssignmentsTensor(), *buffers++);
    //copyToSim(strm, mgr.rewardHyperParamsTensor(), *buffers++);

    mwGPU.runAsync(stepGraph, strm);

    JAXIOObservations io_obs;
    buffers = io_obs.set(buffers);
    copyOutObservations(strm, io_obs, mgr);

    copyFromSim(strm, *buffers++, mgr.rewardTensor());
    copyFromSim(strm, *buffers++, mgr.doneTensor());

    copyFromSim(strm, *buffers++, mgr.episodeResultTensor());
}

void Manager::CUDAImpl::gpuJAXSaveCheckpoints(cudaStream_t strm,
                                              void **buffers,
                                              Manager &)
{
    JAXIOSaveCheckpoint jax_io = JAXIOSaveCheckpoint::gpuSetup(buffers);

    cudaMemcpyAsync(mwGPU.getExported((i32)ExportID::CheckpointControl),
        jax_io.ckptCtrl, sizeof(CheckpointControl) * cfg.numWorlds,
        cudaMemcpyDeviceToDevice, strm);

    mwGPU.runAsync(saveCkptGraph, strm);

    cudaMemcpyAsync(jax_io.ckptData,
        mwGPU.getExported((i32)ExportID::Checkpoint),
        sizeof(Checkpoint) * cfg.numWorlds,
        cudaMemcpyDeviceToDevice, strm);
}

void Manager::CUDAImpl::gpuJAXLoadCheckpoints(
    cudaStream_t strm, void **buffers, Manager &mgr)
{
    JAXIORestoreCheckpoint jax_io =
        JAXIORestoreCheckpoint::gpuSetup(buffers);

    cudaMemcpyAsync(mwGPU.getExported((i32)ExportID::CheckpointControl),
        jax_io.ckptCtrl,
        sizeof(CheckpointControl) * cfg.numWorlds,
        cudaMemcpyDeviceToDevice, strm);

    cudaMemcpyAsync(mwGPU.getExported((i32)ExportID::Checkpoint),
        jax_io.ckptData,
        sizeof(Checkpoint) * cfg.numWorlds,
        cudaMemcpyDeviceToDevice, strm);

    mwGPU.runAsync(loadCkptGraph, strm);

    copyOutObservations(strm, jax_io.obs, mgr);
}

#endif

static void loadPhysicsObjects(imp::AssetImporter &asset_importer,
                               PhysicsLoader &loader)
{
    SourceCollisionPrimitive sphere_prim {
        .type = CollisionPrimitive::Type::Sphere,
        .sphere = CollisionPrimitive::Sphere {
            .radius = 1.f,
        },
    };

    SourceCollisionPrimitive plane_prim {
        .type = CollisionPrimitive::Type::Plane,
        .plane = {},
    };

    char import_err_buffer[4096];
    auto imported_hulls = asset_importer.importFromDisk({
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "wall_collision.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "agent_collision.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "ramp_collision.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "elongated_collision.obj").string().c_str(),
    }, import_err_buffer, true);

    if (!imported_hulls.has_value()) {
        FATAL("%s", import_err_buffer);
    }

    DynArray<imp::SourceMesh> src_convex_hulls(
        imported_hulls->objects.size());

    DynArray<DynArray<SourceCollisionPrimitive>> prim_arrays(0);
    HeapArray<SourceCollisionObject> src_objs((uint32_t)SimObject::NumObjects);

    // Sphere (0)
    src_objs[(uint32_t)SimObject::Sphere] = {
        .prims = Span<const SourceCollisionPrimitive>(&sphere_prim, 1),
        .invMass = 1.f,
        .friction = {
            .muS = 0.5f,
            .muD = 0.5f,
        },
    };

    // Plane (1)
    src_objs[(uint32_t)SimObject::Plane] = {
        .prims = Span<const SourceCollisionPrimitive>(&plane_prim, 1),
        .invMass = 0.f,
        .friction = {
            .muS = 2.f,
            .muD = 2.f,
        },
    };

    auto setupHull = [&](CountT obj_idx, float inv_mass,
                         RigidBodyFrictionData friction) {
        auto meshes = imported_hulls->objects[obj_idx].meshes;
        DynArray<SourceCollisionPrimitive> prims(meshes.size());

        for (const imp::SourceMesh &mesh : meshes) {
            src_convex_hulls.push_back(mesh);
            prims.push_back({
                .type = CollisionPrimitive::Type::Hull,
                .hullInput = {
                    .hullIDX = uint32_t(src_convex_hulls.size() - 1),
                },
            });
        }

        prim_arrays.emplace_back(std::move(prims));

        return SourceCollisionObject {
            .prims = Span<const SourceCollisionPrimitive>(prim_arrays.back()),
            .invMass = inv_mass,
            .friction = friction,
        };
    };

    { // Cube
        src_objs[(uint32_t)SimObject::Cube] = setupHull(0, 0.5f, {
            .muS = 0.5f,
            .muD = 2.f,
        });
    }

    { // Wall
        src_objs[(uint32_t)SimObject::Wall] = setupHull(1, 0.f, {
            .muS = 0.5f,
            .muD = 2.f,
        });
    }

    { // Cylinder
        src_objs[(uint32_t)SimObject::Hider] = setupHull(2, 1.f, {
            .muS = 0.5f,
            .muD = 16.f,
        });
    }

    { // Cylinder
        src_objs[(uint32_t)SimObject::Seeker] = setupHull(2, 1.f, {
            .muS = 0.5f,
            .muD = 16.f,
        });
    }

    { // Ramp
        src_objs[(uint32_t)SimObject::Ramp] = setupHull(3, 0.5f, {
            .muS = 0.5f,
            .muD = 1.f,
        });
    }

    { // Elongated Box
        src_objs[(uint32_t)SimObject::Box] = setupHull(4, 0.5f, {
            .muS = 0.5f,
            .muD = 4.f,
        });
    }

    StackAlloc tmp_alloc;
    RigidBodyAssets rigid_body_assets;
    CountT num_rigid_body_data_bytes;
    void *rigid_body_data = RigidBodyAssets::processRigidBodyAssets(
        src_convex_hulls,
        src_objs,
        false,
        tmp_alloc,
        &rigid_body_assets,
        &num_rigid_body_data_bytes);

    if (rigid_body_data == nullptr) {
        FATAL("Invalid collision hull input");
    }

    // HACK:
    rigid_body_assets.metadatas[
        (uint32_t)SimObject::Hider].mass.invInertiaTensor.x = 0.f;
    rigid_body_assets.metadatas[
        (uint32_t)SimObject::Hider].mass.invInertiaTensor.y = 0.f;
    rigid_body_assets.metadatas[
        (uint32_t)SimObject::Seeker].mass.invInertiaTensor.x = 0.f;
    rigid_body_assets.metadatas[
        (uint32_t)SimObject::Seeker].mass.invInertiaTensor.y = 0.f;

    loader.loadRigidBodies(rigid_body_assets);
    free(rigid_body_data);
}

static void loadRenderObjects(imp::AssetImporter &asset_importer,
                              render::RenderManager &render_mgr)
{
    std::array<std::string, (size_t)SimObject::NumObjects> render_asset_paths;
    render_asset_paths[(size_t)SimObject::Sphere] =
        (std::filesystem::path(DATA_DIR) / "sphere.obj").string();
    render_asset_paths[(size_t)SimObject::Plane] =
        (std::filesystem::path(DATA_DIR) / "plane.obj").string();
    render_asset_paths[(size_t)SimObject::Cube] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::Wall] =
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string();
    render_asset_paths[(size_t)SimObject::Hider] =
        (std::filesystem::path(DATA_DIR) / "agent_render.obj").string();
    render_asset_paths[(size_t)SimObject::Seeker] =
        (std::filesystem::path(DATA_DIR) / "agent_render.obj").string();
    render_asset_paths[(size_t)SimObject::Ramp] =
        (std::filesystem::path(DATA_DIR) / "ramp_render.obj").string();
    render_asset_paths[(size_t)SimObject::Box] =
        (std::filesystem::path(DATA_DIR) / "elongated_render.obj").string();

    std::array<const char *, (size_t)SimObject::NumObjects> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs[i] = render_asset_paths[i].c_str();
    }

    std::array<char, 1024> import_err;
    auto render_assets = asset_importer.importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()));

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    auto materials = std::to_array<imp::SourceMaterial>({
        { math::Vector4{0.4f, 0.4f, 0.4f, 0.0f}, -1, 0.8f, 0.2f,},
        { math::Vector4{1.0f, 0.1f, 0.1f, 0.0f}, -1, 0.8f, 0.2f,},
        { math::Vector4{1.0f, 1.0f, 1.0f, 0.0f}, 1, 0.8f, 1.0f,},
        { math::Vector4{0.5f, 0.3f, 0.3f, 0.0f},  0, 0.8f, 0.2f,},
        { render::rgb8ToFloat(191, 108, 10), -1, 0.8f, 0.2f },
        { render::rgb8ToFloat(12, 144, 150), -1, 0.8f, 0.2f },
        { render::rgb8ToFloat(230, 230, 230),   -1, 0.8f, 1.0f },
        { math::Vector4{1.0f, 1.0f, 1.0f, 0.0f}, 2, 0.8f, 1.0f,},
    });

    // Override materials
    render_assets->objects[(uint32_t)SimObject::Sphere].meshes[0].materialIDX = 0;
    render_assets->objects[(uint32_t)SimObject::Plane].meshes[0].materialIDX = 3;
    render_assets->objects[(uint32_t)SimObject::Cube].meshes[0].materialIDX = 1;
    render_assets->objects[(uint32_t)SimObject::Wall].meshes[0].materialIDX = 0;
    render_assets->objects[(uint32_t)SimObject::Hider].meshes[0].materialIDX = 2;
    render_assets->objects[(uint32_t)SimObject::Hider].meshes[1].materialIDX = 6;
    render_assets->objects[(uint32_t)SimObject::Hider].meshes[2].materialIDX = 6;
    render_assets->objects[(uint32_t)SimObject::Seeker].meshes[0].materialIDX = 7;
    render_assets->objects[(uint32_t)SimObject::Seeker].meshes[1].materialIDX = 6;
    render_assets->objects[(uint32_t)SimObject::Seeker].meshes[2].materialIDX = 6;
    render_assets->objects[(uint32_t)SimObject::Ramp].meshes[0].materialIDX = 4;
    render_assets->objects[(uint32_t)SimObject::Box].meshes[0].materialIDX = 5;

    auto &image_importer = asset_importer.imageImporter();

    StackAlloc tmp_alloc;
    auto imported_textures = image_importer.importImages(tmp_alloc, {
        (std::filesystem::path(DATA_DIR) /
           "green_grid.png").string().c_str(),
        (std::filesystem::path(DATA_DIR) /
           "smile.png").string().c_str(),
        (std::filesystem::path(DATA_DIR) /
           "red_smile.png").string().c_str(),
    });

    if (imported_textures.size() == 0) {
        FATAL("Failed to load textures");
    }

    render_mgr.loadObjects(render_assets->objects, materials, imported_textures);

    image_importer.deallocImportedImages(imported_textures);

    render_mgr.configureLighting({
        { true, math::Vector3{1.0f, 1.0f, -2.0f}, math::Vector3{1.0f, 1.0f, 1.0f} }
    });
}

Manager::Impl * Manager::Impl::make(const Config &cfg)
{
    GPUHideSeek::Config app_cfg;
    app_cfg.simFlags = cfg.simFlags;
    app_cfg.initRandKey = rand::initKey(cfg.randSeed);
    app_cfg.minHiders = cfg.minHiders;
    app_cfg.maxHiders = cfg.maxHiders;
    app_cfg.minSeekers = cfg.minSeekers;
    app_cfg.maxSeekers = cfg.maxSeekers;

    int32_t max_agents_per_world = cfg.maxHiders + cfg.maxSeekers;

    imp::AssetImporter asset_importer;

    switch (cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_MWGPU_SUPPORT
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(cfg.gpuID);

        PhysicsLoader phys_loader(cfg.execMode, 10);
        loadPhysicsObjects(asset_importer, phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
        app_cfg.rigidBodyObjMgr = phys_obj_mgr;

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(cfg, render_gpu_state);

        if (render_mgr.has_value()) {
            loadRenderObjects(asset_importer, *render_mgr);
            app_cfg.renderBridge = render_mgr->bridge();
         } else {
            app_cfg.renderBridge = nullptr;
         }

        HeapArray<WorldInit> world_inits(cfg.numWorlds);

        MWCudaExecutor mwgpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(WorldInit),
            .userConfigPtr = &app_cfg,
            .numUserConfigBytes = sizeof(GPUHideSeek::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = cfg.numWorlds,
            .numTaskGraphs = (uint32_t)TaskGraphID::NumTaskGraphs,
            .numExportedBuffers = (uint32_t)ExportID::NumExports,
        }, {
            { GPU_HIDESEEK_SRC_LIST },
            { GPU_HIDESEEK_COMPILE_FLAGS },
            CompileConfig::OptMode::LTO,
        }, cu_ctx);

        MWCudaLaunchGraph step_graph = mwgpu_exec.buildLaunchGraph(
            TaskGraphID::Step);

        MWCudaLaunchGraph save_ckpt_graph = mwgpu_exec.buildLaunchGraph(
            TaskGraphID::SaveCheckpoints);

        MWCudaLaunchGraph load_ckpt_graph = mwgpu_exec.buildLaunchGraph(
            TaskGraphID::LoadCheckpoints);

        WorldReset *world_reset_buffer = 
            (WorldReset *)mwgpu_exec.getExported((uint32_t)ExportID::Reset);

        Action *agent_actions_buffer = 
            (Action *)mwgpu_exec.getExported((uint32_t)ExportID::Action);

        HostEventLogging(HostEvent::initEnd);
        return new CUDAImpl {
            { 
                cfg,
                max_agents_per_world,
                std::move(phys_loader),
                std::move(render_gpu_state),
                std::move(render_mgr),
                world_reset_buffer,
                agent_actions_buffer,
            },
            std::move(mwgpu_exec),
            std::move(step_graph),
            std::move(save_ckpt_graph),
            std::move(load_ckpt_graph),
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        PhysicsLoader phys_loader(cfg.execMode, 10);
        loadPhysicsObjects(asset_importer, phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
        app_cfg.rigidBodyObjMgr = phys_obj_mgr;

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(cfg, render_gpu_state);

        if (render_mgr.has_value()) {
            loadRenderObjects(asset_importer, *render_mgr);
            app_cfg.renderBridge = render_mgr->bridge();
         } else {
            app_cfg.renderBridge = nullptr;
         }

        HeapArray<WorldInit> world_inits(cfg.numWorlds);

        CPUImpl::TaskGraphT cpu_exec {
            ThreadPoolExecutor::Config {
                .numWorlds = cfg.numWorlds,
                .numExportedBuffers = (uint32_t)ExportID::NumExports,
            },
            app_cfg,
            world_inits.data(),
            (uint32_t)TaskGraphID::NumTaskGraphs,
        };

        WorldReset *world_reset_buffer =
            (WorldReset *)cpu_exec.getExported((uint32_t)ExportID::Reset);

        Action *agent_actions_buffer = 
            (Action *)cpu_exec.getExported((uint32_t)ExportID::Action);

        auto cpu_impl = new CPUImpl {
            { 
                cfg,
                max_agents_per_world,
                std::move(phys_loader),
                std::move(render_gpu_state),
                std::move(render_mgr),
                world_reset_buffer,
                agent_actions_buffer,
            },
            std::move(cpu_exec),
        };

        HostEventLogging(HostEvent::initEnd);

        return cpu_impl;
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

template <EnumType EnumT>
Tensor Manager::Impl::exportStateTensor(EnumT slot,
                                        TensorElementType type,
                                        Span<const int64_t> dimensions)
{
    void *dev_ptr = nullptr;
    Optional<int> gpu_id = Optional<int>::none();
    if (cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_MWGPU_SUPPORT
        dev_ptr =
            static_cast<CUDAImpl *>(this)->mwGPU.getExported((uint32_t)slot);
        gpu_id = cfg.gpuID;
#endif
    } else {
        dev_ptr = static_cast<CPUImpl *>(this)->cpuExec.getExported((uint32_t)slot);
    }

    return Tensor(dev_ptr, type, dimensions, gpu_id);
}

Manager::Manager(const Config &cfg)
    : impl_(Impl::make(cfg))
{}

Manager::~Manager() {
    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_MWGPU_SUPPORT
        delete static_cast<CUDAImpl *>(impl_);
#endif
    } break;
    case ExecMode::CPU : {
        delete static_cast<CPUImpl *>(impl_);
    } break;
    }
}

void Manager::init()
{
    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_MWGPU_SUPPORT
        static_cast<CUDAImpl *>(impl_)->init();
#endif
    } break;
    case ExecMode::CPU: {
        static_cast<CPUImpl *>(impl_)->init();
    } break;
    }

    if (impl_->renderMgr.has_value()) {
        impl_->renderMgr->readECS();
    }

    if (impl_->cfg.enableBatchRenderer) {
        impl_->renderMgr->batchRender();
    }
}

void Manager::step()
{
    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_MWGPU_SUPPORT
        static_cast<CUDAImpl *>(impl_)->step();
#endif
    } break;
    case ExecMode::CPU: {
        static_cast<CPUImpl *>(impl_)->step();
    } break;
    }

    if (impl_->renderMgr.has_value()) {
        impl_->renderMgr->readECS();
    }

    if (impl_->cfg.enableBatchRenderer) {
        impl_->renderMgr->batchRender();
    }
}

void Manager::saveCheckpoint(CountT world_idx)
{
    auto *ctrls =
        (CheckpointControl *)checkpointControlTensor().devicePtr();
    //auto *ckpts =
    //    (Checkpoint *)checkpointTensor().devicePtr();

    CheckpointControl ctrl {
      .trigger = 1,
    };

    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_MWGPU_SUPPORT
        cudaMemcpy(&ctrls[world_idx], &ctrl, sizeof(CheckpointControl),
                   cudaMemcpyHostToDevice);
        static_cast<CUDAImpl *>(impl_)->saveCheckpoints();
#endif
    } break;
    case ExecMode::CPU: {
        ctrls[world_idx] = ctrl;
        static_cast<CPUImpl *>(impl_)->saveCheckpoints();
    } break;
    }
}

void Manager::loadCheckpoint(CountT world_idx)
{
    auto *ctrls =
        (CheckpointControl *)checkpointControlTensor().devicePtr();
    //auto *ckpts =
    //    (Checkpoint *)checkpointTensor().devicePtr();

    CheckpointControl ctrl {
      .trigger = 1,
    };

    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_MWGPU_SUPPORT
        cudaMemcpy(&ctrls[world_idx], &ctrl, sizeof(CheckpointControl),
                   cudaMemcpyHostToDevice);
        static_cast<CUDAImpl *>(impl_)->loadCheckpoints();
#endif
    } break;
    case ExecMode::CPU: {
        ctrls[world_idx] = ctrl;
        static_cast<CPUImpl *>(impl_)->loadCheckpoints();
    } break;
    }

    if (impl_->renderMgr.has_value()) {
        impl_->renderMgr->readECS();
    }

    if (impl_->cfg.enableBatchRenderer) {
        impl_->renderMgr->batchRender();
    }
}

void Manager::loadCheckpoints()
{
    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_MWGPU_SUPPORT
        static_cast<CUDAImpl *>(impl_)->loadCheckpoints();
#endif
    } break;
    case ExecMode::CPU: {
        static_cast<CPUImpl *>(impl_)->loadCheckpoints();
    } break;
    }

    if (impl_->renderMgr.has_value()) {
        impl_->renderMgr->readECS();
    }

    if (impl_->cfg.enableBatchRenderer) {
        impl_->renderMgr->batchRender();
    }
}

#ifdef MADRONA_MWGPU_SUPPORT
void Manager::gpuJAXInit(cudaStream_t strm, void **buffers)
{
    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_MWGPU_SUPPORT
        static_cast<CUDAImpl *>(impl_)->gpuStreamInit(strm, buffers, *this);
#endif
    } break;
    case ExecMode::CPU: {
        static_cast<CPUImpl *>(impl_)->gpuStreamInit(strm, buffers, *this);
    } break;
    }

    if (impl_->renderMgr.has_value()) {
        assert(false);
    }
}

void Manager::gpuJAXStep(cudaStream_t strm, void **buffers)
{
    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_MWGPU_SUPPORT
        static_cast<CUDAImpl *>(impl_)->gpuStreamStep(strm, buffers, *this);
#endif
    } break;
    case ExecMode::CPU: {
        static_cast<CPUImpl *>(impl_)->gpuStreamStep(strm, buffers, *this);
    } break;
    }
    
    if (impl_->renderMgr.has_value()) {
        assert(false);
    }
}

void Manager::gpuJAXSaveCheckpoints(cudaStream_t strm, void **buffers)
{
    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_MWGPU_SUPPORT
        static_cast<CUDAImpl *>(impl_)->gpuJAXSaveCheckpoints(strm, buffers, *this);
#endif
    } break;
    case ExecMode::CPU: {
        assert(false);
    } break;
    }
    
    if (impl_->renderMgr.has_value()) {
        assert(false);
    }
}

void Manager::gpuJAXLoadCheckpoints(cudaStream_t strm, void **buffers)
{
    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_MWGPU_SUPPORT
        static_cast<CUDAImpl *>(impl_)->gpuJAXLoadCheckpoints(strm, buffers, *this);
#endif
    } break;
    case ExecMode::CPU: {
        assert(false);
    } break;
    }
    
    if (impl_->renderMgr.has_value()) {
        assert(false);
    }
}

#endif

Tensor Manager::resetTensor() const
{
    return impl_->exportStateTensor(
        ExportID::Reset, TensorElementType::Int32,
        {impl_->cfg.numWorlds, 1});
}

Tensor Manager::doneTensor() const
{
    return impl_->exportStateTensor(
        ExportID::Done, TensorElementType::Int32,
        {impl_->cfg.numWorlds * impl_->maxAgentsPerWorld, 1});
}

madrona::py::Tensor Manager::prepCounterTensor() const
{
    return impl_->exportStateTensor(
        ExportID::PrepCounter, TensorElementType::Int32,
        {impl_->cfg.numWorlds * impl_->maxAgentsPerWorld, 1});
}

Tensor Manager::actionTensor() const
{
    return impl_->exportStateTensor(
        ExportID::Action, TensorElementType::Int32,
        {impl_->cfg.numWorlds * impl_->maxAgentsPerWorld, 5});
}

Tensor Manager::rewardTensor() const
{
    return impl_->exportStateTensor(
        ExportID::Reward, TensorElementType::Float32,
        {impl_->cfg.numWorlds * impl_->maxAgentsPerWorld, 1});
}


Tensor Manager::selfDataTensor() const
{
    return impl_->exportStateTensor(
        ExportID::SelfObs, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            sizeof(SelfObservation) / sizeof(float),
        });
}

Tensor Manager::selfTypeTensor() const
{
    return impl_->exportStateTensor(
        ExportID::SelfType, TensorElementType::Int32,
        {impl_->cfg.numWorlds * impl_->maxAgentsPerWorld, 1});
}

Tensor Manager::selfMaskTensor() const
{
    return impl_->exportStateTensor(
        ExportID::SelfMask, TensorElementType::Float32,
        {impl_->cfg.numWorlds * impl_->maxAgentsPerWorld, 1});
}


madrona::py::Tensor Manager::agentDataTensor() const
{
    return impl_->exportStateTensor(
        ExportID::AgentObsData, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            consts::maxAgents - 1,
            sizeof(AgentObservation) / sizeof(float),
        });
}

madrona::py::Tensor Manager::boxDataTensor() const
{
    return impl_->exportStateTensor(
        ExportID::BoxObsData, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            consts::maxBoxes,
            sizeof(BoxObservation) / sizeof(float),
        });
}

madrona::py::Tensor Manager::rampDataTensor() const
{
    return impl_->exportStateTensor(
        ExportID::RampObsData, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            consts::maxRamps,
            sizeof(RampObservation) / sizeof(float),
        });
}

madrona::py::Tensor Manager::visibleAgentsMaskTensor() const
{
    return impl_->exportStateTensor(
        ExportID::AgentVisMasks, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            consts::maxAgents - 1,
            1,
        });
}

madrona::py::Tensor Manager::visibleBoxesMaskTensor() const
{
    return impl_->exportStateTensor(
        ExportID::BoxVisMasks, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            consts::maxBoxes,
            1,
        });
}

madrona::py::Tensor Manager::visibleRampsMaskTensor() const
{
    return impl_->exportStateTensor(
        ExportID::RampVisMasks, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            consts::maxRamps,
            1,
        });
}

madrona::py::Tensor Manager::lidarTensor() const
{
    return impl_->exportStateTensor(
        ExportID::Lidar, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            30,
        });
}

madrona::py::Tensor Manager::seedTensor() const
{
    return impl_->exportStateTensor(
        ExportID::Seed, TensorElementType::Int32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            sizeof(Seed) / sizeof(int32_t),
        });
}

madrona::py::Tensor Manager::checkpointControlTensor() const
{
    return impl_->exportStateTensor(
        ExportID::CheckpointControl, TensorElementType::UInt8,
        {
            impl_->cfg.numWorlds,
            sizeof(CheckpointControl),
        });
}

madrona::py::Tensor Manager::checkpointTensor() const
{
    return impl_->exportStateTensor(
        ExportID::Checkpoint, TensorElementType::UInt8,
        {
            impl_->cfg.numWorlds,
            sizeof(Checkpoint),
        });
}

madrona::py::Tensor Manager::globalPositionsTensor() const
{
    return impl_->exportStateTensor(
        ExportID::GlobalDebugPositions, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds,
            consts::maxBoxes + consts::maxRamps +
                consts::maxAgents,
            2,
        });
}

Tensor Manager::depthTensor() const
{
    const float *depth_ptr = impl_->renderMgr->batchRendererDepthOut();

    return Tensor((void *)depth_ptr, TensorElementType::Float32, {
        impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

Tensor Manager::rgbTensor() const
{
    const uint8_t *rgb_ptr = impl_->renderMgr->batchRendererRGBOut();

    return Tensor((void *)rgb_ptr, TensorElementType::UInt8, {
        impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);
}

void Manager::triggerReset(CountT world_idx, CountT level_idx)
{
    WorldReset reset {
        (int32_t)level_idx,
    };

    auto *reset_ptr = impl_->resetsPointer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_MWGPU_SUPPORT
        cudaMemcpy(reset_ptr, &reset, sizeof(WorldReset),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *reset_ptr = reset;
    }
}

void Manager::setAction(CountT agent_idx,
                        int32_t x, int32_t y, int32_t r,
                        bool g, bool l)
{
    Action action { 
        .x = x,
        .y = y,
        .r = r,
        .g = (int32_t)g,
        .l = (int32_t)l,
    };

    auto *action_ptr = impl_->actionsPointer + agent_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_MWGPU_SUPPORT
        cudaMemcpy(action_ptr, &action, sizeof(Action),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

render::RenderManager & Manager::getRenderManager()
{
    return *impl_->renderMgr;
}

Tensor Manager::episodeResultTensor() const
{
    return impl_->exportStateTensor(ExportID::EpisodeResult,
                                    TensorElementType::Float32,
                                    {
                                        impl_->cfg.numWorlds,
                                        sizeof(EpisodeResult) / sizeof(float),
                                    });
}

Tensor Manager::policyAssignmentsTensor() const
{
    return impl_->exportStateTensor(
        ExportID::AgentPolicy,
        TensorElementType::Int32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            sizeof(AgentPolicy) / sizeof(int32_t)
        });
}

Tensor Manager::simControlTensor() const
{
  return Tensor(nullptr, TensorElementType::Int32, {0}, Optional<int>::none());
}

TrainInterface Manager::trainInterface() const
{
    return TrainInterface {
        {
            .actions = { 
                { "actions", actionTensor() },
            },
            .resets = resetTensor(),
            .simCtrl = simControlTensor(),
            .pbt = {
                { "policy_assignments", policyAssignmentsTensor() },
            },
        },
        {
            .observations = {
                { "prep_counter", prepCounterTensor() },
                { "self_data", selfDataTensor() },
                { "self_type", selfTypeTensor() },
                { "self_mask", selfMaskTensor() },
                { "self_lidar", lidarTensor() },
                { "agent_data", agentDataTensor() },
                { "box_data", boxDataTensor() },
                { "ramp_data", rampDataTensor() },
                { "vis_agents_mask", visibleAgentsMaskTensor() },
                { "vis_boxes_mask", visibleBoxesMaskTensor() },
                { "vis_ramps_mask", visibleRampsMaskTensor() },
            },
            .rewards = rewardTensor(),
            .dones = doneTensor(),
            .pbt = {
                { "episode_results", episodeResultTensor() },
            },
        },
        TrainCheckpointingInterface {
            .checkpointData = checkpointTensor(),
        },
    };
}

}
