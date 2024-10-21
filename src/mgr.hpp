#pragma once

#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include <madrona/render/render_mgr.hpp>

#include "sim_flags.hpp"

namespace GPUHideSeek {

class Manager {
public:
    struct Config {
        madrona::ExecMode execMode;
        int gpuID;
        uint32_t numWorlds;
        SimFlags simFlags;
        uint32_t randSeed;
        uint32_t minHiders;
        uint32_t maxHiders;
        uint32_t minSeekers;
        uint32_t maxSeekers;
        uint32_t numPBTPolicies;
        bool enableBatchRenderer;
        uint32_t batchRenderViewWidth = 64;
        uint32_t batchRenderViewHeight = 64;
        madrona::render::APIBackend *extRenderAPI = nullptr;
        madrona::render::GPUDevice *extRenderDev = nullptr;
    };

    Manager(const Config &cfg);
    ~Manager();

    void init();
    void step();

    void saveCheckpoint(madrona::CountT world_idx);
    void loadCheckpoint(madrona::CountT world_idx);
    void loadCheckpoints();

    void cpuStreamInit(void **, void **) {}
    void cpuStreamStep(void **, void **) {}

#ifdef MADRONA_MWGPU_SUPPORT
    void gpuJAXInit(cudaStream_t strm, void **buffers);
    void gpuJAXStep(cudaStream_t strm, void **buffers);

    void gpuJAXSaveCheckpoints(cudaStream_t strm, void **buffers);
    void gpuJAXLoadCheckpoints(cudaStream_t strm, void **buffers);
#endif

    madrona::py::Tensor resetTensor() const;
    madrona::py::Tensor doneTensor() const;
    madrona::py::Tensor prepCounterTensor() const;
    madrona::py::Tensor actionTensor() const;
    madrona::py::Tensor rewardTensor() const;
    madrona::py::Tensor selfDataTensor() const;
    madrona::py::Tensor selfTypeTensor() const;
    madrona::py::Tensor selfMaskTensor() const;
    madrona::py::Tensor agentDataTensor() const;
    madrona::py::Tensor boxDataTensor() const;
    madrona::py::Tensor rampDataTensor() const;
    madrona::py::Tensor visibleAgentsMaskTensor() const;
    madrona::py::Tensor visibleBoxesMaskTensor() const;
    madrona::py::Tensor visibleRampsMaskTensor() const;
    madrona::py::Tensor globalPositionsTensor() const;
    madrona::py::Tensor lidarTensor() const;
    madrona::py::Tensor seedTensor() const;

    madrona::py::Tensor checkpointControlTensor() const;
    madrona::py::Tensor checkpointTensor() const;

    madrona::py::Tensor depthTensor() const;
    madrona::py::Tensor rgbTensor() const;

    void triggerReset(madrona::CountT world_idx,
                      madrona::CountT level_idx);
    void setAction(madrona::CountT agent_idx,
                   int32_t x, int32_t y, int32_t r,
                   bool g, bool l);

    madrona::render::RenderManager & getRenderManager();

    madrona::py::Tensor policyAssignmentsTensor() const;
    madrona::py::Tensor episodeResultTensor() const;
    madrona::py::TrainInterface trainInterface() const;

private:
    struct Impl;
    struct CPUImpl;
    struct CUDAImpl;

    Impl *impl_;
};

}
