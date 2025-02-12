#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "level_gen.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace RenderingSystem = madrona::render::RenderingSystem;

namespace GPUHideSeek {

constexpr inline float deltaT = 1.f / 30.f;
constexpr inline CountT numPhysicsSubsteps = 4;
constexpr inline CountT numPrepSteps = 96;
constexpr inline CountT episodeLen = 240;

constexpr inline auto physicsSolverSelector = PhysicsSystem::Solver::XPBD;

void Sim::registerTypes(ECSRegistry &registry,
                        const Config &cfg)
{
    base::registerTypes(registry);
    PhysicsSystem::registerTypes(registry, physicsSolverSelector);

    RenderingSystem::registerTypes(registry, cfg.renderBridge);

    registry.registerSingleton<CheckpointControl>();
    registry.registerSingleton<Checkpoint>();

    registry.registerComponent<AgentPrepCounter>();
    registry.registerComponent<Action>();
    registry.registerComponent<OwnerTeam>();
    registry.registerComponent<AgentType>();
    registry.registerComponent<GrabData>();

    registry.registerComponent<SimEntity>();

    registry.registerComponent<AgentActiveMask>();
    registry.registerComponent<SelfObservation>();
    registry.registerComponent<RelativeAgentObservations>();
    registry.registerComponent<RelativeBoxObservations>();
    registry.registerComponent<RelativeRampObservations>();
    registry.registerComponent<AgentVisibilityMasks>();
    registry.registerComponent<BoxVisibilityMasks>();
    registry.registerComponent<RampVisibilityMasks>();
    registry.registerComponent<Lidar>();
    registry.registerComponent<Seed>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<AgentPolicy>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<GlobalDebugPositions>();
    registry.registerSingleton<TeamState>();
    registry.registerSingleton<EpisodeStats>();
    registry.registerSingleton<EpisodeResult>();

    registry.registerArchetype<DynamicObject>();
    registry.registerArchetype<AgentInterface>();
    registry.registerArchetype<DynAgent>();

    registry.exportSingleton<WorldReset>(
        ExportID::Reset);
    registry.exportColumn<AgentInterface, AgentPrepCounter>(
        ExportID::PrepCounter);
    registry.exportColumn<AgentInterface, Action>(
        ExportID::Action);
    registry.exportColumn<AgentInterface, SelfObservation>(
        ExportID::SelfObs);
    registry.exportColumn<AgentInterface, AgentType>(
        ExportID::SelfType);
    registry.exportColumn<AgentInterface, AgentActiveMask>(
        ExportID::SelfMask);
    registry.exportColumn<AgentInterface, RelativeAgentObservations>(
        ExportID::AgentObsData);
    registry.exportColumn<AgentInterface, RelativeBoxObservations>(
        ExportID::BoxObsData);
    registry.exportColumn<AgentInterface, RelativeRampObservations>(
        ExportID::RampObsData);
    registry.exportColumn<AgentInterface, AgentVisibilityMasks>(
        ExportID::AgentVisMasks);
    registry.exportColumn<AgentInterface, BoxVisibilityMasks>(
        ExportID::BoxVisMasks);
    registry.exportColumn<AgentInterface, RampVisibilityMasks>(
        ExportID::RampVisMasks);
    registry.exportColumn<AgentInterface, AgentPolicy>(
        ExportID::AgentPolicy);
    registry.exportColumn<AgentInterface, Lidar>(ExportID::Lidar);
    registry.exportColumn<AgentInterface, Seed>(ExportID::Seed);
    registry.exportColumn<AgentInterface, Reward>(ExportID::Reward);
    registry.exportColumn<AgentInterface, Done>(ExportID::Done);
    registry.exportSingleton<GlobalDebugPositions>(
        ExportID::GlobalDebugPositions);
    registry.exportSingleton<EpisodeResult>(
        ExportID::EpisodeResult);

    registry.exportSingleton<CheckpointControl>(
        ExportID::CheckpointControl);
    registry.exportSingleton<Checkpoint>(
        ExportID::Checkpoint);
}

static void initEpisodeRNG(Engine &ctx)
{
    RandKey new_rnd_counter = {
        .a = ctx.data().curWorldEpisode++,
        .b = (uint32_t)ctx.worldID().idx,
    };
    ctx.data().curEpisodeRNDCounter = new_rnd_counter;
    ctx.data().rng = RNG(rand::split_i(ctx.data().initRandKey,
        new_rnd_counter.a, new_rnd_counter.b));
}

static inline void resetEnvironment(Engine &ctx, bool update_rng)
{
    ctx.data().curEpisodeStep = 0;

    phys::PhysicsSystem::reset(ctx);

    Entity *all_entities = ctx.data().obstacles;
    for (CountT i = 0; i < ctx.data().numObstacles; i++) {
        Entity e = all_entities[i];
        ctx.destroyRenderableEntity(e);
    }
    ctx.data().numObstacles = 0;
    ctx.data().numActiveBoxes = 0;
    ctx.data().numActiveRamps = 0;

    auto destroyAgent = [&](Entity e) {
        auto grab_data = ctx.getSafe<GrabData>(e);

        if (grab_data.valid()) {
            auto constraint_entity = grab_data.value().constraintEntity;
            if (constraint_entity != Entity::none()) {
                ctx.destroyEntity(constraint_entity);
            }
        }

        ctx.destroyRenderableEntity(e);
    };

    for (CountT i = 0; i < ctx.data().numHiders; i++) {
        destroyAgent(ctx.data().hiders[i]);
    }
    ctx.data().numHiders = 0;

    for (CountT i = 0; i < ctx.data().numSeekers; i++) {
        destroyAgent(ctx.data().seekers[i]);
    }
    ctx.data().numSeekers = 0;

    ctx.data().numActiveAgents = 0;

    if (update_rng) {
      initEpisodeRNG(ctx);
    }
}

static RandKey levelGenRandKey(Engine &ctx)
{
    RandKey lvl_rnd_key = ctx.data().rng.randKey();
    if ((ctx.data().simFlags & SimFlags::UseFixedWorld) ==
            SimFlags::UseFixedWorld) {
        lvl_rnd_key = { 0, 0 };
    } 

    return lvl_rnd_key;
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    int32_t level = reset.resetLevel;

    if ((ctx.data().simFlags & SimFlags::IgnoreEpisodeLength) !=
                SimFlags::IgnoreEpisodeLength &&
            ctx.data().curEpisodeStep == episodeLen - 1) {
        level = 1;
    }

    if (level != 0) {
        resetEnvironment(ctx, true);

        reset.resetLevel = 0;

        int32_t num_hiders = ctx.data().rng.sampleI32(
            ctx.data().minHiders, ctx.data().maxHiders + 1);
        int32_t num_seekers = ctx.data().rng.sampleI32(
            ctx.data().minSeekers, ctx.data().maxSeekers + 1);

        
        generateEnvironment(ctx, levelGenRandKey(ctx), level,
                            num_hiders, num_seekers);
    } else {
        ctx.data().curEpisodeStep += 1;
    }

    ctx.data().hiderTeamReward.store_relaxed(1.f);
}

inline void movementSystem(Engine &ctx, Action &action, SimEntity sim_e,
                                 AgentType agent_type)
{
    if (sim_e.e == Entity::none()) return;
    if (agent_type == AgentType::Seeker &&
            ctx.data().curEpisodeStep < numPrepSteps - 1) {
        return;
    }

    constexpr CountT discrete_action_buckets = 11;
    constexpr CountT half_buckets = discrete_action_buckets / 2;
    constexpr float move_discrete_action_max = 60;
    constexpr float move_delta_per_bucket = move_discrete_action_max / half_buckets;

    constexpr float turn_discrete_action_max = 15;
    constexpr float turn_delta_per_bucket = turn_discrete_action_max / half_buckets;

    Quat cur_rot = ctx.get<Rotation>(sim_e.e);

    float f_x = move_delta_per_bucket * (action.x - 5);
    float f_y = move_delta_per_bucket * (action.y - 5);
    float t_z = turn_delta_per_bucket * (action.r - 5);

    ctx.get<ExternalForce>(sim_e.e) = cur_rot.rotateVec({ f_x, f_y, 0 });
    ctx.get<ExternalTorque>(sim_e.e) = Vector3 { 0, 0, t_z };
}

inline void instantMovementSystem(Engine &ctx, Action &action, SimEntity sim_e,
                                  AgentType agent_type)
{
    if (sim_e.e == Entity::none()) return;
    if (agent_type == AgentType::Seeker &&
            ctx.data().curEpisodeStep < numPrepSteps - 1) {
        return;
    }

    constexpr CountT discrete_action_buckets = 5;
    constexpr CountT half_buckets = discrete_action_buckets / 2;
    constexpr float move_discrete_action_max = 800;
    constexpr float move_delta_per_bucket = move_discrete_action_max / half_buckets;

    constexpr float turn_discrete_action_max = 240;
    constexpr float turn_delta_per_bucket = turn_discrete_action_max / half_buckets;

    Quat cur_rot = ctx.get<Rotation>(sim_e.e);

    float f_x = move_delta_per_bucket * (action.x - 2);
    float f_y = move_delta_per_bucket * (action.y - 2);
    float t_z = turn_delta_per_bucket * (action.r - 2);

    ctx.get<ExternalForce>(sim_e.e) = cur_rot.rotateVec({ f_x, f_y, 0 });
    ctx.get<ExternalTorque>(sim_e.e) = Vector3 { 0, 0, t_z };
}

// Make the agents easier to control by zeroing out their velocity
// after each step.
inline void agentZeroVelSystem(Engine &,
                               Velocity &vel,
                               GrabData &)
{
    vel.linear.x = 0;
    vel.linear.y = 0;
    vel.linear.z = fminf(vel.linear.z, 0);

    vel.angular = Vector3::zero();
    return;
}

inline void actionSystem(Engine &ctx,
                         Action &action,
                         SimEntity sim_e,
                         AgentType agent_type)
{
    if (sim_e.e == Entity::none()) return;
    if (agent_type == AgentType::Seeker &&
            ctx.data().curEpisodeStep < numPrepSteps - 1) {
        return;
    }

    if (action.l == 1) {
        Vector3 cur_pos = ctx.get<Position>(sim_e.e);
        Quat cur_rot = ctx.get<Rotation>(sim_e.e);

        auto &bvh = ctx.singleton<broadphase::BVH>();
        float hit_t;
        Vector3 hit_normal;
        Entity lock_entity = bvh.traceRay(cur_pos + 0.5f * math::up,
            cur_rot.rotateVec(math::fwd), &hit_t, &hit_normal, 2.5f);

        if (lock_entity != Entity::none()) {
            auto &owner = ctx.get<OwnerTeam>(lock_entity);
            auto &response_type = ctx.get<ResponseType>(lock_entity);

            if (response_type == ResponseType::Static) {
                if ((agent_type == AgentType::Seeker &&
                        owner == OwnerTeam::Seeker) ||
                        (agent_type == AgentType::Hider &&
                         owner == OwnerTeam::Hider)) {
                    response_type = ResponseType::Dynamic;
                    owner = OwnerTeam::None;
                }
            } else {
                if (owner == OwnerTeam::None) {
                    response_type = ResponseType::Static;
                    owner = agent_type == AgentType::Hider ?
                        OwnerTeam::Hider : OwnerTeam::Seeker;
                }
            }
        }
    }

    if (action.g == 1) {
        Vector3 cur_pos = ctx.get<Position>(sim_e.e);
        Quat cur_rot = ctx.get<Rotation>(sim_e.e);

        auto &grab_data = ctx.get<GrabData>(sim_e.e);

        if (grab_data.constraintEntity != Entity::none()) {
            ctx.destroyEntity(grab_data.constraintEntity);
            grab_data.constraintEntity = Entity::none();
        } else {
            auto &bvh = ctx.singleton<broadphase::BVH>();
            float hit_t;
            Vector3 hit_normal;

            Vector3 ray_o = cur_pos + 0.5f * math::up;
            Vector3 ray_d = cur_rot.rotateVec(math::fwd);

            Entity grab_entity =
                bvh.traceRay(ray_o, ray_d, &hit_t, &hit_normal, 2.5f);

            if (grab_entity != Entity::none()) {
                auto &owner = ctx.get<OwnerTeam>(grab_entity);
                auto &response_type = ctx.get<ResponseType>(grab_entity);

                if (owner == OwnerTeam::None &&
                    response_type == ResponseType::Dynamic) {

                    Vector3 other_pos = ctx.get<Position>(grab_entity);
                    Quat other_rot = ctx.get<Rotation>(grab_entity);

                    Vector3 r1 = 1.25f * math::fwd + 0.5f * math::up;

                    Vector3 hit_pos = ray_o + ray_d * hit_t;
                    Vector3 r2 =
                        other_rot.inv().rotateVec(hit_pos - other_pos);

                    Quat attach1 = { 1, 0, 0, 0 };
                    Quat attach2 = (other_rot.inv() * cur_rot).normalize();

                    float separation = hit_t - 1.25f;

                    grab_data.constraintEntity = PhysicsSystem::makeFixedJoint(
                        ctx, sim_e.e, grab_entity, attach1, attach2,
                        r1, r2, separation);

                }
            }
        }
    }

    // "Consume" the actions. This isn't strictly necessary but
    // allows step to be called without every agent having acted
    action.x = 2;
    action.y = 2;
    action.r = 2;
    action.g = 0;
    action.l = 0;
}

static inline Vector3 quatToEuler(Quat q)
{
    // Roll (x-axis rotation)
    float sinr_cosp = 2.f * (q.w * q.x + q.y * q.z);
    float cosr_cosp = 1.f - 2.f * (q.x * q.x + q.y * q.y);
    float roll = atan2f(sinr_cosp, cosr_cosp);

    // Pitch (y-axis rotation)
    float sinp = 2.f * (q.w * q.y - q.z * q.x);
    float pitch;
    if (std::abs(sinp) >= 1.f) {
        // use 90 degrees if out of range
        pitch = copysignf(math::pi / 2.f, sinp);
    } else {
        pitch = asinf(sinp);
    }

    // Yaw (z-axis rotation)
    float siny_cosp = 2.f * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.f - 2.f * (q.y * q.y + q.z * q.z);
    float yaw = atan2f(siny_cosp, cosy_cosp);

    return {
        .x = roll,
        .y = pitch,
        .z = yaw,
    };
}

static inline PosVelObservation computeRelativePosVelObs(
    Vector3 frame_origin, Quat to_frame, Velocity frame_vel, 
    Vector3 x, Quat q, Velocity vel)
{
    x -= frame_origin;
    x = to_frame.rotateVec(x);

    q = to_frame * q;
    q = q.normalize();

    vel.linear -= frame_vel.linear;
    vel.angular -= frame_vel.angular;

    return PosVelObservation {
        .pos = x,
        .rotEuler = quatToEuler(q),
        .linearVel = to_frame.rotateVec(vel.linear),
        .angularVel = to_frame.rotateVec(vel.angular),
    };
}

static inline LockObservation computeLockObservation(Engine &ctx, Entity e)
{
    auto response_type = ctx.get<ResponseType>(e);

    if (response_type != ResponseType::Static) {
        return {
            .hiderLocked = 0.f,
            .seekerLocked = 0.f,
        };
    }

    auto owner = ctx.get<OwnerTeam>(e);

    if (owner == OwnerTeam::Hider) {
        return {
            .hiderLocked = 1.f,
            .seekerLocked = 0.f,
        };
    } else {
        return {
            .hiderLocked = 0.f,
            .seekerLocked = 1.f,
        };
    }
}

inline void collectObservationsSystem(Engine &ctx,
                                      Entity agent_e,
                                      SimEntity sim_e,
                                      SelfObservation &self_obs,
                                      RelativeAgentObservations &agent_obs,
                                      RelativeBoxObservations &box_obs,
                                      RelativeRampObservations &ramp_obs,
                                      AgentPrepCounter &prep_counter)
{
    if (sim_e.e == Entity::none()) {
        return;
    }

    CountT cur_step = ctx.data().curEpisodeStep;
    if (cur_step <= numPrepSteps) {
        prep_counter.numPrepStepsLeft = numPrepSteps - cur_step;
    } 

    Vector3 agent_pos = ctx.get<Position>(sim_e.e);
    Velocity agent_vel = ctx.get<Velocity>(sim_e.e);
    Quat agent_rot = ctx.get<Rotation>(sim_e.e);
    Quat to_agent_frame = agent_rot.inv();

    {
        bool is_grabbing =
            ctx.get<GrabData>(sim_e.e).constraintEntity != Entity::none();

        self_obs.posVel = PosVelObservation {
            .pos = agent_pos,
            .rotEuler = quatToEuler(agent_rot),
            .linearVel = to_agent_frame.rotateVec(agent_vel.linear),
            .angularVel = to_agent_frame.rotateVec(agent_vel.angular),
        };

        self_obs.isGrabbing = is_grabbing;
    }

    CountT num_boxes = ctx.data().numActiveBoxes;
    for (CountT box_idx = 0; box_idx < consts::maxBoxes; box_idx++) {
        auto &obs = box_obs.obs[box_idx];

        if (box_idx >= num_boxes) {
            obs= {};
            continue;
        }

        Entity box_e = ctx.data().boxes[box_idx];

        Vector3 box_pos = ctx.get<Position>(box_e);
        Quat box_rot = ctx.get<Rotation>(box_e);
        Velocity box_vel = ctx.get<Velocity>(box_e);

        obs.posVel = computeRelativePosVelObs(
            agent_pos, to_agent_frame, agent_vel, box_pos, box_rot, box_vel);

        obs.boxSize = ctx.data().boxSizes[box_idx];
        obs.locked = computeLockObservation(ctx, box_e);
    }

    CountT num_ramps = ctx.data().numActiveRamps;
    for (CountT ramp_idx = 0; ramp_idx < consts::maxRamps; ramp_idx++) {
        auto &obs = ramp_obs.obs[ramp_idx];

        if (ramp_idx >= num_ramps) {
            obs = {};
            continue;
        }

        Entity ramp_e = ctx.data().ramps[ramp_idx];

        Vector3 ramp_pos = ctx.get<Position>(ramp_e);
        Quat ramp_rot = ctx.get<Rotation>(ramp_e);
        Velocity ramp_vel = ctx.get<Velocity>(ramp_e);

        obs.posVel = computeRelativePosVelObs(
            agent_pos, to_agent_frame, agent_vel, ramp_pos, ramp_rot, ramp_vel);
        obs.locked = computeLockObservation(ctx, ramp_e);
    }

    CountT num_agents = ctx.data().numActiveAgents;
    CountT num_other_agents = 0;
    for (CountT agent_idx = 0; agent_idx < consts::maxAgents; agent_idx++) {
        if (agent_idx >= num_agents) {
            agent_obs.obs[num_other_agents++] = {};
            continue;
        }

        Entity other_agent_e = ctx.data().agentInterfaces[agent_idx];
        if (agent_e == other_agent_e) {
            continue;
        }

        Entity other_agent_sim_e = ctx.get<SimEntity>(other_agent_e).e;

        auto &obs = agent_obs.obs[num_other_agents++];

        Vector3 other_agent_pos =
            ctx.get<Position>(other_agent_sim_e);
        Quat other_agent_rot =
            ctx.get<Rotation>(other_agent_sim_e);
        Velocity other_agent_vel =
            ctx.get<Velocity>(other_agent_sim_e);

        obs.posVel = computeRelativePosVelObs(
            agent_pos, to_agent_frame, agent_vel, 
            other_agent_pos, other_agent_rot, other_agent_vel);

        AgentType other_agent_type = ctx.get<AgentType>(other_agent_e);

        obs.isHider = other_agent_type == AgentType::Hider ? 1.f : 0.f;

        bool is_grabbing =
            ctx.get<GrabData>(other_agent_sim_e).constraintEntity !=
                Entity::none();

        obs.isGrabbing = is_grabbing;
    }
}

inline void computeVisibilitySystem(Engine &ctx,
                                    Entity agent_e,
                                    SimEntity sim_e,
                                    AgentType agent_type,
                                    AgentVisibilityMasks &agent_vis,
                                    BoxVisibilityMasks &box_vis,
                                    RampVisibilityMasks &ramp_vis)
{
    if (sim_e.e == Entity::none()) {
        return;
    }

    Vector3 agent_pos = ctx.get<Position>(sim_e.e);
    Quat agent_rot = ctx.get<Rotation>(sim_e.e);
    Vector3 agent_fwd = agent_rot.rotateVec(math::fwd);
    const float cos_angle_threshold = cosf(toRadians(135.f / 2.f));

    auto &bvh = ctx.singleton<broadphase::BVH>();

    auto checkVisibility = [&](Entity other_e) {
        Vector3 other_pos = ctx.get<Position>(other_e);

        Vector3 to_other = other_pos - agent_pos;

        Vector3 to_other_norm = to_other.normalize();

        float cos_angle = dot(to_other_norm, agent_fwd);

        if (cos_angle < cos_angle_threshold) {
            return 0.f;
        }

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(agent_pos, to_other, &hit_t, &hit_normal, 1.f);

        return hit_entity == other_e ? 1.f : 0.f;
    };

#ifdef MADRONA_GPU_MODE
    constexpr int32_t num_total_vis =
        consts::maxBoxes + consts::maxRamps + consts::maxAgents;
    const int32_t lane_id = threadIdx.x % 32;
    for (int32_t global_offset = 0; global_offset < num_total_vis;
         global_offset += 32) {
        int32_t cur_idx = global_offset + lane_id;

        Entity check_e = Entity::none();
        float *vis_out = nullptr;

        bool checking_agent = cur_idx < consts::maxAgents;
        uint32_t agent_mask = __ballot_sync(mwGPU::allActive, checking_agent);
        if (checking_agent) {
            bool valid_check = true;
            if (cur_idx < ctx.data().numActiveAgents) {
                Entity other_agent_e = ctx.data().agentInterfaces[cur_idx];
                valid_check = other_agent_e != agent_e;

                if (valid_check) {
                    check_e = ctx.get<SimEntity>(other_agent_e).e;
                }
            }

            uint32_t valid_mask = __ballot_sync(agent_mask, valid_check);
            valid_mask <<= (32 - lane_id);
            uint32_t num_lower_valid = __popc(valid_mask);

            if (valid_check) {
                vis_out = &agent_vis.visible[num_lower_valid];
            }
        } else if (int32_t box_idx = cur_idx - consts::maxAgents;
                   box_idx < consts::maxBoxes) {
            if (cur_idx < ctx.data().numActiveBoxes) {
                check_e = ctx.data().boxes[cur_idx];
            }
            vis_out = &box_vis.visible[cur_idx];
        } else if (int32_t ramp_idx =
                       cur_idx - consts::maxAgents - consts::maxBoxes;
                   ramp_idx < consts::maxRamps) {
            if (ramp_idx < ctx.data().numActiveRamps) {
                check_e = ctx.data().ramps[ramp_idx];
            }
            vis_out = &ramp_vis.visible[ramp_idx];
        } 

        if (check_e == Entity::none()) {
           if (vis_out != nullptr) {
               *vis_out = 0.f;
           }
        } else {
            bool is_visible = checkVisibility(check_e);
            *vis_out = is_visible ? 1.f : 0.f;
        }
    }
#else
    CountT num_boxes = ctx.data().numActiveBoxes;
    for (CountT box_idx = 0; box_idx < consts::maxBoxes; box_idx++) {
        if (box_idx < num_boxes) {
            Entity box_e = ctx.data().boxes[box_idx];
            box_vis.visible[box_idx] = checkVisibility(box_e);
        } else {
            box_vis.visible[box_idx] = 0.f;
        }
    }

    CountT num_ramps = ctx.data().numActiveRamps;
    for (CountT ramp_idx = 0; ramp_idx < consts::maxRamps; ramp_idx++) {
        if (ramp_idx < num_ramps) {
            Entity ramp_e = ctx.data().ramps[ramp_idx];
            ramp_vis.visible[ramp_idx] = checkVisibility(ramp_e);
        } else {
            ramp_vis.visible[ramp_idx] = 0.f;
        }
    }

    CountT num_agents = ctx.data().numActiveAgents;
    CountT num_other_agents = 0;
    for (CountT agent_idx = 0; agent_idx < consts::maxAgents; agent_idx++) {
        if (agent_idx >= num_agents) {
            agent_vis.visible[num_other_agents++] = 0.f;
            continue;
        }

        Entity other_agent_e = ctx.data().agentInterfaces[agent_idx];
        if (agent_e == other_agent_e) {
            continue;
        }

        Entity other_agent_sim_e = ctx.get<SimEntity>(other_agent_e).e;

        bool is_visible = checkVisibility(other_agent_sim_e);

        if (agent_type == AgentType::Seeker && is_visible) {
            AgentType other_type = ctx.get<AgentType>(other_agent_e);
            if (other_type == AgentType::Hider) {
                ctx.data().hiderTeamReward.store_relaxed(-1.f);
            }
        }

        agent_vis.visible[num_other_agents++] = is_visible;
    }
#endif
}

inline void lidarSystem(Engine &ctx,
                        SimEntity sim_e,
                        Lidar &lidar)
{
    if (sim_e.e == Entity::none()) {
        return;
    }

    Vector3 pos = ctx.get<Position>(sim_e.e);
    Quat rot = ctx.get<Rotation>(sim_e.e);
    auto &bvh = ctx.singleton<broadphase::BVH>();

    Vector3 agent_fwd = rot.rotateVec(math::fwd);
    Vector3 right = rot.rotateVec(math::right);

    auto traceRay = [&](int32_t idx) {
        float theta = 2.f * math::pi * (float(idx) / float(30)) +
            math::pi / 2.f;
        float x = cosf(theta);
        float y = sinf(theta);

        Vector3 ray_dir = (x * right + y * agent_fwd).normalize();

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(pos, ray_dir, &hit_t, &hit_normal, 200.f);

        if (hit_entity == Entity::none()) {
            lidar.depth[idx] = 0.f;
        } else {
            lidar.depth[idx] = hit_t;
        }
    };


#ifdef MADRONA_GPU_MODE
    int32_t idx = threadIdx.x % 32;

    if (idx < 30) {
        traceRay(idx);
    }
#else
    for (int32_t i = 0; i < 30; i++) {
        traceRay(i);
    }
#endif
}

// FIXME: refactor this so the observation systems can reuse these raycasts
// (unless a reset has occurred)
inline void rewardsVisSystem(Engine &ctx,
                             SimEntity sim_e,
                             AgentType agent_type)
{
    const float cos_angle_threshold = cosf(toRadians(135.f / 2.f));

    if (sim_e.e == Entity::none() || agent_type != AgentType::Seeker) {
        return;
    }

    auto &bvh = ctx.singleton<broadphase::BVH>();

    Vector3 seeker_pos = ctx.get<Position>(sim_e.e);
    Quat seeker_rot = ctx.get<Rotation>(sim_e.e);
    Vector3 seeker_fwd = seeker_rot.rotateVec(math::fwd);

    for (CountT i = 0; i < ctx.data().numHiders; i++) {
        Entity hider_sim_e = ctx.data().hiders[i];

        Vector3 hider_pos = ctx.get<Position>(hider_sim_e);

        Vector3 to_hider = hider_pos - seeker_pos;

        Vector3 to_hider_norm = to_hider.normalize();

        float cos_angle = dot(to_hider_norm, seeker_fwd);

        if (cos_angle < cos_angle_threshold) {
            continue;
        }

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(seeker_pos, to_hider, &hit_t, &hit_normal, 1.f);

        if (hit_entity == hider_sim_e) {
            ctx.data().hiderTeamReward.store_relaxed(-1);
            break;
        }
    }
}

inline void outputRewardsDonesSystem(Engine &ctx,
                                     SimEntity sim_e,
                                     AgentType agent_type,
                                     Reward &reward,
                                     Done &done)
{
    if (sim_e.e == Entity::none()) {
        return;
    }

    CountT cur_step = ctx.data().curEpisodeStep;

    if (cur_step == 0) {
        done.v = 0;
    }

    if (cur_step < numPrepSteps - 1) {
        reward.v = 0.f;
        return;
    } else if (cur_step == episodeLen - 1) {
        done.v = 1;
    }

    float reward_val = ctx.data().hiderTeamReward.load_relaxed();
    if (agent_type == AgentType::Seeker) {
        reward_val *= -1.f;
    }

    Vector3 pos = ctx.get<Position>(sim_e.e);

    if (fabsf(pos.x) >= 18.f || fabsf(pos.y) >= 18.f) {
        reward_val -= 10.f;
    }

    reward.v = reward_val;
}

inline void updateEpisodeResultsSystem(Engine &ctx,
                                       EpisodeResult &ep_result)
{
    auto &stats = ctx.singleton<EpisodeStats>();

    CountT cur_step = ctx.data().curEpisodeStep;

    if (cur_step == 0) {
        ep_result.finishedScores[0] = 0.f;
        ep_result.finishedScores[1] = 0.f;

        stats.runningScores[0] = 0;
        stats.runningScores[1] = 0;
    }

    // FIXME, don't rely on hiderTeamReward here.

    if (cur_step >= numPrepSteps) {
        bool seekers_first = ctx.singleton<TeamState>().seekersFirst;

        bool hiders_hidden =
            ctx.data().hiderTeamReward.load_relaxed() == 1.f;

        if (hiders_hidden) {
            if (seekers_first) {
                stats.runningScores[1] += 1;
            } else {
                stats.runningScores[0] += 1;
            }
        } else {
            if (seekers_first) {
                stats.runningScores[0] += 1;
            } else {
                stats.runningScores[1] += 1;
            }
        }
    }

    if (cur_step == episodeLen - 1) {
        if (stats.runningScores[0] > stats.runningScores[1]) {
            ep_result.finishedScores[0] = 1.f;
            ep_result.finishedScores[1] = 0.f;
        } else if (stats.runningScores[0] < stats.runningScores[1]) {
            ep_result.finishedScores[0] = 0.f;
            ep_result.finishedScores[1] = 1.f;
        } else {
            ep_result.finishedScores[0] = 0.5f;
            ep_result.finishedScores[1] = 0.5f;
        }
    }
}

inline void globalPositionsDebugSystem(Engine &ctx,
                                       GlobalDebugPositions &global_positions)
{
    auto getXY = [](Vector3 v) {
        return Vector2 {
            v.x,
            v.y,
        };
    };

    for (CountT i = 0; i < consts::maxBoxes; i++) {
        if (i >= ctx.data().numActiveBoxes) {
            global_positions.boxPositions[i] = Vector2 {0, 0};
            continue;
        }

        global_positions.boxPositions[i] =
            getXY(ctx.get<Position>(ctx.data().boxes[i]));
    }

    for (CountT i = 0; i < consts::maxRamps; i++) {
        if (i >= ctx.data().numActiveRamps) {
            global_positions.rampPositions[i] = Vector2 {0, 0};
            continue;
        }

        global_positions.rampPositions[i] =
            getXY(ctx.get<Position>(ctx.data().ramps[i]));
    }

    {
        CountT out_offset = 0;
        for (CountT i = 0; i < ctx.data().numHiders; i++) {
            global_positions.agentPositions[out_offset++] = 
                getXY(ctx.get<Position>(ctx.data().hiders[i]));
        }

        for (CountT i = 0; i < ctx.data().numSeekers; i++) {
            global_positions.agentPositions[out_offset++] = 
                getXY(ctx.get<Position>(ctx.data().seekers[i]));
        }

        for (; out_offset < consts::maxAgents; out_offset++) {
            global_positions.agentPositions[out_offset++] = Vector2 {0, 0};
        }
    }
}

inline void updateCameraSystem(Engine &ctx,
                               Position &pos,
                               Rotation &rot,
                               SimEntity sim_e)
{
    if (sim_e.e == Entity::none()) {
        return;
    }

    pos = ctx.get<Position>(sim_e.e);
    rot = ctx.get<Rotation>(sim_e.e);
}

inline void loadCheckpointSystem(Engine &ctx,
                                 CheckpointControl &ckpt_ctrl)
{
    if (!ckpt_ctrl.trigger) {
        return;
    }

    ckpt_ctrl.trigger = 1;

    Checkpoint &ckpt = ctx.singleton<Checkpoint>();

    resetEnvironment(ctx, false);

    ctx.data().curEpisodeRNDCounter = ckpt.episodeRNDKey;
    ctx.data().rng = RNG(rand::split_i(ctx.data().initRandKey,
        ckpt.episodeRNDKey.a, ckpt.episodeRNDKey.b));

    ctx.singleton<EpisodeStats>() = ckpt.episodeStats;
    ctx.data().curEpisodeStep = ckpt.episodeStep;

    // HACK, need to burn RNG state to get same result in generateEnv
    ctx.data().rng.sampleI32(
        ctx.data().minHiders, ctx.data().maxHiders + 1);
    ctx.data().rng.sampleI32(
        ctx.data().minSeekers, ctx.data().maxSeekers + 1);

    generateEnvironment(
      ctx, levelGenRandKey(ctx), 1, ckpt.numHiders, ckpt.numSeekers);

    auto loadObjCkpt =
      [&]
    (Entity e, Checkpoint::DynObjectState &saved)
    {
      ctx.get<Position>(e) = saved.pos;
      ctx.get<Rotation>(e) = saved.rot;
      ctx.get<Velocity>(e) = saved.vel;
      ctx.get<OwnerTeam>(e) = saved.team;
      if (saved.isLocked) {
        ctx.get<ResponseType>(e) = ResponseType::Static;
      } else {
        ctx.get<ResponseType>(e) = ResponseType::Dynamic;
      }
    };

    assert(ctx.data().numActiveBoxes == ckpt.numBoxes);
    for (i32 i = 0; i < ckpt.numBoxes; i++) {
      loadObjCkpt(ctx.data().boxes[i], ckpt.boxes[i]);
    }

    assert(ctx.data().numActiveRamps == ckpt.numRamps);
    for (i32 i = 0; i < ckpt.numRamps; i++) {
      loadObjCkpt(ctx.data().ramps[i], ckpt.ramps[i]);
    }

    auto loadAgentCkpt =
      [&]
    (Entity e, Checkpoint::AgentState &saved)
    {
      ctx.get<Position>(e) = saved.pos;
      ctx.get<Rotation>(e) = saved.rot;
      ctx.get<Velocity>(e) = saved.vel;

      auto &grab_data = ctx.get<GrabData>(e);
      if (saved.grabIdx == -1) {
        grab_data.constraintEntity = Entity::none();
      } else {
        Entity other;
        assert(saved.grabIdx < ckpt.numBoxes + ckpt.numRamps);
        if (saved.grabIdx < ckpt.numBoxes) {
          other = ctx.data().boxes[saved.grabIdx];
        } else {
          other = ctx.data().ramps[saved.grabIdx - ckpt.numBoxes];
        }

        grab_data.constraintEntity = PhysicsSystem::makeFixedJoint(
          ctx, e, other,
          saved.grabFixed.attachRot1,
          saved.grabFixed.attachRot2,
          saved.grabR1, saved.grabR2, saved.grabFixed.separation);
      }
    };

    for (i32 i = 0; i < ckpt.numHiders; i++) {
      loadAgentCkpt(ctx.data().hiders[i], ckpt.agents[i]);
    }
    for (i32 i = 0; i < ckpt.numSeekers; i++) {
      loadAgentCkpt(ctx.data().seekers[i], ckpt.agents[i + ckpt.numHiders]);
    }
}

inline void saveCheckpointSystem(Engine &ctx,
                                 CheckpointControl &ctrl)
{
  if (!ctrl.trigger) {
    return;
  }
  ctrl.trigger = 0;

  Checkpoint &ckpt = ctx.singleton<Checkpoint>();

  ckpt.episodeRNDKey = ctx.data().curEpisodeRNDCounter;
  ckpt.episodeStats = ctx.singleton<EpisodeStats>();
  ckpt.episodeStep = ctx.data().curEpisodeStep;

  i32 cur_agent_ckpt_idx = 0;

  auto ckptAgent =
    [&]
  (Entity agent)
  {
    Checkpoint::AgentState &agent_ckpt = ckpt.agents[cur_agent_ckpt_idx++];
    agent_ckpt.pos = ctx.get<Position>(agent);
    agent_ckpt.rot = ctx.get<Rotation>(agent);
    agent_ckpt.vel = ctx.get<Velocity>(agent);

    agent_ckpt.grabIdx = -1;
    agent_ckpt.grabR1 = {};
    agent_ckpt.grabR2 = {};
    agent_ckpt.grabFixed = {};

    GrabData &grab = ctx.get<GrabData>(agent);
    if (grab.constraintEntity != Entity::none()) {
      JointConstraint joint = ctx.get<JointConstraint>(grab.constraintEntity);
      agent_ckpt.grabR1 = joint.r1;
      agent_ckpt.grabR2 = joint.r2;
      agent_ckpt.grabFixed = joint.fixed;

      for (i32 j = 0; j < ctx.data().numActiveBoxes; j++) {
        if (ctx.data().boxes[j] == joint.e2) {
          agent_ckpt.grabIdx = j;
        }
      }

      for (i32 j = 0; j < ctx.data().numActiveRamps; j++) {
        if (ctx.data().ramps[j] == joint.e2) {
          agent_ckpt.grabIdx = j + ctx.data().numActiveBoxes;
        }
      }

      assert(agent_ckpt.grabIdx != -1);
    }
  };

  ckpt.numHiders = ctx.data().numHiders;
  for (i32 i = 0; i < ckpt.numHiders; i++) {
    Entity hider = ctx.data().hiders[i];
    ckptAgent(hider);
  }

  assert(cur_agent_ckpt_idx <= consts::maxAgents);

  ckpt.numSeekers = ctx.data().numSeekers;
  for (i32 i = 0; i < ckpt.numSeekers; i++) {
    Entity seeker = ctx.data().seekers[i];
    ckptAgent(seeker);
  }

  assert(cur_agent_ckpt_idx <= consts::maxAgents);

  auto ckptObject =
    [&]
  (Entity e, Checkpoint::DynObjectState *out)
  {
    out->pos = ctx.get<Position>(e);
    out->rot = ctx.get<Rotation>(e);
    out->vel = ctx.get<Velocity>(e);
    out->team = ctx.get<OwnerTeam>(e);
    out->isLocked = ctx.get<ResponseType>(e) == ResponseType::Static;
  };

  ckpt.numBoxes = ctx.data().numActiveBoxes;
  for (i32 i = 0; i < ckpt.numBoxes; i++) {
    Entity box = ctx.data().boxes[i];
    ckptObject(box, &ckpt.boxes[i]);
  }

  ckpt.numRamps = ctx.data().numActiveRamps;
  for (i32 i = 0; i < ckpt.numRamps; i++) {
    Entity ramp = ctx.data().ramps[i];
    ckptObject(ramp, &ckpt.ramps[i]);
  }
}


static TaskGraphNodeID processActionsAndPhysicsTasks(TaskGraphBuilder &builder,
                                                     const Config &cfg)
{
    bool instantaneous_move =
        (cfg.simFlags & SimFlags::ZeroAgentVelocity) == SimFlags::ZeroAgentVelocity;

    TaskGraphNodeID move_sys;

    if (instantaneous_move) {
        move_sys = builder.addToGraph<ParallelForNode<Engine, instantMovementSystem,
            Action, SimEntity, AgentType>>({});
    } else {
        move_sys = builder.addToGraph<ParallelForNode<Engine, movementSystem,
            Action, SimEntity, AgentType>>({});
    }

    auto broadphase_setup_sys = phys::PhysicsSystem::setupBroadphaseTasks(builder,
        {move_sys});

    auto action_sys = builder.addToGraph<ParallelForNode<Engine, actionSystem,
        Action, SimEntity, AgentType>>({broadphase_setup_sys});

    auto substep_sys = PhysicsSystem::setupPhysicsStepTasks(builder,
        {action_sys}, numPhysicsSubsteps, physicsSolverSelector);

    auto sim_done = substep_sys;

    sim_done = phys::PhysicsSystem::setupCleanupTasks(
        builder, {sim_done});

    if (instantaneous_move) {
        sim_done = builder.addToGraph<ParallelForNode<Engine, agentZeroVelSystem,
            Velocity, GrabData>>({sim_done});
    }

    return sim_done;
}

static TaskGraphNodeID rewardsAndDonesTasks(TaskGraphBuilder &builder,
                                            Span<const TaskGraphNodeID> deps)
{
    auto rewards_vis = builder.addToGraph<ParallelForNode<Engine,
        rewardsVisSystem,
            SimEntity,
            AgentType
        >>(deps);

    auto output_rewards_dones = builder.addToGraph<ParallelForNode<Engine,
        outputRewardsDonesSystem,
            SimEntity,
            AgentType,
            Reward,
            Done
        >>({rewards_vis});

    auto update_episode_results = builder.addToGraph<ParallelForNode<Engine,
        updateEpisodeResultsSystem,
            EpisodeResult
        >>({output_rewards_dones});

    return update_episode_results;
}

static TaskGraphNodeID postGenTasks(TaskGraphBuilder &builder,
                                    Span<const TaskGraphNodeID> deps)
{
    auto clear_tmp = builder.addToGraph<ResetTmpAllocNode>(deps);

    auto compact_dyn_agent = builder.addToGraph<CompactArchetypeNode<DynAgent>>({clear_tmp});
    auto compact_objects = builder.addToGraph<CompactArchetypeNode<DynamicObject>>({compact_dyn_agent});
    auto reset_finish = compact_objects;

#ifdef MADRONA_GPU_MODE
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({reset_finish});
    (void)recycle_sys;
#endif

    auto post_reset_broadphase = phys::PhysicsSystem::setupBroadphaseTasks(
        builder, {reset_finish});

    return post_reset_broadphase;
}

static TaskGraphNodeID resetTasks(TaskGraphBuilder &builder,
                                  Span<const TaskGraphNodeID> deps)
{
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem, WorldReset>>(deps);

    return postGenTasks(builder, {reset_sys});
}

static void observationsTasks(const Config &cfg,
                              TaskGraphBuilder &builder,
                              Span<const TaskGraphNodeID> deps)
{
    auto collect_observations = builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
            Entity,
            SimEntity,
            SelfObservation,
            RelativeAgentObservations,
            RelativeBoxObservations,
            RelativeRampObservations,
            AgentPrepCounter
        >>(deps);

#ifdef MADRONA_GPU_MODE
    auto compute_visibility = builder.addToGraph<CustomParallelForNode<Engine,
        computeVisibilitySystem, 32, 1,
#else
    auto compute_visibility = builder.addToGraph<ParallelForNode<Engine,
        computeVisibilitySystem,
#endif
            Entity,
            SimEntity,
            AgentType,
            AgentVisibilityMasks,
            BoxVisibilityMasks,
            RampVisibilityMasks
        >>(deps);

#ifdef MADRONA_GPU_MODE
    auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
        lidarSystem, 32, 1,
#else
    auto lidar = builder.addToGraph<ParallelForNode<Engine,
        lidarSystem,
#endif
            SimEntity,
            Lidar
        >>(deps);

    auto global_positions_debug = builder.addToGraph<ParallelForNode<Engine,
        globalPositionsDebugSystem,
            GlobalDebugPositions
        >>(deps);

    if (cfg.renderBridge) {
        auto update_camera = builder.addToGraph<ParallelForNode<Engine,
            updateCameraSystem,
                Position,
                Rotation,
                SimEntity
            >>(deps);

        RenderingSystem::setupTasks(builder, {update_camera});
    }

    (void)lidar;
    (void)compute_visibility;
    (void)collect_observations;
    (void)global_positions_debug;
}

static void setupInitTasks(TaskGraphBuilder &builder, const Config &cfg)
{
    // Agent interfaces only need to be sorted during init
    auto compact_agent_iface =
        builder.addToGraph<CompactArchetypeNode<AgentInterface>>({});

    auto resets = resetTasks(builder, {
        compact_agent_iface
    });
    observationsTasks(cfg, builder, {resets});
}

static void setupStepTasks(TaskGraphBuilder &builder, const Config &cfg)
{
    auto sim_done = processActionsAndPhysicsTasks(builder, cfg);
    auto rewards_and_dones = rewardsAndDonesTasks(builder, {sim_done});
    auto resets = resetTasks(builder, {rewards_and_dones});
    observationsTasks(cfg, builder, {resets});
}

static void setupSaveCheckpointTasks(TaskGraphBuilder &builder,
                                     const Config &)
{
  builder.addToGraph<ParallelForNode<Engine,
    saveCheckpointSystem,
      CheckpointControl
    >>({});
}

static void setupLoadCheckpointTasks(TaskGraphBuilder &builder,
                                     const Config &cfg)
{
  auto load = builder.addToGraph<ParallelForNode<Engine,
    loadCheckpointSystem,
      CheckpointControl
    >>({});

  auto post_gen = postGenTasks(builder, {load});
  observationsTasks(cfg, builder, {post_gen});
}

void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &cfg)
{
    setupInitTasks(taskgraph_mgr.init(TaskGraphID::Init), cfg);
    setupStepTasks(taskgraph_mgr.init(TaskGraphID::Step), cfg);
    setupSaveCheckpointTasks(
        taskgraph_mgr.init(TaskGraphID::SaveCheckpoints), cfg);
    setupLoadCheckpointTasks(
        taskgraph_mgr.init(TaskGraphID::LoadCheckpoints), cfg);
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    simFlags = cfg.simFlags;

    initRandKey = cfg.initRandKey;
    curWorldEpisode = 0;

    const CountT max_total_entities = consts::maxBoxes + consts::maxRamps +
        consts::maxAgents + 35;

    PhysicsSystem::init(ctx, cfg.rigidBodyObjMgr, deltaT,
         numPhysicsSubsteps, -9.8 * math::up, max_total_entities,
         physicsSolverSelector);

    enableRender = cfg.renderBridge != nullptr;

    if (enableRender) {
        RenderingSystem::init(ctx, cfg.renderBridge);
    }

    obstacles =
        (Entity *)rawAlloc(sizeof(Entity) * size_t(max_total_entities));
    numObstacles = 0;

    numHiders = 0;
    numSeekers = 0;
    numActiveAgents = 0;
    
    curEpisodeStep = 0;

    minHiders = cfg.minHiders;
    maxHiders = cfg.maxHiders;
    minSeekers = cfg.minSeekers;
    maxSeekers = cfg.maxSeekers;
    maxAgentsPerWorld = cfg.maxHiders + cfg.maxSeekers;

    assert(maxAgentsPerWorld <= consts::maxAgents && maxAgentsPerWorld > 0);

    ctx.singleton<WorldReset>() = {
        .resetLevel = 1,
    };
     
    ctx.singleton<CheckpointControl>() = {
        .trigger = 0,
    };

    for (CountT i = 0; i < (CountT)maxAgentsPerWorld; i++) {
        Entity agent_iface = agentInterfaces[i] =
            ctx.makeEntity<AgentInterface>();

        if (enableRender) {
            render::RenderingSystem::attachEntityToView(ctx,
                    agent_iface,
                    100.f, 0.001f,
                    0.5f * math::up);
        }
    }

    ctx.data().hiderTeamReward.store_relaxed(1.f);
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Config, WorldInit);

}
