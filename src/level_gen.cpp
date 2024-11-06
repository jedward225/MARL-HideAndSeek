#include "level_gen.hpp"
#include "geo_gen.hpp"

namespace RenderingSystem = madrona::render::RenderingSystem;

namespace GPUHideSeek {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

static Entity makeAgent(Engine &ctx, Vector3 pos, Quat rot, AgentType agent_type)
{
    Entity agent_iface =
        ctx.data().agentInterfaces[ctx.data().numActiveAgents++];
    ctx.get<AgentType>(agent_iface) = agent_type;

    Entity agent = ctx.makeRenderableEntity<DynAgent>();
    ctx.get<SimEntity>(agent_iface).e = agent;

    ctx.get<AgentActiveMask>(agent_iface).mask = 1.f;

    ctx.get<Seed>(agent_iface).key = ctx.data().curEpisodeRNDCounter;

    // Zero out actions
    ctx.get<Action>(agent_iface) = {
        .x = 2,
        .y = 2,
        .r = 2,
        .g = 0,
        .l = 0,
    };

    ctx.get<Position>(agent) = pos;
    ctx.get<Rotation>(agent) = rot;
    ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };

    ObjectID obj_id;
    if (agent_type == AgentType::Seeker) {
        obj_id = { (uint32_t)SimObject::Seeker };

        ctx.data().seekers[ctx.data().numSeekers++] = agent;
    } else {
        obj_id = { (uint32_t)SimObject::Hider };

        ctx.data().hiders[ctx.data().numHiders++] = agent;
    }

    ctx.get<ObjectID>(agent) = obj_id;

    ctx.get<phys::broadphase::LeafID>(agent) =
        PhysicsSystem::registerEntity(ctx, agent, obj_id);

    ctx.get<Velocity>(agent) = {
        Vector3::zero(),
        Vector3::zero(),
    };

    ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
    ctx.get<OwnerTeam>(agent) = OwnerTeam::Unownable;
    ctx.get<ExternalForce>(agent) = Vector3::zero();
    ctx.get<ExternalTorque>(agent) = Vector3::zero();
    ctx.get<GrabData>(agent).constraintEntity = Entity::none();

    return agent;
}

static Entity makePlane(Engine &ctx, Vector3 offset, Quat rot) {
    return makeDynObject(ctx, offset, rot, SimObject::Plane,
                         ResponseType::Static, OwnerTeam::Unownable);
}

// Emergent tool use configuration:
// 1 - 3 Hiders
// 1 - 3 Seekers
// 3 - 9 Movable boxes (at least 3 elongated)
// 2 movable ramps

static void generateTrainingEnvironment(Engine &ctx,
                                        RNG &rng,
                                        CountT num_hiders,
                                        CountT num_seekers)
{
    CountT total_num_boxes = (CountT)rng.sampleI32(3, 10);
    assert(total_num_boxes <= consts::maxBoxes);

    CountT num_elongated = 
        (CountT)rng.sampleI32(3, total_num_boxes);

    CountT num_cubes = total_num_boxes - num_elongated;

    assert(num_elongated + num_cubes == total_num_boxes);

    const Vector2 bounds { -18.f, 18.f };
    float bounds_diff = bounds.y - bounds.x;

    const ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;

    CountT num_entities =
        populateStaticGeometry(ctx, rng, {bounds.y, bounds.y});

    Entity *all_entities = ctx.data().obstacles;

    auto checkOverlap = [&obj_mgr, &ctx,
                         &all_entities, &num_entities](const AABB &aabb) {
        for (int i = 0; i < num_entities; ++i) {
            ObjectID obj_id = ctx.get<ObjectID>(all_entities[i]);
            AABB other = obj_mgr.rigidBodyAABBs[obj_id.idx];

            Position p = ctx.get<Position>(all_entities[i]);
            Rotation r = ctx.get<Rotation>(all_entities[i]);
            Scale s = ctx.get<Scale>(all_entities[i]);
            other = other.applyTRS(p, r, Diag3x3(s));

            if (aabb.overlaps(other)) {
                return false;
            }
        }

        return true;
    };

    const CountT max_rejections = 20;

    for (CountT i = 0; i < num_elongated; i++) {
        CountT rejections = 0;
        // Choose a random room and put the entity in a random position in that room
        while (true) {
            float bounds_diffx = bounds.y - bounds.x;
            float bounds_diffy = bounds.y - bounds.x;

            Vector3 pos {
                bounds.x + rng.sampleUniform() * bounds_diffx,
                bounds.x + rng.sampleUniform() * bounds_diffy,
                1.0f,
            };

            float box_rotation = rng.sampleUniform() * math::pi;
            const auto rot = Quat::angleAxis(box_rotation, {0, 0, 1});
            Diag3x3 scale = {1.0f, 1.0f, 1.0f};

            AABB aabb = obj_mgr.rigidBodyAABBs[(uint32_t)SimObject::Box];
            aabb = aabb.applyTRS(pos, rot, scale);

            // Check overlap with all other entities
            if (checkOverlap(aabb) || rejections == max_rejections) {
                ctx.data().boxes[i] = all_entities[num_entities++] =
                    makeDynObject(ctx, pos, rot, SimObject::Box);

                ctx.data().boxSizes[i] = { 8, 1.5, 2.f };
                break;
            }

            rejections++;
        }
    }

    for (CountT i = 0; i < num_cubes; i++) {
        CountT rejections = 0;
        while (true) {
            float bounds_diffx = bounds.y - bounds.x;
            float bounds_diffy = bounds.y - bounds.x;

#if 0
            Vector3 pos {
                room->low.x + rng.rand() * bounds_diffx,
                room->low.y + rng.rand() * bounds_diffy,
                1.0f,
            };
#endif

            Vector3 pos {
                bounds.x + rng.sampleUniform() * bounds_diffx,
                bounds.x + rng.sampleUniform() * bounds_diffy,
                1.0f,
            };

            float box_rotation = rng.sampleUniform() * math::pi;
            const auto rot = Quat::angleAxis(box_rotation, {0, 0, 1});
            Diag3x3 scale = {1.0f, 1.0f, 1.0f};

            AABB aabb = obj_mgr.rigidBodyAABBs[(uint32_t)SimObject::Cube];
            aabb = aabb.applyTRS(pos, rot, scale);

            if (checkOverlap(aabb) || rejections == max_rejections) {
                CountT box_idx = i + num_elongated;

                ctx.data().boxes[box_idx] = all_entities[num_entities++] =
                    makeDynObject(ctx, pos, rot, SimObject::Cube);

                ctx.data().boxSizes[box_idx] = { 2, 2, 2 };
                break;
            }

            ++rejections;
        }
    }

    ctx.data().numActiveBoxes = total_num_boxes;

    const CountT num_ramps = consts::maxRamps;
    for (CountT i = 0; i < num_ramps; i++) {
        CountT rejections = 0;
        while (true) {
            float bounds_diffx = bounds.y - bounds.x;
            float bounds_diffy = bounds.y - bounds.x;

            Vector3 pos {
                bounds.x + rng.sampleUniform() * bounds_diffx,
                bounds.x + rng.sampleUniform() * bounds_diffy,
                1.0f,
            };

            float ramp_rotation = rng.sampleUniform() * math::pi;
            const auto rot = Quat::angleAxis(ramp_rotation, {0, 0, 1});
            Diag3x3 scale = {1.0f, 1.0f, 1.0f};

            AABB aabb = obj_mgr.rigidBodyAABBs[(uint32_t)SimObject::Ramp];
            aabb = aabb.applyTRS(pos, rot, scale);

            if (checkOverlap(aabb) || rejections == max_rejections) {
                ctx.data().ramps[i] = all_entities[num_entities++] =
                    makeDynObject(ctx, pos, rot, SimObject::Ramp);
                break;
            }

            ++rejections;
        }
    }
    ctx.data().numActiveRamps = num_ramps;

    // Use the episode RNG not level gen RNG here so we can randomly
    // flip starting team even given a fixed seed.
    bool seekers_first = ctx.data().rng.sampleI32(0, 2) == 1;
    if ((ctx.data().simFlags & SimFlags::RandomFlipTeams) !=
            SimFlags::RandomFlipTeams) {
        seekers_first = false;
    }

    ctx.singleton<TeamState>() = {
        .seekersFirst = seekers_first,
    };

    CountT team_sizes[2];
    SimObject team_agent_obj_ids[2];
    AgentType team_agent_types[2];

    if (seekers_first) {
        team_sizes[0] = num_seekers;
        team_agent_obj_ids[0] = SimObject::Seeker;
        team_agent_types[0] = AgentType::Seeker;

        team_sizes[1] = num_hiders;
        team_agent_obj_ids[1] = SimObject::Hider;
        team_agent_types[1] = AgentType::Hider;
    } else {
        team_sizes[0] = num_hiders;
        team_agent_obj_ids[0] = SimObject::Hider;
        team_agent_types[0] = AgentType::Hider;

        team_sizes[1] = num_seekers;
        team_agent_obj_ids[1] = SimObject::Seeker;
        team_agent_types[1] = AgentType::Seeker;
    }

    MADRONA_UNROLL
    for (CountT team_idx = 0; team_idx < 2; team_idx++) {
        for (CountT i = 0; i < team_sizes[team_idx]; i++) {
            CountT rejections = 0;
            while (true) {
                Vector3 pos {
                    bounds.x + rng.sampleUniform() * bounds_diff,
                    bounds.x + rng.sampleUniform() * bounds_diff,
                    1.f,
                };

                const auto rot = Quat::angleAxis(
                    rng.sampleUniform() * math::pi, {0, 0, 1});
                Diag3x3 scale = {1.0f, 1.0f, 1.0f};

                AABB aabb = obj_mgr.rigidBodyAABBs[
                    (uint32_t)team_agent_obj_ids[team_idx]];
                aabb = aabb.applyTRS(pos, rot, scale);
                if (checkOverlap(aabb) || rejections == max_rejections) {
                    makeAgent(ctx, pos, rot, team_agent_types[team_idx]);
                    break;
                }

                rejections++;
            }
        }
    }

    all_entities[num_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));
    //all_entities[num_entities++] =
    //    makePlane(ctx, {0, 0, 100}, Quat::angleAxis(pi, {1, 0, 0}));
    //all_entities[num_entities++] =
    //    makePlane(ctx, {-100, 0, 0}, Quat::angleAxis(pi_d2, {0, 1, 0}));
    //all_entities[num_entities++] =
    //    makePlane(ctx, {100, 0, 0}, Quat::angleAxis(-pi_d2, {0, 1, 0}));
    //all_entities[num_entities++] =
    //    makePlane(ctx, {0, -100, 0}, Quat::angleAxis(-pi_d2, {1, 0, 0}));
    //all_entities[num_entities++] =
    //    makePlane(ctx, {0, 100, 0}, Quat::angleAxis(pi_d2, {1, 0, 0}));

    ctx.data().numObstacles = num_entities;
}

static void generateDebugEnvironment(Engine &ctx, CountT level_id);

void generateEnvironment(Engine &ctx,
                         RandKey level_gen_rnd,
                         CountT level_id,
                         CountT num_hiders,
                         CountT num_seekers)
{
    RNG level_rng(level_gen_rnd);

    if (level_id == 1) {
        generateTrainingEnvironment(ctx, level_rng, num_hiders, num_seekers);
    } else {
        generateDebugEnvironment(ctx, level_id);
    }

    for (CountT i = (CountT)ctx.data().numActiveAgents;
         i < ctx.data().maxAgentsPerWorld;
         i++) {
        Entity agent_iface = ctx.data().agentInterfaces[i];

        ctx.get<SimEntity>(agent_iface).e = Entity::none();
        ctx.get<AgentActiveMask>(agent_iface).mask = 0.f;
    }
}

static void singleCubeLevel(Engine &ctx, Vector3 pos, Quat rot)
{
    Entity *all_entities = ctx.data().obstacles;

    CountT total_entities = 0;

    Entity test_cube = makeDynObject(ctx, pos, rot, SimObject::Cube);
    all_entities[total_entities++] = test_cube;

    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));

    ctx.data().numObstacles = total_entities;
}

static void level2(Engine &ctx)
{
    Quat cube_rotation = (Quat::angleAxis(atanf(1.f/sqrtf(2.f)), {0, 1, 0}) *
        Quat::angleAxis(toRadians(45), {1, 0, 0})).normalize().normalize();
    singleCubeLevel(ctx, { 0, 0, 5 }, cube_rotation);
}

static void level3(Engine &ctx)
{
    singleCubeLevel(ctx, { 0, 0, 5 }, Quat::angleAxis(0, {0, 0, 1}));
}

static void level4(Engine &ctx)
{
    Vector3 pos { 0, 0, 5 };

    Quat rot = (
        Quat::angleAxis(toRadians(45), {0, 1, 0})).normalize();
#if 0
        Quat::angleAxis(toRadians(45), {0, 1, 0}) *
        Quat::angleAxis(toRadians(40), {1, 0, 0})).normalize();
#endif

    Entity *all_entities = ctx.data().obstacles;

    CountT total_entities = 0;

    //Entity test_cube = makeDynObject(ctx, pos, rot, 2);
    //all_entities[total_entities++] = test_cube;

    all_entities[total_entities++] =
        makeDynObject(ctx, pos + Vector3 {0, 0, 5}, rot, SimObject::Box,
                      ResponseType::Dynamic, OwnerTeam::None,
                      {1, 1, 1});

    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));
    //all_entities[total_entities++] =
    //    makePlane(ctx, {-20, 0, 0}, Quat::angleAxis(pi_d2, {0, 1, 0}));
    //all_entities[total_entities++] =
    //    makePlane(ctx, {20, 0, 0}, Quat::angleAxis(-pi_d2, {0, 1, 0}));

    ctx.data().numObstacles = total_entities;
}

static void level5(Engine &ctx)
{
    Entity *all_entities = ctx.data().obstacles;

    CountT num_entities = 0;
    all_entities[num_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));

    makeAgent(ctx, {0, 0, 1}, Quat { 1, 0, 0, 0 }, AgentType::Hider);
}

static void level6(Engine &ctx)
{
    Entity *all_entities = ctx.data().obstacles;

    CountT num_entities = 0;
    all_entities[num_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));

    all_entities[num_entities++] =
        makeDynObject(
            ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}), SimObject::Wall,
            ResponseType::Static, OwnerTeam::Unownable, {10.f, 0.2f, 1.f} );

    all_entities[num_entities++] =
        makeDynObject(
            ctx, {0, -5, 1}, Quat::angleAxis(0, {1, 0, 0}), SimObject::Cube,
            ResponseType::Dynamic, OwnerTeam::None, {1.f, 1.f, 1.f} );

    makeAgent(ctx, { -15, -15, 1.5 },
        Quat::angleAxis(toRadians(-45), {0, 0, 1}), AgentType::Hider);

    makeAgent(ctx, { -15, -10, 1.5 },
        Quat::angleAxis(toRadians(45), {0, 0, 1}), AgentType::Seeker);

    ctx.data().numObstacles = num_entities;
}

static void level7(Engine &ctx)
{
    Vector3 pos { 0, 0, 5 };

    Quat rot = (
        Quat::angleAxis(toRadians(45), {0, 1, 0}) *
        Quat::angleAxis(toRadians(40), {1, 0, 0})).normalize();

    Entity *all_entities = ctx.data().obstacles;

    CountT total_entities = 0;

    all_entities[total_entities++] = 
        makeDynObject(ctx, pos, rot, SimObject::Cube);

    all_entities[total_entities++] =
        makeDynObject(ctx, pos + Vector3 {0, 0, 5}, rot, SimObject::Cube,
                      ResponseType::Dynamic, OwnerTeam::None,
                      {1, 1, 1});

    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {-20, 0, 0}, Quat::angleAxis(pi_d2, {0, 1, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {20, 0, 0}, Quat::angleAxis(-pi_d2, {0, 1, 0}));

    ctx.data().numObstacles = total_entities;
}

static void level8(Engine &ctx)
{
    Entity *all_entities = ctx.data().obstacles;

    CountT total_entities = 0;

    Vector3 ramp_pos { 0, 0, 10 };

    Quat ramp_rot = (
        Quat::angleAxis(toRadians(25), {0, 1, 0}) *
        Quat::angleAxis(toRadians(90), {0, 0, 1}) *
        Quat::angleAxis(toRadians(45), {1, 0, 0})).normalize();

    Entity ramp_dyn = all_entities[total_entities++] =
        makeDynObject(ctx, ramp_pos, ramp_rot, SimObject::Ramp);

    ctx.get<Velocity>(ramp_dyn).linear = {0, 0, -30};

    all_entities[total_entities++] =
        makeDynObject(ctx,
                      {-0.5, -0.5, 1},
                      (Quat::angleAxis(toRadians(-90), {1, 0, 0 }) *
                          Quat::angleAxis(pi, {0, 1, 0 })).normalize(),
                      SimObject::Ramp,
                      ResponseType::Static, OwnerTeam::None,
                      {1, 1, 1});

    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {-20, 0, 0}, Quat::angleAxis(pi_d2, {0, 1, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {20, 0, 0}, Quat::angleAxis(-pi_d2, {0, 1, 0}));

    ctx.data().numObstacles = total_entities;
}

static void generateDebugEnvironment(Engine &ctx, CountT level_id)
{
    switch (level_id) {
    case 2: {
        level2(ctx);
    } break;
    case 3: {
        level3(ctx);
    } break;
    case 4: {
        level4(ctx);
    } break;
    case 5: {
        level5(ctx);
    } break;
    case 6: {
        level6(ctx);
    } break;
    case 7: {
        level7(ctx);
    } break;
    case 8: {
        level8(ctx);
    } break;
    }
}

}
