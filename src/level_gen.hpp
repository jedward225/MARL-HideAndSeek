#pragma once

#include "sim.hpp"

namespace GPUHideSeek {

void generateEnvironment(Engine &ctx,
                         RandKey level_gen_rnd,
                         CountT level_id,
                         CountT num_hiders,
                         CountT num_seekers);

}
