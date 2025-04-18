#pragma once

#include <cstdint>

namespace GPUHideSeek {

enum class SimFlags : uint32_t {
    Default                = 0,
    UseFixedWorld          = 1 << 0,
    IgnoreEpisodeLength    = 1 << 1,
    RandomFlipTeams        = 1 << 2,
    ZeroAgentVelocity      = 1 << 3,
};

inline SimFlags & operator|=(SimFlags &a, SimFlags b);
inline SimFlags operator|(SimFlags a, SimFlags b);
inline SimFlags & operator&=(SimFlags &a, SimFlags b);
inline SimFlags operator&(SimFlags a, SimFlags b);

}

#include "sim_flags.inl"
