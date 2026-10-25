#pragma once

#include <cuda_runtime.h>

namespace qd_attention {

// Current baseline uses 1D CTAs with 32..256 threads, multiple of 32.
// Scratch must provide at least 8 floats (256 / warpSize).
__device__ __forceinline__ float block_sum(float x, float* scratch) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    for (int delta = 16; delta >= 1; delta >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, delta);
    }
    if (lane == 0) scratch[warp_id] = x;
    __syncthreads();

    float total = 0.f;
    if (warp_id == 0) {
        total = lane < (blockDim.x >> 5) ? scratch[lane] : 0.f;
        for (int delta = 16; delta >= 1; delta >>= 1) {
            total += __shfl_down_sync(0xffffffffu, total, delta);
        }
        if (lane == 0) scratch[0] = total;
    }
    __syncthreads();
    return scratch[0];
}

}  // namespace qd_attention
