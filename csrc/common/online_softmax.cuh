#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>

namespace qd_attention {
struct SoftmaxState {
    float max, sum;
    __device__ SoftmaxState() : max(-INFINITY), sum(0.f) {}
    // Returns coefficients for the old and new, unnormalized accumulators.
    __device__ void add(float score, float& old_weight, float& new_weight) {
        float next = fmaxf(max, score);
        old_weight = sum == 0.f ? 0.f : expf(max - next);
        new_weight = expf(score - next);
        sum = old_weight * sum + new_weight;
        max = next;
    }
    __device__ void merge(float other_max, float other_sum, float& old_weight, float& new_weight) {
        if (other_sum == 0.f) { old_weight = 1.f; new_weight = 0.f; return; }
        float next = fmaxf(max, other_max);
        old_weight = sum == 0.f ? 0.f : expf(max - next);
        new_weight = expf(other_max - next);
        sum = old_weight * sum + new_weight * other_sum;
        max = next;
    }
};
}
