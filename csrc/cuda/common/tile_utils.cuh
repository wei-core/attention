#pragma once
namespace qd_attention {
__host__ __device__ inline int ceil_div(int n, int d) { return (n + d - 1) / d; }
__host__ __device__ inline int head_to_kv_head(int head, int hq, int hkv) { return head / (hq / hkv); }
// Right-aligned causal mask: query i corresponds to KV position kv_len-q_len+i.
__host__ __device__ inline int causal_end(int i, int q_len, int kv_len) {
    int end = kv_len - q_len + i + 1;
    return end < 0 ? 0 : (end > kv_len ? kv_len : end);
}
}
