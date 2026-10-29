#include <torch/extension.h>
#include <torch/library.h>

// Implementation defined in csrc/dense/gqa_decode.cu.
torch::Tensor gqa_decode_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_table,
    const torch::Tensor& context_lens,
    double scale);

torch::Tensor gqa_decode_split_cuda(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, double, int64_t);

void kv_append_cuda(const torch::Tensor&, const torch::Tensor&, torch::Tensor, torch::Tensor, const torch::Tensor&, const torch::Tensor&);

torch::Tensor gqa_prefill_cuda(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, double, bool);

torch::Tensor gqa_decode_split_workspace_cuda(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, double, int64_t, const torch::Tensor&);

torch::Tensor mla_project_cuda(const torch::Tensor&, const torch::Tensor&, bool);
void latent_kv_write_cuda(const torch::Tensor&, const torch::Tensor&, torch::Tensor, const torch::Tensor&);
torch::Tensor mla_attend_cuda(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const c10::optional<torch::Tensor>&, double, bool);

torch::Tensor gated_attention_cuda(const torch::Tensor&,const torch::Tensor&);
torch::Tensor delta_output_cuda(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,double);
std::tuple<torch::Tensor,torch::Tensor> delta_recurrent_cuda(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,double);

std::tuple<torch::Tensor,torch::Tensor> causal_conv1d_cuda(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
torch::Tensor lightning_indexer_cuda(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
torch::Tensor topk_select_cuda(const torch::Tensor&,int64_t);

std::tuple<torch::Tensor,torch::Tensor> delta_chunkwise_cuda(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,int64_t,double);

torch::Tensor mla_prefill_direct_cuda(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,double);

TORCH_LIBRARY(qd_attention, m) {
    m.def("mla_prefill_direct(Tensor q, Tensor qr, Tensor latent, Tensor rope, Tensor wk, Tensor wv, Tensor lengths, Tensor positions, float scale) -> Tensor");
    m.def("delta_chunkwise(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state, int chunk, float eps) -> (Tensor, Tensor)");
    m.def("causal_conv1d(Tensor x, Tensor weight, Tensor state) -> (Tensor, Tensor)");
    m.def("lightning_indexer(Tensor q, Tensor k, Tensor weights, Tensor lengths, Tensor positions) -> Tensor");
    m.def("topk_select(Tensor scores, int k) -> Tensor");
    m.def("gated_attention(Tensor attn, Tensor gate) -> Tensor");
    m.def("delta_output(Tensor x, Tensor z, Tensor weight, float eps) -> Tensor");
    m.def("delta_recurrent(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state, float eps) -> (Tensor, Tensor)");
    m.def("mla_project(Tensor x, Tensor weight, bool absorb) -> Tensor");
    m.def("latent_kv_write(Tensor latent, Tensor rope, Tensor(a!) cache, Tensor slots) -> ()");
    m.def("mla_attend(Tensor q, Tensor q_rope, Tensor cache, Tensor table, Tensor lens, Tensor positions, Tensor? indices, float scale, bool causal) -> Tensor");
    m.def("gqa_prefill(Tensor q, Tensor k, Tensor v, Tensor q_lens, Tensor kv_lens, float scale, bool causal) -> Tensor");
    m.def("kv_append(Tensor key, Tensor value, Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor block_table, Tensor positions) -> ()");
    m.def("gqa_decode_split_workspace(Tensor q, Tensor k, Tensor v, Tensor table, Tensor lengths, float scale, int splits, Tensor(a!) workspace) -> Tensor");
    m.def("gqa_decode_split(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_table, Tensor context_lens, float scale, int splits) -> Tensor");
    m.def("gqa_decode(Tensor q, Tensor k_cache, Tensor v_cache, "
          "Tensor block_table, Tensor context_lens, float scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(qd_attention, CUDA, m) {
    m.impl("mla_prefill_direct", TORCH_FN(mla_prefill_direct_cuda));
    m.impl("delta_chunkwise", TORCH_FN(delta_chunkwise_cuda));
    m.impl("causal_conv1d", TORCH_FN(causal_conv1d_cuda));
    m.impl("lightning_indexer", TORCH_FN(lightning_indexer_cuda));
    m.impl("topk_select", TORCH_FN(topk_select_cuda));
    m.impl("gated_attention", TORCH_FN(gated_attention_cuda));
    m.impl("delta_output", TORCH_FN(delta_output_cuda));
    m.impl("delta_recurrent", TORCH_FN(delta_recurrent_cuda));
    m.impl("mla_project", TORCH_FN(mla_project_cuda));
    m.impl("latent_kv_write", TORCH_FN(latent_kv_write_cuda));
    m.impl("mla_attend", TORCH_FN(mla_attend_cuda));
    m.impl("gqa_prefill", TORCH_FN(gqa_prefill_cuda));
    m.impl("kv_append", TORCH_FN(kv_append_cuda));
    m.impl("gqa_decode_split_workspace", TORCH_FN(gqa_decode_split_workspace_cuda));
    m.impl("gqa_decode_split", TORCH_FN(gqa_decode_split_cuda));
    m.impl("gqa_decode", TORCH_FN(gqa_decode_cuda));
}

// Imports this extension as a Python module; ops are exposed via torch.ops.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
