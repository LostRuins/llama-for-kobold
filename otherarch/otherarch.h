#pragma once

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "utils.h"
#include "model_adapter.h"

//for sampler params
struct kcpp_params {
    uint32_t seed                 = 0xFFFFFFFF; // RNG seed
    int32_t n_predict             =    -1; // new tokens to predict
    int32_t n_ctx                 =     0; // context size
    int32_t n_batch               =  2048; // logical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_ubatch              =   512; // physical batch size for prompt processing (must be >=32 to use BLAS)
    int      n_threads                   = -1;
    int      n_blasthreads               = -1;

    // sampling parameters
    int32_t top_k             = 40;    // <= 0 to use vocab size
    float   top_p             = 0.95f; // 1.0 = disabled
    float   min_p             = 0.0f; // 0.0 = disabled
    float   tfs_z             = 1.00f; // 1.0 = disabled
    float   nsigma            = -1.00f; // -1.0 - disabled
    float   typical_p         = 1.00f; // 1.0 = disabled
    float   temp              = 0.80f; // 1.0 = disabled
    float   smoothing_factor  = 0.00f; // 0.00 = disabled
    float   repeat_penalty    = 1.10f; // 1.0 = disabled
    int32_t repeat_last_n     = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   rep_pen_slope     = 1.0f;
    float   presence_penalty  = 0.00f; // 0.0 = disabled
    int32_t mirostat          = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostat_tau      = 5.00f; // target entropy
    float   mirostat_eta      = 0.10f; // learning rate
    float   dry_multiplier    = 0.0f;  // penalty multiplier, 0.0 = disabled
    float   dry_base          = 1.75f; // exponential base
    int32_t dry_allowed_length = 2;    // repeated sequences longer than this are penalized
    int32_t dry_penalty_last_n = 0;    // how many tokens to scan for repetitions (0 = entire context)
    std::vector<std::string> dry_sequence_breakers; // DRY sequence breakers
    float xtc_threshold        = 0;
    float xtc_probability      = 0;
    float   dynatemp_range     = 0.0f;  // enables DynaTemp if greater than 0. dynatemp_min = temperature - dt_range, dynatemp_max = temperature + dt_range
    float   dynatemp_exponent  = 1.0f;

    std::string model_filename       = ""; // model path
    std::string prompt               = "";
    bool flash_attn                  = false; // flash attention
    bool use_smartcontext            = false;
    bool use_contextshift            = false;
    bool use_fastforward             = false;
};

// default hparams (GPT-J 6B)
struct gptj_hparams {
    int32_t n_vocab = 50400;
    int32_t n_ctx   = 2048;
    int32_t n_embd  = 4096;
    int32_t n_head  = 16;
    int32_t n_layer = 28;
    int32_t n_rot   = 64;
    int32_t ftype   = 1;

    float rope_freq_base  = 10000.0f;
    float rope_freq_scale = 1.0f;
};

struct gptj_layer {
    // normalization
    struct ggml_v3_tensor * ln_1_g;
    struct ggml_v3_tensor * ln_1_b;

    // attention
    struct ggml_v3_tensor * c_attn_q_proj_w;
    struct ggml_v3_tensor * c_attn_k_proj_w;
    struct ggml_v3_tensor * c_attn_v_proj_w;

    struct ggml_v3_tensor * c_attn_proj_w;

    // ff
    struct ggml_v3_tensor * c_mlp_fc_w;
    struct ggml_v3_tensor * c_mlp_fc_b;

    struct ggml_v3_tensor * c_mlp_proj_w;
    struct ggml_v3_tensor * c_mlp_proj_b;
};
struct gptj_layer_v2 {
    // normalization
    struct ggml_v2_tensor * ln_1_g;
    struct ggml_v2_tensor * ln_1_b;

    // attention
    struct ggml_v2_tensor * c_attn_q_proj_w;
    struct ggml_v2_tensor * c_attn_k_proj_w;
    struct ggml_v2_tensor * c_attn_v_proj_w;

    struct ggml_v2_tensor * c_attn_proj_w;

    // ff
    struct ggml_v2_tensor * c_mlp_fc_w;
    struct ggml_v2_tensor * c_mlp_fc_b;

    struct ggml_v2_tensor * c_mlp_proj_w;
    struct ggml_v2_tensor * c_mlp_proj_w_trans; //for backwards compatibility
    struct ggml_v2_tensor * c_mlp_proj_b;
};
struct gptj_layer_v1 {
    // normalization
    struct ggml_v1_tensor * ln_1_g;
    struct ggml_v1_tensor * ln_1_b;

    // attention
    struct ggml_v1_tensor * c_attn_q_proj_w;
    struct ggml_v1_tensor * c_attn_k_proj_w;
    struct ggml_v1_tensor * c_attn_v_proj_w;

    struct ggml_v1_tensor * c_attn_proj_w;

    // ff
    struct ggml_v1_tensor * c_mlp_fc_w;
    struct ggml_v1_tensor * c_mlp_fc_b;

    struct ggml_v1_tensor * c_mlp_proj_w;
    struct ggml_v1_tensor * c_mlp_proj_w_trans; //for backwards compatibility
    struct ggml_v1_tensor * c_mlp_proj_b;
};

struct gptj_v1_model {
    gptj_hparams hparams;

    // normalization
    struct ggml_v1_tensor * ln_f_g;
    struct ggml_v1_tensor * ln_f_b;

    struct ggml_v1_tensor * wte; // position embedding

    struct ggml_v1_tensor * lmh_g; // language model head
    struct ggml_v1_tensor * lmh_b; // language model bias

    std::vector<gptj_layer_v1> layers;

    // key + value memory
    struct ggml_v1_tensor * memory_k;
    struct ggml_v1_tensor * memory_v;

    //
    struct ggml_v1_context * ctx;
    std::map<std::string, struct ggml_v1_tensor *> tensors;
};

struct gptj_v2_model {
    gptj_hparams hparams;

    // normalization
    struct ggml_v2_tensor * ln_f_g;
    struct ggml_v2_tensor * ln_f_b;

    struct ggml_v2_tensor * wte; // position embedding

    struct ggml_v2_tensor * lmh_g; // language model head
    struct ggml_v2_tensor * lmh_b; // language model bias

    std::vector<gptj_layer_v2> layers;

    // key + value memory
    struct ggml_v2_tensor * memory_k;
    struct ggml_v2_tensor * memory_v;

    //
    struct ggml_v2_context * ctx;
    std::map<std::string, struct ggml_v2_tensor *> tensors;
};

struct gptj_model {
    gptj_hparams hparams;

    // normalization
    struct ggml_v3_tensor * ln_f_g;
    struct ggml_v3_tensor * ln_f_b;

    struct ggml_v3_tensor * wte; // position embedding

    struct ggml_v3_tensor * lmh_g; // language model head
    struct ggml_v3_tensor * lmh_b; // language model bias

    std::vector<gptj_layer> layers;

    // key + value memory
    struct ggml_v3_tensor * memory_k;
    struct ggml_v3_tensor * memory_v;

    //
    struct ggml_v3_context * ctx;
    std::map<std::string, struct ggml_v3_tensor *> tensors;
};

// default hparams (GPT-2 117M)
struct gpt2_hparams {
    int32_t n_vocab = 50257;
    int32_t n_ctx   = 1024;
    int32_t n_embd  = 768;
    int32_t n_head  = 12;
    int32_t n_layer = 12;
    int32_t ftype     = 1;
};

struct gpt2_v1_layer {
    // normalization
    struct ggml_v1_tensor * ln_1_g;
    struct ggml_v1_tensor * ln_1_b;

    struct ggml_v1_tensor * ln_2_g;
    struct ggml_v1_tensor * ln_2_b;

    // attention
    struct ggml_v1_tensor * c_attn_attn_w;
    struct ggml_v1_tensor * c_attn_attn_b;

    struct ggml_v1_tensor * c_attn_proj_w;
    struct ggml_v1_tensor * c_attn_proj_b;

    // mlp
    struct ggml_v1_tensor * c_mlp_fc_w;
    struct ggml_v1_tensor * c_mlp_fc_b;

    struct ggml_v1_tensor * c_mlp_proj_w_trans; // transposed for efficiency
    struct ggml_v1_tensor * c_mlp_proj_b;
};

struct gpt2_v1_model {
    gpt2_hparams hparams;

    // normalization
    struct ggml_v1_tensor * ln_f_g;
    struct ggml_v1_tensor * ln_f_b;

    struct ggml_v1_tensor * wte; // position embedding
    struct ggml_v1_tensor * wpe; //    token embedding

    std::vector<gpt2_v1_layer> layers;

    // key + value memory
    struct ggml_v1_tensor * memory_k;
    struct ggml_v1_tensor * memory_v;

    //
    struct ggml_v1_context * ctx;
    std::map<std::string, struct ggml_v1_tensor *> tensors;
};

struct gpt2_layer_v2 {
    // normalization
    struct ggml_v2_tensor * ln_1_g;
    struct ggml_v2_tensor * ln_1_b;

    struct ggml_v2_tensor * ln_2_g;
    struct ggml_v2_tensor * ln_2_b;

    // attention
    struct ggml_v2_tensor * c_attn_attn_w;
    struct ggml_v2_tensor * c_attn_attn_b;

    struct ggml_v2_tensor * c_attn_proj_w;
    struct ggml_v2_tensor * c_attn_proj_b;

    // mlp
    struct ggml_v2_tensor * c_mlp_fc_w;
    struct ggml_v2_tensor * c_mlp_fc_b;

    struct ggml_v2_tensor * c_mlp_proj_w;
    struct ggml_v2_tensor * c_mlp_proj_b;
};

struct gpt2_v2_model {
    gpt2_hparams hparams;

    // normalization
    struct ggml_v2_tensor * ln_f_g;
    struct ggml_v2_tensor * ln_f_b;

    struct ggml_v2_tensor * wte;     // position embedding
    struct ggml_v2_tensor * wpe;     //    token embedding
    struct ggml_v2_tensor * lm_head; // language model head

    std::vector<gpt2_layer_v2> layers;

    // key + value memory
    struct ggml_v2_tensor * memory_k;
    struct ggml_v2_tensor * memory_v;

    //
    struct ggml_v2_context * ctx;
    std::map<std::string, struct ggml_v2_tensor *> tensors;
};

struct gpt2_layer {
    // normalization
    struct ggml_v3_tensor * ln_1_g;
    struct ggml_v3_tensor * ln_1_b;

    struct ggml_v3_tensor * ln_2_g;
    struct ggml_v3_tensor * ln_2_b;

    // attention
    struct ggml_v3_tensor * c_attn_attn_w;
    struct ggml_v3_tensor * c_attn_attn_b;

    struct ggml_v3_tensor * c_attn_proj_w;
    struct ggml_v3_tensor * c_attn_proj_b;

    // mlp
    struct ggml_v3_tensor * c_mlp_fc_w;
    struct ggml_v3_tensor * c_mlp_fc_b;

    struct ggml_v3_tensor * c_mlp_proj_w;
    struct ggml_v3_tensor * c_mlp_proj_b;
};

struct gpt2_model {
    gpt2_hparams hparams;

    // normalization
    struct ggml_v3_tensor * ln_f_g;
    struct ggml_v3_tensor * ln_f_b;

    struct ggml_v3_tensor * wte;     // position embedding
    struct ggml_v3_tensor * wpe;     //    token embedding
    struct ggml_v3_tensor * lm_head; // language model head

    std::vector<gpt2_layer> layers;

    // key + value memory
    struct ggml_v3_tensor * memory_k;
    struct ggml_v3_tensor * memory_v;

    //
    struct ggml_v3_context * ctx;
    std::map<std::string, struct ggml_v3_tensor *> tensors;
};

// default hparams (StableLM 3B)
struct gpt_neox_hparams {
    int32_t n_vocab = 50257;
    int32_t n_ctx   = 4096;
    int32_t n_embd  = 4096;
    int32_t n_head  = 32;
    int32_t n_layer = 16;
    int32_t n_rot   = 32; // rotary_pct * (n_embd / n_head)
    int32_t par_res = 1; // 1 = true, 0 = false
    int32_t ftype   = 1;

    float rope_freq_base  = 10000.0f;
    float rope_freq_scale = 1.0f;
};

struct gpt_neox_layer_v2 {
    // pre normalization
    struct ggml_v2_tensor * ln_1_g;
    struct ggml_v2_tensor * ln_1_b;

    // attention
    struct ggml_v2_tensor * c_attn_attn_w;
    struct ggml_v2_tensor * c_attn_attn_b;

    struct ggml_v2_tensor * c_attn_proj_w;
    struct ggml_v2_tensor * c_attn_proj_b;

    // post normalization
    struct ggml_v2_tensor * ln_2_g;
    struct ggml_v2_tensor * ln_2_b;

    // ff
    struct ggml_v2_tensor * c_mlp_fc_w;
    struct ggml_v2_tensor * c_mlp_fc_b;

    struct ggml_v2_tensor * c_mlp_proj_w;
    struct ggml_v2_tensor * c_mlp_proj_b;
};

struct gpt_neox_v2_model {
    gpt_neox_hparams hparams;

    // normalization
    struct ggml_v2_tensor * ln_f_g;
    struct ggml_v2_tensor * ln_f_b;

    struct ggml_v2_tensor * wte; // position embedding

    struct ggml_v2_tensor * lmh_g; // language model head
    //struct ggml_v3_tensor * lmh_b; // language model bias

    std::vector<gpt_neox_layer_v2> layers;

    // key + value memory
    struct ggml_v2_tensor * memory_k;
    struct ggml_v2_tensor * memory_v;

    //
    struct ggml_v2_context * ctx;
    std::map<std::string, struct ggml_v2_tensor *> tensors;
};

struct gpt_neox_layer {
    // pre normalization
    struct ggml_v3_tensor * ln_1_g;
    struct ggml_v3_tensor * ln_1_b;

    // attention
    struct ggml_v3_tensor * c_attn_attn_w;
    struct ggml_v3_tensor * c_attn_attn_b;

    struct ggml_v3_tensor * c_attn_proj_w;
    struct ggml_v3_tensor * c_attn_proj_b;

    // post normalization
    struct ggml_v3_tensor * ln_2_g;
    struct ggml_v3_tensor * ln_2_b;

    // ff
    struct ggml_v3_tensor * c_mlp_fc_w;
    struct ggml_v3_tensor * c_mlp_fc_b;

    struct ggml_v3_tensor * c_mlp_proj_w;
    struct ggml_v3_tensor * c_mlp_proj_b;
};

struct gpt_neox_model {
    gpt_neox_hparams hparams;

    // normalization
    struct ggml_v3_tensor * ln_f_g;
    struct ggml_v3_tensor * ln_f_b;

    struct ggml_v3_tensor * wte; // position embedding

    struct ggml_v3_tensor * lmh_g; // language model head
    //struct ggml_v3_tensor * lmh_b; // language model bias

    std::vector<gpt_neox_layer> layers;

    // key + value memory
    struct ggml_v3_tensor * memory_k;
    struct ggml_v3_tensor * memory_v;

    //
    struct ggml_v3_context * ctx;
    std::map<std::string, struct ggml_v3_tensor *> tensors;
};


// no defaults for now
struct mpt_hparams {
    int32_t d_model      = 0;
    int32_t max_seq_len  = 0;
    int32_t n_heads      = 0;
    int32_t n_layers     = 0;
    int32_t n_vocab      = 0;
    float alibi_bias_max = 0;
    float clip_qkv       = 0;
    int32_t ftype        = 0;
    int32_t n_ctx        = 0;

};

struct mpt_layer {
    // pre normalization
    struct ggml_v3_tensor * norm_1_weight;

    // attention
    struct ggml_v3_tensor * c_attn_wqkv_weight;
    struct ggml_v3_tensor * c_attn_out_proj_weight;

    // post normalization
    struct ggml_v3_tensor * norm_2_weight;

    // ff
    struct ggml_v3_tensor * ffn_up_proj;
    struct ggml_v3_tensor * ffn_down_proj;
};

struct mpt_model {
    mpt_hparams hparams;

    struct ggml_v3_tensor * wte_weight;    // position embedding
    struct ggml_v3_tensor * norm_f_weight; // language model head

    std::vector<mpt_layer> layers;

    // key + value memory
    struct ggml_v3_tensor * memory_k;
    struct ggml_v3_tensor * memory_v;

    struct ggml_v3_context * ctx;
    std::map<std::string, struct ggml_v3_tensor *> tensors;
};

struct llava_image
{
    std::string b64data = "";
    int32_t clp_image_tokens = 0; //holds number of tokens llava used
    float * clp_img_embd = nullptr; //this holds dynamic memory and must be freed each use!
};

struct speculative_draft_result
{
    std::vector<int32_t> draftids;
    std::vector<float *> actual_logits;
    bool draft_success = false;
    int drafted_amount = 0;
};

const float default_norm_eps = 1e-5f;
