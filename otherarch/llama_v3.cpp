// Defines fileno on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstddef>
#include <cstdint>
#include <cstdio>
#endif

#define GGML_USE_K_QUANTS //forced on, now that the flag has been removed upstream

#include "llama-util.h"
#include "llama_v3.h"

#include "ggml_v3.h"
#include "otherarch.h"
#ifdef GGML_USE_CUDA
#include "ggml_v3-cuda.h"
#endif
#if defined(GGML_USE_CLBLAST)
#include "ggml_v3-opencl.h"
#endif


#ifdef GGML_USE_K_QUANTS
#ifndef QK_K
#ifdef GGML_QKK_64
#define QK_K 64
#else
#define QK_K 256
#endif
#endif
#endif

#include <array>
#include <ctime>
#include <cinttypes>
#include <fstream>
#include <random>
#include <map>
#include <unordered_map>
#include <queue>
#include <cassert>
#include <cstring>
#include <climits>
#include <memory>
#include <algorithm>
#include <initializer_list>
#include <thread>
#include <atomic>
#include <mutex>
#include <sstream>
#include <numeric>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static void llama_v3_log_internal(llama_v3_log_level level, const char* format, ...);
static void llama_v3_log_callback_default(llama_v3_log_level level, const char * text, void * user_data);
#define LLAMA_V3_LOG_INFO(...)  llama_v3_log_internal(LLAMA_V3_LOG_LEVEL_INFO , __VA_ARGS__)
#define LLAMA_V3_LOG_WARN(...)  llama_v3_log_internal(LLAMA_V3_LOG_LEVEL_WARN , __VA_ARGS__)
#define LLAMA_V3_LOG_ERROR(...) llama_v3_log_internal(LLAMA_V3_LOG_LEVEL_ERROR, __VA_ARGS__)

#if !defined(GGML_USE_CUDA)
#define LLAMA_V3_USE_ALLOCATOR
#else
#define LLAMA_V3_USE_SCRATCH
#define LLAMA_V3_MAX_SCRATCH_BUFFERS 16
#endif


// available llama models
enum e_model3 {
    MODEL_UNKNOWN_3,
    MODEL_3B_3,
    MODEL_7B_3,
    MODEL_13B_3,
    MODEL_30B_3,
    MODEL_34B_3,
    MODEL_65B_3,
    MODEL_70B_3,
};

static const size_t kB3 = 1024;
static const size_t MB3 = 1024*1024;

// computed for n_ctx == 2048
// TODO: dynamically determine these sizes
//       needs modifications in ggml

typedef void (*offload_func_v3_t)(struct ggml_v3_tensor * tensor);

void llama_v3_nop(struct ggml_v3_tensor * tensor) { // don't offload by default
    (void) tensor;
}

//
// ggml helpers
//

static void llv3_graph_compute_helper(std::vector<uint8_t> & buf, ggml_v3_cgraph * graph, int n_threads) {
    struct ggml_v3_cplan plan = ggml_v3_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_v3_graph_compute(graph, &plan);
}


//
// memory sizes (calculated for n_batch == 512)
//

static std::map<e_model3, size_t> MEM_REQ_SCRATCH0_3(int n_ctx)
{
    std::map<e_model3, size_t> k_sizes = {
        { MODEL_3B_3,   ((size_t) n_ctx / 16ull + 156ull) * MB3 },
        { MODEL_7B_3,   ((size_t) n_ctx / 16ull + 164ull) * MB3 },
        { MODEL_13B_3,  ((size_t) n_ctx / 12ull + 184ull) * MB3 },
        { MODEL_30B_3,  ((size_t) n_ctx /  9ull + 224ull) * MB3 },
        { MODEL_34B_3,  ((size_t) n_ctx /  8ull + 256ull) * MB3 }, // guess
        { MODEL_65B_3,  ((size_t) n_ctx /  6ull + 320ull) * MB3 }, // guess
        { MODEL_70B_3,  ((size_t) n_ctx /  7ull + 320ull) * MB3 },
    };
    return k_sizes;
}

static const std::map<e_model3, size_t> & MEM_REQ_SCRATCH1_3()
{
    static std::map<e_model3, size_t> k_sizes = {
        { MODEL_3B_3,  192ull * MB3 },
        { MODEL_7B_3,  224ull * MB3 },
        { MODEL_13B_3, 256ull * MB3 },
        { MODEL_30B_3, 320ull * MB3 },
        { MODEL_34B_3, 380ull * MB3 }, // guess
        { MODEL_65B_3, 448ull * MB3 }, // guess
        { MODEL_70B_3, 448ull * MB3 },
    };
    return k_sizes;
}

// used to store the compute graph tensors + non-scratch data
static const std::map<e_model3, size_t> & MEM_REQ_EVAL_3()
{
    static std::map<e_model3, size_t> k_sizes = {
        { MODEL_3B_3,  16ull * MB3 },
        { MODEL_7B_3,  20ull * MB3 },
        { MODEL_13B_3, 24ull * MB3 },
        { MODEL_30B_3, 32ull * MB3 },
        { MODEL_34B_3, 38ull * MB3 }, // guess
        { MODEL_65B_3, 48ull * MB3 }, // guess
        { MODEL_70B_3, 48ull * MB3 },
    };
    return k_sizes;
}

// amount of VRAM needed per batch size to hold temporary results
// the values for 3b are not derived from testing but instead chosen conservatively
static const std::map<e_model3, size_t> & VRAM_REQ_SCRATCH_BASE_3()
{
    static std::map<e_model3, size_t> k_sizes = {
        { MODEL_3B_3,   512ull * kB3 },
        { MODEL_7B_3,   512ull * kB3 },
        { MODEL_13B_3,  640ull * kB3 },
        { MODEL_30B_3,  768ull * kB3 },
        { MODEL_34B_3,  960ull * kB3 },
        { MODEL_65B_3, 1360ull * kB3 },
        { MODEL_70B_3, 1360ull * kB3 },
    };
    return k_sizes;
}

// amount of VRAM needed per batch size and context to hold temporary results
// the values for 3b are not derived from testing but instead chosen conservatively
static const std::map<e_model3, size_t> & VRAM_REQ_SCRATCH_PER_CONTEXT_3()
{
    static std::map<e_model3, size_t> k_sizes = {
        { MODEL_3B_3,  128ull },
        { MODEL_7B_3,  128ull },
        { MODEL_13B_3, 160ull },
        { MODEL_30B_3, 208ull },
        { MODEL_34B_3, 256ull },
        { MODEL_65B_3, 320ull },
        { MODEL_70B_3, 320ull },
    };
    return k_sizes;
}

// default hparams (LLaMA 7B)
struct llama_v3_hparams {
    uint32_t n_vocab   = 32000;
    uint32_t n_ctx     = 512;   // this is provided as user input?
    uint32_t n_embd    = 4096;
    uint32_t n_mult    = 256;
    uint32_t n_head    = 32;
    uint32_t n_head_kv = 32;
    uint32_t n_layer   = 32;
    uint32_t n_rot     = 64;

    // LLaMAv2
    // TODO: load from model data hparams
    float f_ffn_mult = 1.0f;
    float f_rms_norm_eps = LLAMA_V3_DEFAULT_RMS_EPS;

    float rope_freq_base  = 10000.0f;
    float rope_freq_scale = 1.0f;

    enum llama_v3_ftype ftype = LLAMA_V3_FTYPE_MOSTLY_F16;

    bool operator!=(const llama_v3_hparams & other) const {
        return static_cast<bool>(memcmp(this, &other, sizeof(llama_v3_hparams))); // NOLINT
    }

    uint32_t n_gqa() const {
        return n_head/n_head_kv;
    }

    uint32_t n_embd_head() const {
        return n_embd/n_head;
    }

    uint32_t n_embd_gqa() const {
        return n_embd/n_gqa();
    }

    size_t kv_size() const {
        size_t result = 2ull;
        result *= (size_t) n_embd_gqa();
        result *= (size_t) n_ctx;
        result *= (size_t) n_layer;
        result *= sizeof(ggml_v3_fp16_t);
        return result;
    }
};

struct llama_v3_layer {
    // normalization
    struct ggml_v3_tensor * attention_norm;

    // attention
    struct ggml_v3_tensor * wq;
    struct ggml_v3_tensor * wk;
    struct ggml_v3_tensor * wv;
    struct ggml_v3_tensor * wo;

    // normalization
    struct ggml_v3_tensor * ffn_norm;

    // ff
    struct ggml_v3_tensor * w1;
    struct ggml_v3_tensor * w2;
    struct ggml_v3_tensor * w3;
};

struct llama_v3_kv_cache {
    struct ggml_v3_tensor * k = NULL;
    struct ggml_v3_tensor * v = NULL;

    struct ggml_v3_context * ctx = NULL;

    llama_v3_ctx_buffer buf;

    int n; // number of tokens currently in the cache

    ~llama_v3_kv_cache() {
        if (ctx) {
            ggml_v3_free(ctx);
        }

#ifdef GGML_USE_CUDA
        ggml_v3_cuda_free_data(k);
        ggml_v3_cuda_free_data(v);
#endif // GGML_USE_CUDA
    }
};

struct llama_v3_vocab {
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
};

struct llama_v3_model {
    e_model3 type = MODEL_UNKNOWN_3;

    llama_v3_hparams hparams;

    struct ggml_v3_tensor * tok_embeddings;

    struct ggml_v3_tensor * norm;
    struct ggml_v3_tensor * output;

    std::vector<llama_v3_layer> layers;
    int n_gpu_layers;

    // context
    struct ggml_v3_context * ctx = NULL;

    // the model memory buffer
    llama_v3_ctx_buffer buf;

    // model memory mapped file
    std::unique_ptr<llama_v3_mmap> mapping;

    // objects representing data potentially being locked in memory
    llama_v3_mlock mlock_buf;
    llama_v3_mlock mlock_mmap;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_v3_tensor *>> tensors_by_name;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    llama_v3_vocab vocab;

    ~llama_v3_model() {
        if (ctx) {
            ggml_v3_free(ctx);
        }

#ifdef GGML_USE_CUDA
        for (size_t i = 0; i < tensors_by_name.size(); ++i) {
            ggml_v3_cuda_free_data(tensors_by_name[i].second);
        }
        ggml_v3_cuda_free_scratch();
#elif defined(GGML_USE_CLBLAST)
        for (size_t i = 0; i < tensors_by_name.size(); ++i) {
            ggml_v3_cl_free_data(tensors_by_name[i].second);
        }
#endif
    }
};

struct llama_v3_context {
    llama_v3_context(const llama_v3_model & model) : model(model), t_load_us(model.t_load_us), t_start_us(model.t_start_us) {}
    ~llama_v3_context() {
        if (model_owner) {
            delete &model;
        }

#ifdef LLAMA_V3_USE_ALLOCATOR
        if (alloc) {
            ggml_v3_allocr_free(alloc);
        }
#endif
    }

    std::mt19937 rng;

    bool has_evaluated_once = false;

    int64_t t_sample_us = 0;
    int64_t t_eval_us   = 0;
    int64_t t_p_eval_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_eval   = 0; // number of eval calls
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)

    const llama_v3_model & model;

    bool model_owner = false;

    int64_t t_load_us;
    int64_t t_start_us;

    // key + value cache for the self attention
    struct llama_v3_kv_cache kv_self;

    size_t mem_per_token = 0;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;
    bool logits_all = false;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;

    // reusable buffer for `struct ggml_v3_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;

    // memory buffers used to evaluate the model
    // TODO: move in llama_v3_state
    llama_v3_ctx_buffer buf_compute;

#ifdef LLAMA_V3_USE_ALLOCATOR
    llama_v3_ctx_buffer buf_alloc;
    ggml_v3_allocr * alloc = NULL;
#endif

#ifdef LLAMA_V3_USE_SCRATCH
    llama_v3_ctx_buffer buf_scratch[LLAMA_V3_MAX_SCRATCH_BUFFERS];
    int    buf_last = 0;
    size_t buf_max_size[LLAMA_V3_MAX_SCRATCH_BUFFERS] = { 0 };
#endif


    void use_buf(struct ggml_v3_context * ctx, int i) {
#if defined(LLAMA_V3_USE_SCRATCH)
        size_t last_size = 0;

        if (i == -1) {
            last_size = ggml_v3_set_scratch(ctx, { 0, 0, nullptr, });
        } else {
            auto & buf = buf_scratch[i];
            last_size = ggml_v3_set_scratch(ctx, { 0, buf.size, buf.addr, });
        }

        if (buf_last >= 0) {
            buf_max_size[buf_last] = std::max(buf_max_size[buf_last], last_size);
        }

        buf_last = i;
#else
        (void) i;
        (void) ctx;
#endif
    }

    size_t get_buf_max_mem(int i) const {
#if defined(LLAMA_V3_USE_SCRATCH)
        return buf_max_size[i];
#else
        (void) i;
        return 0;
#endif
    }
};

struct llama_v3_state {
    // We save the log callback globally
    llama_v3_log_callback log_callback = llama_v3_log_callback_default;
    void * log_callback_user_data = nullptr;
};
// global state
static llama_v3_state llv3_g_state;

template <typename T>
static T checked_mul(T a, T b) {
    T ret = a * b;
    if (a != 0 && ret / a != b) {
        throw std::runtime_error(format_old("overflow multiplying %llu * %llu",
                     (unsigned long long) a, (unsigned long long) b));
    }
    return ret;
}

static size_t checked_div(size_t a, size_t b) {
    if (b == 0 || a % b != 0) {
        throw std::runtime_error(format_old("error dividing %zu / %zu", a, b));
    }
    return a / b;
}

static std::string llama_v3_format_tensor_shape(const std::vector<uint32_t> & ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5u", ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), " x %5u", ne.at(i));
    }
    return buf;
}

static size_t llama_v3_calc_tensor_size(const std::vector<uint32_t> & ne, enum ggml_v3_type type) {
    size_t size = ggml_v3_type_size(type);
    for (uint32_t dim : ne) {
        size = checked_mul<size_t>(size, dim);
    }
    return size / ggml_v3_blck_size(type);
}

struct llama_v3_load_tensor {
    std::string name;
    enum ggml_v3_type type = GGML_V3_TYPE_F32;
    std::vector<uint32_t> ne;
    size_t file_off;
    size_t size;
    struct ggml_v3_tensor * ggml_v3_tensor = NULL;
    uint8_t * data;
};

struct llama_v3_load_tensors_map {
    // tensors is kept in a separate vector to preserve file order
    std::vector<llama_v3_load_tensor> tensors;
    std::unordered_map<std::string, size_t> name_to_idx;
};

enum llama_v3_file_version {
    LLAMA_V3_FILE_VERSION_GGML,
    LLAMA_V3_FILE_VERSION_GGMF_V1, // added version field and scores in vocab
    LLAMA_V3_FILE_VERSION_GGJT_V1, // added padding
    LLAMA_V3_FILE_VERSION_GGJT_V2, // changed quantization format
    LLAMA_V3_FILE_VERSION_GGJT_V3, // changed Q4 and Q8 quantization format
};

struct llama_v3_file_loader {
    llama_v3_file file;
    llama_v3_file_version file_version;
    llama_v3_hparams hparams;
    llama_v3_vocab vocab;

    llama_v3_file_loader(const char * fname, llama_v3_load_tensors_map & tensors_map)
        : file(fname, "rb") {
        LLAMA_V3_LOG_INFO("llama.cpp: loading model from %s\n", fname);
        read_magic();
        read_hparams();
        read_vocab();
        read_tensor_metadata(tensors_map);
    }
    void read_magic() {
        uint32_t magic = file.read_u32();

        if (magic == LLAMA_V3_FILE_MAGIC_GGML) {
            file_version = LLAMA_V3_FILE_VERSION_GGML;
            return;
        }

        uint32_t version = file.read_u32();

        switch (magic) {
            case LLAMA_V3_FILE_MAGIC_GGMF:
                switch (version) {
                    case 1: file_version = LLAMA_V3_FILE_VERSION_GGMF_V1; return;
                }
                break;
            case LLAMA_V3_FILE_MAGIC_GGJT:
                switch (version) {
                    case 1: file_version = LLAMA_V3_FILE_VERSION_GGJT_V1; return;
                    case 2: file_version = LLAMA_V3_FILE_VERSION_GGJT_V2; return;
                    case 3: file_version = LLAMA_V3_FILE_VERSION_GGJT_V3; return;
                }
        }

        throw std::runtime_error(format_old("unknown (magic, version) combination: %08x, %08x; is this really a GGML file?",
                     magic, version));
    }
    void read_hparams() {
        hparams.n_vocab = file.read_u32();
        hparams.n_embd  = file.read_u32();
        hparams.n_mult  = file.read_u32();
        hparams.n_head  = file.read_u32();
        hparams.n_layer = file.read_u32();
        hparams.n_rot   = file.read_u32();
        hparams.ftype   = (enum llama_v3_ftype) file.read_u32();

        // LLaMAv2
        // TODO: read from header
        hparams.n_head_kv = hparams.n_head;
    }
    void read_vocab() {
        vocab.id_to_token.resize(hparams.n_vocab);

        for (uint32_t i = 0; i < hparams.n_vocab; i++) {
            uint32_t len = file.read_u32();
            std::string word = file.read_string(len);

            float score = 0.0f;
            file.read_raw(&score, sizeof(score));

            vocab.token_to_id[word] = i;

            auto & tok_score = vocab.id_to_token[i];
            tok_score.tok = std::move(word);
            tok_score.score = score;
        }
    }
    void read_tensor_metadata(llama_v3_load_tensors_map & tensors_map) {
        while (file.tell() < file.size) {
            llama_v3_load_tensor tensor;
            uint32_t n_dims = file.read_u32();
            uint32_t name_len = file.read_u32();
            tensor.type = (enum ggml_v3_type) file.read_u32();
            tensor.ne.resize(n_dims);
            file.read_raw(tensor.ne.data(), sizeof(tensor.ne[0]) * n_dims);
            std::string name = file.read_string(name_len);
            if (n_dims < 1 || n_dims > 2) {
                throw std::runtime_error(format_old("llama.cpp: tensor '%s' should not be %u-dimensional", name.c_str(), n_dims));
            }
            switch (tensor.type) {
                case GGML_V3_TYPE_F32:
                case GGML_V3_TYPE_F16:
                case GGML_V3_TYPE_Q4_0:
                case GGML_V3_TYPE_Q4_1:
                case GGML_V3_TYPE_Q5_0:
                case GGML_V3_TYPE_Q5_1:
                case GGML_V3_TYPE_Q8_0:
                case GGML_V3_TYPE_Q2_K:
                case GGML_V3_TYPE_Q3_K:
                case GGML_V3_TYPE_Q4_K:
                case GGML_V3_TYPE_Q5_K:
                case GGML_V3_TYPE_Q6_K:
                    break;
                default: {
                    throw std::runtime_error(format_old("unrecognized tensor type %u\n", tensor.type));
                }
            }

            // skip to the next multiple of 32 bytes
            if (file_version >= LLAMA_V3_FILE_VERSION_GGJT_V1) {
                file.seek(-static_cast<ptrdiff_t>(file.tell()) & 31, SEEK_CUR);
            }

            tensor.file_off = file.tell();
            tensor.name = name;
            tensor.size = llama_v3_calc_tensor_size(tensor.ne, tensor.type);
            file.seek(tensor.size, SEEK_CUR);

            tensors_map.tensors.push_back(tensor);
            tensors_map.name_to_idx[name] = tensors_map.tensors.size() - 1;
        }
    }
};

struct llama_v3_file_saver {
    llama_v3_file file;
    llama_v3_file_loader * any_file_loader;
    llama_v3_file_saver(const char * fname, llama_v3_file_loader * any_file_loader, enum llama_v3_ftype new_ftype)
        : file(fname, "wb"), any_file_loader(any_file_loader) {
        LLAMA_V3_LOG_INFO("llama.cpp: saving model to %s\n", fname);
        write_magic();
        write_hparams(new_ftype);
        write_vocab();
    }
    void write_magic() {
        file.write_u32(LLAMA_V3_FILE_MAGIC);   // magic
        file.write_u32(LLAMA_V3_FILE_VERSION); // version
    }
    void write_hparams(enum llama_v3_ftype new_ftype) {
        const llama_v3_hparams & hparams = any_file_loader->hparams;
        file.write_u32(hparams.n_vocab);
        file.write_u32(hparams.n_embd);
        file.write_u32(hparams.n_mult);
        file.write_u32(hparams.n_head);
        file.write_u32(hparams.n_layer);
        file.write_u32(hparams.n_rot);
        file.write_u32(new_ftype);
    }
    void write_vocab() {
        if (any_file_loader->file_version == LLAMA_V3_FILE_VERSION_GGML) {
            LLAMA_V3_LOG_WARN("llama.cpp: WARNING: input is an old file that doesn't have scores; will add dummy scores\n");
        }
        uint32_t n_vocab = any_file_loader->hparams.n_vocab;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const auto & token_score = any_file_loader->vocab.id_to_token.at(i);
            file.write_u32((uint32_t) token_score.tok.size());
            file.write_raw(token_score.tok.data(), token_score.tok.size());
            file.write_raw(&token_score.score, sizeof(token_score.score));
        }
    }
    void write_tensor(llama_v3_load_tensor & tensor, enum ggml_v3_type new_type, const void * new_data, size_t new_size) {
        switch (new_type) {
            case GGML_V3_TYPE_F32:
            case GGML_V3_TYPE_F16:
            case GGML_V3_TYPE_Q4_0:
            case GGML_V3_TYPE_Q4_1:
            case GGML_V3_TYPE_Q5_0:
            case GGML_V3_TYPE_Q5_1:
            case GGML_V3_TYPE_Q8_0:
            case GGML_V3_TYPE_Q2_K:
            case GGML_V3_TYPE_Q3_K:
            case GGML_V3_TYPE_Q4_K:
            case GGML_V3_TYPE_Q5_K:
            case GGML_V3_TYPE_Q6_K:
                break;
            default: LLAMA_V3_ASSERT(false);
        }
        file.write_u32((uint32_t) tensor.ne.size());
        file.write_u32((uint32_t) tensor.name.size());
        file.write_u32(new_type);
        file.write_raw(tensor.ne.data(), sizeof(tensor.ne[0]) * tensor.ne.size());
        file.write_raw(tensor.name.data(), tensor.name.size());
        file.seek(-static_cast<ptrdiff_t>(file.tell()) & 31, SEEK_CUR);
        LLAMA_V3_ASSERT(new_size == llama_v3_calc_tensor_size(tensor.ne, new_type));
        file.write_raw(new_data, new_size);
    }
};

struct llama_v3_model_loader {
    std::unique_ptr<llama_v3_file_loader> file_loader;
    llama_v3_load_tensors_map tensors_map;
    bool use_mmap;
    size_t num_ggml_v3_tensors_created = 0;
    struct ggml_v3_context * ggml_v3_ctx = NULL;
    std::unique_ptr<llama_v3_mmap> mapping;

    llama_v3_model_loader(const std::string & fname_base, bool use_mmap) {
        file_loader = std::unique_ptr<llama_v3_file_loader>(new llama_v3_file_loader(fname_base.c_str(), tensors_map));
        if (!llama_v3_mmap::SUPPORTED) {
            use_mmap = false;
        }
        this->use_mmap = use_mmap;
    }

    void calc_sizes(size_t * ctx_size_p, size_t * mmapped_size_p) const {
        *ctx_size_p = *mmapped_size_p = 0;
        for (const llama_v3_load_tensor & lt : tensors_map.tensors) {
            *ctx_size_p += sizeof(struct ggml_v3_tensor) + GGML_V3_OBJECT_SIZE;
            *(use_mmap ? mmapped_size_p : ctx_size_p) += lt.size + 16;
        }
    }

    struct ggml_v3_tensor * get_tensor(const std::string & name, const std::vector<uint32_t> & ne, ggml_v3_backend_type backend) {
        auto it = tensors_map.name_to_idx.find(name);
        if (it == tensors_map.name_to_idx.end()) {
            throw std::runtime_error(std::runtime_error(format_old("llama.cpp: tensor '%s' is missing from model", name.c_str())));
        }
        llama_v3_load_tensor & lt = tensors_map.tensors.at(it->second);
        if (lt.ne != ne) {
            throw std::runtime_error(format_old("llama.cpp: tensor '%s' has wrong shape; expected %s, got %s",
                         name.c_str(), llama_v3_format_tensor_shape(ne).c_str(), llama_v3_format_tensor_shape(lt.ne).c_str()));
        }

        return get_tensor_for(lt, backend);
    }

    struct ggml_v3_tensor * get_tensor_for(llama_v3_load_tensor & lt, ggml_v3_backend_type backend) {
        struct ggml_v3_tensor * tensor;
        if (backend != GGML_V3_BACKEND_CPU) {
            ggml_v3_set_no_alloc(ggml_v3_ctx, true);
        }
        if (lt.ne.size() == 2) {
            tensor = ggml_v3_new_tensor_2d(ggml_v3_ctx, lt.type, lt.ne.at(0), lt.ne.at(1));
        } else {
            LLAMA_V3_ASSERT(lt.ne.size() == 1);
            tensor = ggml_v3_new_tensor_1d(ggml_v3_ctx, lt.type, lt.ne.at(0));
        }
        ggml_v3_set_name(tensor, lt.name.c_str());
        LLAMA_V3_ASSERT(lt.ggml_v3_tensor == NULL); // if this fails, we called get_tensor twice on the same tensor

        if (backend != GGML_V3_BACKEND_CPU) {
            ggml_v3_set_no_alloc(ggml_v3_ctx, use_mmap);
        }
        tensor->backend = backend;
        lt.ggml_v3_tensor = tensor;
        num_ggml_v3_tensors_created++;
        return tensor;
    }

    void done_getting_tensors() const {
        if (num_ggml_v3_tensors_created != tensors_map.tensors.size()) {
            throw std::runtime_error(std::string("llama.cpp: file contained more tensors than expected"));
        }
    }

    void load_all_data(llama_v3_progress_callback progress_callback, void *  progress_callback_user_data, llama_v3_mlock * lmlock) {
        size_t data_size = 0;
        size_t prefetch_size = file_loader->file.size;
        size_t lock_size = 0;
        for (const llama_v3_load_tensor & lt : tensors_map.tensors) {
            data_size += lt.size;
            if (lt.ggml_v3_tensor->backend != GGML_V3_BACKEND_CPU) {
                prefetch_size -= lt.size;
            }
        }

        if (use_mmap) {
            mapping.reset(new llama_v3_mmap(&file_loader->file, prefetch_size, ggml_v3_is_numa()));
            if (lmlock) {
                lmlock->init(mapping->addr);
            }
        }

        size_t done_size = 0;
        for (llama_v3_load_tensor & lt : tensors_map.tensors) {
            if (progress_callback) {
                progress_callback((float) done_size / data_size, progress_callback_user_data);
            }
            LLAMA_V3_ASSERT(lt.ggml_v3_tensor); // unused tensors should have been caught by load_data already
            lt.data = (uint8_t *) lt.ggml_v3_tensor->data;

            // allocate temp buffer if not using mmap
            if (!use_mmap && lt.data == NULL) {
                GGML_V3_ASSERT(lt.ggml_v3_tensor->backend != GGML_V3_BACKEND_CPU);
                lt.data = (uint8_t*)malloc(ggml_v3_nbytes(lt.ggml_v3_tensor));
            }

            load_data_for(lt);

            switch(lt.ggml_v3_tensor->backend) {
                case GGML_V3_BACKEND_CPU:
                    lt.ggml_v3_tensor->data = lt.data;
                    if (use_mmap && lmlock) {
                        lock_size += lt.size;
                        lmlock->grow_to(lock_size);
                    }
                    break;
#if defined(GGML_USE_CUDA)
                case GGML_V3_BACKEND_GPU:
                case GGML_V3_BACKEND_GPU_SPLIT:
                    ggml_v3_cuda_transform_tensor(lt.data, lt.ggml_v3_tensor);
                    if (!use_mmap) {
                        free(lt.data);
                    }
                    break;
#elif defined(GGML_USE_CLBLAST)
                case GGML_V3_BACKEND_GPU:
                    ggml_v3_cl_transform_tensor(lt.data, lt.ggml_v3_tensor);
                    if (!use_mmap) {
                        free(lt.data);
                    }
                    break;
#endif
                default:
                    continue;
            }

            done_size += lt.size;
        }
    }

    void load_data_for(llama_v3_load_tensor & lt) {
        if (use_mmap) {
            lt.data = (uint8_t *) mapping->addr + lt.file_off;
        } else {
            llama_v3_file & file = file_loader->file;
            file.seek(lt.file_off, SEEK_SET);
            file.read_raw(lt.data, lt.size);
        }

        if (0) {
            print_checksum(lt);
        }
    }

    static void print_checksum(llama_v3_load_tensor & lt) {
        uint32_t sum = 0;
        for (size_t i = 0; i < lt.size; i++) {
            uint8_t byte = lt.data[i];
            sum = byte + (sum << 6) + (sum << 16) - sum; // sdbm hash
        }
        LLAMA_V3_LOG_INFO("%s checksum: %#08x (%s, size %zu)\n", lt.name.c_str(), sum,
                llama_v3_format_tensor_shape(lt.ne).c_str(), lt.size);
    }

};

//
// kv cache
//

static bool kv_cache_init(
        const struct llama_v3_hparams & hparams,
             struct llama_v3_kv_cache & cache,
                         ggml_v3_type   wtype,
                               int   n_ctx,
                               int   n_gpu_layers) {
    const int n_embd  = hparams.n_embd_gqa();
    const int n_layer = hparams.n_layer;

    const int64_t n_mem      = n_layer*n_ctx;
    const int64_t n_elements = n_embd*n_mem;

    cache.buf.resize(2u*n_elements*ggml_v3_type_size(wtype) + 2u*MB3);
    cache.n = 0;

    struct ggml_v3_init_params params;
    params.mem_size   = cache.buf.size;
    params.mem_buffer = cache.buf.addr;
    params.no_alloc   = false;

    cache.ctx = ggml_v3_init(params);

    if (!cache.ctx) {
        LLAMA_V3_LOG_ERROR("%s: failed to allocate memory for kv cache\n", __func__);
        return false;
    }

    cache.k = ggml_v3_new_tensor_1d(cache.ctx, wtype, n_elements);
    cache.v = ggml_v3_new_tensor_1d(cache.ctx, wtype, n_elements);
    ggml_v3_set_name(cache.k, "cache_k");
    ggml_v3_set_name(cache.v, "cache_v");

    (void) n_gpu_layers;
#ifdef GGML_USE_CUDA
    if (n_gpu_layers > n_layer + 1) {
        ggml_v3_cuda_assign_buffers_no_scratch(cache.v);
    }
    if (n_gpu_layers > n_layer + 2) {
        ggml_v3_cuda_assign_buffers_no_scratch(cache.k);
    }
#endif // GGML_USE_CUDA

    return true;
}

struct llama_v3_context_params llama_v3_context_default_params() {
    struct llama_v3_context_params result = {
        /*.seed                        =*/ LLAMA_V3_DEFAULT_SEED,
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 512,
        /*.n_gqa                       =*/ 1,
        /*.rms_norm_eps                =*/ LLAMA_V3_DEFAULT_RMS_EPS,
        /*.gpu_layers                  =*/ 0,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.rope_freq_base              =*/ 10000.0f,
        /*.rope_freq_scale             =*/ 1.0f,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.low_vram                    =*/ false,
        /*.mul_mat_q                   =*/ false,
        /*.f16_kv                      =*/ true,
        /*.logits_all                  =*/ false,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_mlock                   =*/ false,
        /*.embedding                   =*/ false,
    };

    return result;
}

struct llama_v3_model_quantize_params llama_v3_model_quantize_default_params() {
    struct llama_v3_model_quantize_params result = {
        /*.nthread                     =*/ 0,
        /*.ftype                       =*/ LLAMA_V3_FTYPE_MOSTLY_Q5_1,
        /*.allow_requantize            =*/ false,
        /*.quantize_output_tensor      =*/ true,
    };

    return result;
}

int llama_v3_max_devices() {
    return LLAMA_V3_MAX_DEVICES;
}

bool llama_v3_mmap_supported() {
    return llama_v3_mmap::SUPPORTED;
}

bool llama_v3_mlock_supported() {
    return llama_v3_mlock::SUPPORTED;
}

int get_blas_batch_mul3(int batch)
{
    return (batch>512?(batch>1024?4:2):1);
}

void llama_v3_backend_init(bool numa) {
    ggml_v3_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_v3_init_params params = { 0, NULL, false };
        struct ggml_v3_context * ctx = ggml_v3_init(params);
        ggml_v3_free(ctx);
    }

    if (numa) {
        ggml_v3_numa_init();
    }


}

void llama_v3_backend_free() {

}

int64_t llama_v3_time_us() {
    return ggml_v3_time_us();
}

//
// model loading
//

static const char * llama_v3_file_version_name(llama_v3_file_version version) {
    switch (version) {
        case LLAMA_V3_FILE_VERSION_GGML: return "'ggml' (old version with low tokenizer quality and no mmap support)";
        case LLAMA_V3_FILE_VERSION_GGMF_V1: return "ggmf v1 (old version with no mmap support)";
        case LLAMA_V3_FILE_VERSION_GGJT_V1: return "ggjt v1 (pre #1405)";
        case LLAMA_V3_FILE_VERSION_GGJT_V2: return "ggjt v2 (pre #1508)";
        case LLAMA_V3_FILE_VERSION_GGJT_V3: return "ggjt v3 (latest)";
    }

    return "unknown";
}

const char * llama_v3_ftype_name(enum llama_v3_ftype ftype) {
    switch (ftype) {
        case LLAMA_V3_FTYPE_ALL_F32:     return "all F32";
        case LLAMA_V3_FTYPE_MOSTLY_F16:  return "mostly F16";
        case LLAMA_V3_FTYPE_MOSTLY_Q4_0: return "mostly Q4_0";
        case LLAMA_V3_FTYPE_MOSTLY_Q4_1: return "mostly Q4_1";
        case LLAMA_V3_FTYPE_MOSTLY_Q4_1_SOME_F16:
                                      return "mostly Q4_1, some F16";
        case LLAMA_V3_FTYPE_MOSTLY_Q5_0: return "mostly Q5_0";
        case LLAMA_V3_FTYPE_MOSTLY_Q5_1: return "mostly Q5_1";
        case LLAMA_V3_FTYPE_MOSTLY_Q8_0: return "mostly Q8_0";
        // K-quants
        case LLAMA_V3_FTYPE_MOSTLY_Q2_K: return "mostly Q2_K";
        case LLAMA_V3_FTYPE_MOSTLY_Q3_K_S: return "mostly Q3_K - Small";
        case LLAMA_V3_FTYPE_MOSTLY_Q3_K_M: return "mostly Q3_K - Medium";
        case LLAMA_V3_FTYPE_MOSTLY_Q3_K_L: return "mostly Q3_K - Large";
        case LLAMA_V3_FTYPE_MOSTLY_Q4_K_S: return "mostly Q4_K - Small";
        case LLAMA_V3_FTYPE_MOSTLY_Q4_K_M: return "mostly Q4_K - Medium";
        case LLAMA_V3_FTYPE_MOSTLY_Q5_K_S: return "mostly Q5_K - Small";
        case LLAMA_V3_FTYPE_MOSTLY_Q5_K_M: return "mostly Q5_K - Medium";
        case LLAMA_V3_FTYPE_MOSTLY_Q6_K: return "mostly Q6_K";
        default:                      return "unknown, may not work";
    }
}

static const char * llama_v3_model_type_name(e_model3 type) {
    switch (type) {
        case MODEL_3B_3: return "3B";
        case MODEL_7B_3: return "7B";
        case MODEL_13B_3: return "13B";
        case MODEL_30B_3: return "30B";
        case MODEL_34B_3: return "34B";
        case MODEL_65B_3: return "65B";
        case MODEL_70B_3: return "70B";
        default: LLAMA_V3_ASSERT(false);
    }
}

static void llama_v3_model_load_internal(
        const std::string & fname,
        llama_v3_model & model,
        llama_v3_vocab & vocab,
        int n_ctx,
        int n_batch,
        int n_gqa,
        float rms_norm_eps,
        int n_gpu_layers,
        int main_gpu,
        const float * tensor_split,
        const bool mul_mat_q,
        float rope_freq_base,
        float rope_freq_scale,
        bool low_vram,
        ggml_v3_type memory_type,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        llama_v3_progress_callback progress_callback,
        void * progress_callback_user_data) {

    model.t_start_us = ggml_v3_time_us();
    size_t blasbatchmul = get_blas_batch_mul3(n_batch);

    std::unique_ptr<llama_v3_model_loader> ml(new llama_v3_model_loader(fname, use_mmap));

    vocab = std::move(ml->file_loader->vocab);
    model.hparams = ml->file_loader->hparams;
    model.n_gpu_layers = n_gpu_layers;
    llama_v3_file_version file_version = ml->file_loader->file_version;

    auto & hparams = model.hparams;

    // TODO: read from file
    hparams.f_rms_norm_eps = rms_norm_eps;

    {
        switch (hparams.n_layer) {
            case 26: model.type = e_model3::MODEL_3B_3; break;
            case 32: model.type = e_model3::MODEL_7B_3; break;
            case 40: model.type = e_model3::MODEL_13B_3; break;
            case 48: model.type = e_model3::MODEL_34B_3; break;
            case 60: model.type = e_model3::MODEL_30B_3; break;
            case 80: model.type = e_model3::MODEL_65B_3; break;
            default:
                {
                    if (hparams.n_layer < 32) {
                        model.type = e_model3::MODEL_7B_3;
                    }
                } break;
        }

        hparams.n_ctx = n_ctx;

        // LLaMAv2
        // TODO: temporary until GGUF
        //patch for llama2 gqa
        if (model.type == e_model3::MODEL_65B_3 && (hparams.n_mult >= 4096 && hparams.n_mult != 5504)) {
            fprintf(stderr, "%s: Applying KCPP Patch for 70B model, setting GQA to 8\n", __func__);
            n_gqa = 8;
        }

        if (model.type == e_model3::MODEL_34B_3) {
	        fprintf(stderr, "%s: Applying KCPP Patch for 34B model, setting GQA to 8\n", __func__);
	        n_gqa = 8;
	    }
        LLAMA_V3_ASSERT(hparams.n_head % n_gqa == 0);
        hparams.n_head_kv = hparams.n_head / n_gqa;
        if (model.type == e_model3::MODEL_65B_3 && n_gqa == 8) {
            LLAMA_V3_LOG_WARN("%s: warning: assuming 70B model based on GQA == %d\n", __func__, n_gqa);
            model.type = e_model3::MODEL_70B_3;
            hparams.f_ffn_mult = 1.3f; // from the params.json of the 70B model
        }

        hparams.rope_freq_base  = rope_freq_base;
        hparams.rope_freq_scale = rope_freq_scale;
    }

    // ref: https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/model.py#L194-L199
    const uint32_t n_ff_raw  = 2*(4*hparams.n_embd)/3;
    const uint32_t n_ff_mult = hparams.f_ffn_mult*n_ff_raw;
    const uint32_t n_ff      = ((n_ff_mult + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;
    //const uint32_t n_ff = 28672;

    {
        LLAMA_V3_LOG_INFO("%s: format     = %s\n",   __func__, llama_v3_file_version_name(file_version));
        LLAMA_V3_LOG_INFO("%s: n_vocab    = %u\n",   __func__, hparams.n_vocab);
        LLAMA_V3_LOG_INFO("%s: n_ctx      = %u\n",   __func__, hparams.n_ctx);
        LLAMA_V3_LOG_INFO("%s: n_embd     = %u\n",   __func__, hparams.n_embd);
        LLAMA_V3_LOG_INFO("%s: n_mult     = %u\n",   __func__, hparams.n_mult);
        LLAMA_V3_LOG_INFO("%s: n_head     = %u\n",   __func__, hparams.n_head);
        LLAMA_V3_LOG_INFO("%s: n_head_kv  = %u\n",   __func__, hparams.n_head_kv);
        LLAMA_V3_LOG_INFO("%s: n_layer    = %u\n",   __func__, hparams.n_layer);
        LLAMA_V3_LOG_INFO("%s: n_rot      = %u\n",   __func__, hparams.n_rot); // a.k.a. n_embd_head, n_head_dim
        LLAMA_V3_LOG_INFO("%s: n_gqa      = %u\n",   __func__, hparams.n_gqa());
        LLAMA_V3_LOG_INFO("%s: rnorm_eps  = %.1e\n", __func__, hparams.f_rms_norm_eps);
        LLAMA_V3_LOG_INFO("%s: n_ff       = %u\n",   __func__, n_ff);
        LLAMA_V3_LOG_INFO("%s: freq_base  = %.1f\n", __func__, hparams.rope_freq_base);
        LLAMA_V3_LOG_INFO("%s: freq_scale = %g\n",   __func__, hparams.rope_freq_scale);
        LLAMA_V3_LOG_INFO("%s: ftype      = %u (%s)\n", __func__, hparams.ftype, llama_v3_ftype_name(hparams.ftype));
        LLAMA_V3_LOG_INFO("%s: model size = %s\n",   __func__, llama_v3_model_type_name(model.type));
    }

    if (file_version < LLAMA_V3_FILE_VERSION_GGJT_V2) {
        if (hparams.ftype != LLAMA_V3_FTYPE_ALL_F32     &&
            hparams.ftype != LLAMA_V3_FTYPE_MOSTLY_F16  &&
            hparams.ftype != LLAMA_V3_FTYPE_MOSTLY_Q8_0) {
            printf("\nthis format is no longer supported (see https://github.com/ggerganov/llama.cpp/pull/1405)");
        }
    }

    if (file_version < LLAMA_V3_FILE_VERSION_GGJT_V3) {
        if (hparams.ftype == LLAMA_V3_FTYPE_MOSTLY_Q4_0 ||
            hparams.ftype == LLAMA_V3_FTYPE_MOSTLY_Q4_1 ||
            hparams.ftype == LLAMA_V3_FTYPE_MOSTLY_Q8_0) {
            printf("\nthis format is no longer supported (see https://github.com/ggerganov/llama.cpp/pull/1508)");
        }
    }

    if (vocab_only) {
        return;
    }

    auto & ctx = model.ctx;

    size_t ctx_size;
    size_t mmapped_size;
    ml->calc_sizes(&ctx_size, &mmapped_size);
    LLAMA_V3_LOG_INFO("%s: ggml ctx size = %7.2f MB\n", __func__, ctx_size/1024.0/1024.0);

    // create the ggml context
    {
        model.buf.resize(ctx_size);
        if (use_mlock) {
            model.mlock_buf.init   (model.buf.addr);
            model.mlock_buf.grow_to(model.buf.size);
        }

        struct ggml_v3_init_params params = {
            /*.mem_size   =*/ model.buf.size,
            /*.mem_buffer =*/ model.buf.addr,
            /*.no_alloc   =*/ ml->use_mmap,
        };

        model.ctx = ggml_v3_init(params);
        if (!model.ctx) {
            throw std::runtime_error(format_old("ggml_v3_init() failed"));
        }
    }

    (void) main_gpu;
    (void) mul_mat_q;
#if defined(GGML_USE_CUDA)
    LLAMA_V3_LOG_INFO("%s: using CUDA for GPU acceleration\n", __func__);
    ggml_v3_cuda_set_main_device(main_gpu);
    ggml_v3_cuda_set_mul_mat_q(mul_mat_q);
#define LLAMA_V3_BACKEND_OFFLOAD       GGML_V3_BACKEND_GPU
#define LLAMA_V3_BACKEND_OFFLOAD_SPLIT GGML_V3_BACKEND_GPU_SPLIT
#elif defined(GGML_USE_CLBLAST)
    LLAMA_V3_LOG_INFO("%s: using OpenCL for GPU acceleration\n", __func__);
#define LLAMA_V3_BACKEND_OFFLOAD       GGML_V3_BACKEND_GPU
#define LLAMA_V3_BACKEND_OFFLOAD_SPLIT GGML_V3_BACKEND_GPU
#else
#define LLAMA_V3_BACKEND_OFFLOAD       GGML_V3_BACKEND_CPU
#define LLAMA_V3_BACKEND_OFFLOAD_SPLIT GGML_V3_BACKEND_CPU
#endif

    // prepare memory for the weights
    size_t vram_weights = 0;
    size_t vram_scratch = 0;
    {
        const uint32_t n_embd     = hparams.n_embd;
        const uint32_t n_embd_gqa = hparams.n_embd_gqa();
        const uint32_t n_layer    = hparams.n_layer;
        const uint32_t n_vocab    = hparams.n_vocab;

        ml->ggml_v3_ctx = ctx;

        model.tok_embeddings = ml->get_tensor("tok_embeddings.weight", {n_embd, n_vocab}, GGML_V3_BACKEND_CPU);

        // "output" tensor
        {
            ggml_v3_backend_type backend_norm;
            ggml_v3_backend_type backend_output;
            if (n_gpu_layers > int(n_layer)) { // NOLINT
                // norm is not performance relevant on its own but keeping it in VRAM reduces data copying
                // on Windows however this is detrimental unless everything is on the GPU
#ifndef _WIN32
                backend_norm = low_vram ? GGML_V3_BACKEND_CPU : LLAMA_V3_BACKEND_OFFLOAD;
#else
                backend_norm = low_vram || n_gpu_layers <= (int) n_layer + 2 ? GGML_V3_BACKEND_CPU : LLAMA_V3_BACKEND_OFFLOAD;
#endif // _WIN32

                backend_output = LLAMA_V3_BACKEND_OFFLOAD_SPLIT;
            } else {
                backend_norm = GGML_V3_BACKEND_CPU;
                backend_output = GGML_V3_BACKEND_CPU;
            }

            model.norm   = ml->get_tensor("norm.weight",   {n_embd},          backend_norm);
            model.output = ml->get_tensor("output.weight", {n_embd, n_vocab}, backend_output);
            if (backend_norm == GGML_V3_BACKEND_GPU) {
                vram_weights += ggml_v3_nbytes(model.norm);
            }
            if (backend_output == GGML_V3_BACKEND_GPU_SPLIT) {
                vram_weights += ggml_v3_nbytes(model.output);
            }
        }

        const int i_gpu_start = n_layer - n_gpu_layers;

        model.layers.resize(n_layer);
        for (uint32_t i = 0; i < n_layer; ++i) {
            const ggml_v3_backend_type backend = int(i) < i_gpu_start ? GGML_V3_BACKEND_CPU : LLAMA_V3_BACKEND_OFFLOAD; // NOLINT
            const ggml_v3_backend_type backend_split = int(i) < i_gpu_start ? GGML_V3_BACKEND_CPU : LLAMA_V3_BACKEND_OFFLOAD_SPLIT; // NOLINT

            auto & layer = model.layers[i];

            std::string layers_i = "layers." + std::to_string(i);

            layer.attention_norm = ml->get_tensor(layers_i + ".attention_norm.weight", {n_embd}, backend);

            layer.wq = ml->get_tensor(layers_i + ".attention.wq.weight", {n_embd, n_embd},     backend_split);
            layer.wk = ml->get_tensor(layers_i + ".attention.wk.weight", {n_embd, n_embd_gqa}, backend_split);
            layer.wv = ml->get_tensor(layers_i + ".attention.wv.weight", {n_embd, n_embd_gqa}, backend_split);
            layer.wo = ml->get_tensor(layers_i + ".attention.wo.weight", {n_embd, n_embd},     backend_split);

            layer.ffn_norm = ml->get_tensor(layers_i + ".ffn_norm.weight", {n_embd}, backend);

            layer.w1 = ml->get_tensor(layers_i + ".feed_forward.w1.weight", {n_embd,   n_ff}, backend_split);
            layer.w2 = ml->get_tensor(layers_i + ".feed_forward.w2.weight", {  n_ff, n_embd}, backend_split);
            layer.w3 = ml->get_tensor(layers_i + ".feed_forward.w3.weight", {n_embd,   n_ff}, backend_split);

            if (backend == GGML_V3_BACKEND_GPU) {
                vram_weights +=
                    ggml_v3_nbytes(layer.attention_norm) + ggml_v3_nbytes(layer.wq) + ggml_v3_nbytes(layer.wk)             +
                    ggml_v3_nbytes(layer.wv)             + ggml_v3_nbytes(layer.wo) + ggml_v3_nbytes(layer.ffn_norm) +
                    ggml_v3_nbytes(layer.w1)             + ggml_v3_nbytes(layer.w2) + ggml_v3_nbytes(layer.w3);
            }
        }
    }

    ml->done_getting_tensors();

    // print memory requirements
    {
        const size_t scale = memory_type == GGML_V3_TYPE_F32 ? 2 : 1;

        // this is the total memory required to run the inference
        size_t mem_required =
            ctx_size +
            mmapped_size - vram_weights; // weights in VRAM not in memory

#ifndef LLAMA_V3_USE_ALLOCATOR
        mem_required +=
            blasbatchmul*MEM_REQ_SCRATCH0_3(hparams.n_ctx).at(model.type) +
            blasbatchmul*MEM_REQ_SCRATCH1_3().at(model.type) +
            blasbatchmul*MEM_REQ_EVAL_3().at(model.type);
#endif

        // this is the memory required by one llama_v3_state
        const size_t mem_required_state =
            scale*hparams.kv_size();

        LLAMA_V3_LOG_INFO("%s: mem required  = %7.2f MB (+ %7.2f MB per state)\n", __func__,
                mem_required / 1024.0 / 1024.0, mem_required_state / 1024.0 / 1024.0);

        (void) vram_scratch;
        (void) n_batch;
#ifdef GGML_USE_CUDA
        if (low_vram) {
            LLAMA_V3_LOG_INFO("%s: not allocating a VRAM scratch buffer due to low VRAM option\n", __func__);
            ggml_v3_cuda_set_scratch_size(0); // disable scratch
        } else {
            const size_t vram_scratch_base = VRAM_REQ_SCRATCH_BASE_3().at(model.type);
            const size_t vram_scratch_per_context = VRAM_REQ_SCRATCH_PER_CONTEXT_3().at(model.type);
            vram_scratch = n_batch * (vram_scratch_base + n_ctx * vram_scratch_per_context);
            ggml_v3_cuda_set_scratch_size(vram_scratch);
            if (n_gpu_layers > 0) {
                LLAMA_V3_LOG_INFO("%s: allocating batch_size x (%zd kB + n_ctx x %zd B) = %zd MB VRAM for the scratch buffer\n",
                        __func__, vram_scratch_base / kB3, vram_scratch_per_context,
                        (vram_scratch + MB3 - 1) / MB3); // round up
            }
        }
#endif // GGML_USE_CUDA

#if defined(GGML_USE_CUDA) || defined(GGML_USE_CLBLAST)
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));

        LLAMA_V3_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_gpu);
        if (n_gpu_layers > (int) hparams.n_layer) {
            LLAMA_V3_LOG_INFO("%s: offloading non-repeating layers to GPU\n", __func__);
        }
        size_t vram_kv_cache = 0;

#ifdef GGML_USE_CUDA
        const int max_backend_supported_layers = hparams.n_layer + 3;
        const int max_offloadable_layers = low_vram ? hparams.n_layer + 1 : hparams.n_layer + 3;
        if (n_gpu_layers > (int) hparams.n_layer + 1) {
            if (low_vram) {
                LLAMA_V3_LOG_INFO("%s: cannot offload v cache to GPU due to low VRAM option\n", __func__);
            } else {
                LLAMA_V3_LOG_INFO("%s: offloading v cache to GPU\n", __func__);
                vram_kv_cache += hparams.kv_size() / 2;
            }
        }
        if (n_gpu_layers > (int) hparams.n_layer + 2) {
            if (low_vram) {
                LLAMA_V3_LOG_WARN("%s: cannot offload k cache to GPU due to low VRAM option\n", __func__);
            } else {
                LLAMA_V3_LOG_INFO("%s: offloading k cache to GPU\n", __func__);
                vram_kv_cache += hparams.kv_size() / 2;
            }
        }
#elif defined(GGML_USE_CLBLAST)
        const int max_backend_supported_layers = hparams.n_layer + 1;
        const int max_offloadable_layers = hparams.n_layer + 1;
#endif // GGML_USE_CUDA

        LLAMA_V3_LOG_INFO("%s: offloaded %d/%d layers to GPU\n",
                __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
        LLAMA_V3_LOG_INFO("%s: total VRAM used: %zu MB\n",
                __func__, (vram_weights + vram_scratch + vram_kv_cache + MB3 - 1) / MB3); // round up
#else
        (void) n_gpu_layers;
#endif // defined(GGML_USE_CUDA) || defined(GGML_USE_CLBLAST)
    }

    // populate `tensors_by_name`
    for (llama_v3_load_tensor & lt : ml->tensors_map.tensors) {
        model.tensors_by_name.emplace_back(lt.name, lt.ggml_v3_tensor);
    }

    (void) tensor_split;
#if defined(GGML_USE_CUDA)
    {
        ggml_v3_cuda_set_tensor_split(tensor_split);
    }
#endif

    ml->load_all_data(progress_callback, progress_callback_user_data, use_mlock ? &model.mlock_mmap : NULL);

    if (progress_callback) {
        progress_callback(1.0f, progress_callback_user_data);
    }

    model.mapping = std::move(ml->mapping);

    // loading time will be recalculate after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model.t_load_us = ggml_v3_time_us() - model.t_start_us;
}

static bool llama_v3_model_load(
        const std::string & fname,
        llama_v3_model & model,
        llama_v3_vocab & vocab,
        int n_ctx,
        int n_batch,
        int n_gqa,
        float rms_norm_eps,
        int n_gpu_layers,
        int main_gpu,
        const float * tensor_split,
        const bool mul_mat_q,
        float rope_freq_base,
        float rope_freq_scale,
        bool low_vram,
        ggml_v3_type memory_type,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        llama_v3_progress_callback progress_callback,
        void *progress_callback_user_data) {
    try {
        llama_v3_model_load_internal(fname, model, vocab, n_ctx, n_batch, n_gqa, rms_norm_eps, n_gpu_layers,
                                  main_gpu, tensor_split, mul_mat_q, rope_freq_base, rope_freq_scale, low_vram, memory_type,
                                  use_mmap, use_mlock, vocab_only, progress_callback, progress_callback_user_data);
        return true;
    } catch (const std::exception & err) {
        LLAMA_V3_LOG_ERROR("error loading model: %s\n", err.what());
        return false;
    }
}

static struct ggml_v3_cgraph * llama_v3_build_graph(
         llama_v3_context & lctx,
     const llama_v3_token * tokens,
           const float * embd,
                   int   n_tokens,
                   int   n_past) {

    LLAMA_V3_ASSERT((!tokens && embd) || (tokens && !embd));

    const int N = n_tokens;

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;

    const auto & kv_self = lctx.kv_self;

    LLAMA_V3_ASSERT(!!kv_self.ctx);

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_layer     = hparams.n_layer;
    const int64_t n_ctx       = hparams.n_ctx;
    const int64_t n_head      = hparams.n_head;
    const int64_t n_head_kv   = hparams.n_head_kv;
    const int64_t n_embd_head = hparams.n_embd_head();
    const int64_t n_embd_gqa  = hparams.n_embd_gqa();

    LLAMA_V3_ASSERT(n_embd_head == hparams.n_rot);

    const float freq_base  = hparams.rope_freq_base;
    const float freq_scale = hparams.rope_freq_scale;
    const float rms_norm_eps = hparams.f_rms_norm_eps;

    const int n_gpu_layers = model.n_gpu_layers;

    auto & mem_per_token = lctx.mem_per_token;
    auto & buf_compute   = lctx.buf_compute;


    struct ggml_v3_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.addr,
        /*.no_alloc   =*/ false,
    };

#ifdef LLAMA_V3_USE_ALLOCATOR
    params.no_alloc = true;
#endif

    struct ggml_v3_context * ctx0 = ggml_v3_init(params);

    ggml_v3_cgraph * gf = ggml_v3_new_graph_custom(ctx0, GGML_V3_MAX_NODES, false);

    struct ggml_v3_tensor * cur;
    struct ggml_v3_tensor * inpL;

    if (tokens) {
        struct ggml_v3_tensor * inp_tokens = ggml_v3_new_tensor_1d(ctx0, GGML_V3_TYPE_I32, N);

#ifdef LLAMA_V3_USE_ALLOCATOR
        ggml_v3_allocr_alloc(lctx.alloc, inp_tokens);
        if (!ggml_v3_allocr_is_measure(lctx.alloc)) {
            memcpy(inp_tokens->data, tokens, N*ggml_v3_element_size(inp_tokens));
        }
#else
        memcpy(inp_tokens->data, tokens, N*ggml_v3_element_size(inp_tokens));
#endif
        ggml_v3_set_name(inp_tokens, "inp_tokens");

        inpL = ggml_v3_get_rows(ctx0, model.tok_embeddings, inp_tokens);
    } else {


        inpL = ggml_v3_new_tensor_2d(ctx0, GGML_V3_TYPE_F32, n_embd, N);

#ifdef LLAMA_V3_USE_ALLOCATOR
        ggml_v3_allocr_alloc(lctx.alloc, inpL);
        if (!ggml_v3_allocr_is_measure(lctx.alloc)) {
            memcpy(inpL->data, embd, N * n_embd * ggml_v3_element_size(inpL));
        }
#else
        memcpy(inpL->data, embd, N * n_embd * ggml_v3_element_size(inpL));
#endif
    }

    const int i_gpu_start = n_layer - n_gpu_layers;
    (void) i_gpu_start;

    // offload functions set the tensor output backend to GPU
    // tensors are GPU-accelerated if any input or the output has been offloaded
    //
    // with the low VRAM option VRAM scratch is disabled in llama_v3_load_model_internal
    // in that case ggml_v3_cuda_assign_buffers has no effect
    offload_func_v3_t offload_func_nr = llama_v3_nop; // nr = non-repeating
    offload_func_v3_t offload_func_kq = llama_v3_nop;
    offload_func_v3_t offload_func_v  = llama_v3_nop;

#ifdef GGML_USE_CUDA
    if (n_gpu_layers > n_layer) {
        offload_func_nr = ggml_v3_cuda_assign_buffers;
    }
    if (n_gpu_layers > n_layer + 1) {
        offload_func_v  = ggml_v3_cuda_assign_buffers;
    }
    if (n_gpu_layers > n_layer + 2) {
        offload_func_kq = ggml_v3_cuda_assign_buffers;
    }
#endif // GGML_USE_CUDA

    struct ggml_v3_tensor * KQ_scale = ggml_v3_new_tensor_1d(ctx0, GGML_V3_TYPE_F32, 1);
#ifdef LLAMA_V3_USE_ALLOCATOR
    ggml_v3_allocr_alloc(lctx.alloc, KQ_scale);
    if (!ggml_v3_allocr_is_measure(lctx.alloc)) {
        ggml_v3_set_f32(KQ_scale, 1.0f/sqrtf(float(n_embd)/n_head));
    }
#else
    ggml_v3_set_f32(KQ_scale, 1.0f/sqrtf(float(n_embd)/n_head));
#endif

    float KQ_scale_float = 1.0f/sqrtf(float(n_embd)/n_head);

    ggml_v3_set_name(KQ_scale, "1/sqrt(n_embd_head)");

    for (int il = 0; il < n_layer; ++il) {
        ggml_v3_format_name(inpL, "layer_inp_%d", il);

        offload_func_v3_t offload_func = llama_v3_nop;

#ifdef GGML_USE_CUDA
        if (il >= i_gpu_start) {
            offload_func = ggml_v3_cuda_assign_buffers;
        }
#endif // GGML_USE_CUDA

        struct ggml_v3_tensor * inpSA = inpL;

        lctx.use_buf(ctx0, 0);

        // norm
        {
            cur = ggml_v3_rms_norm(ctx0, inpL, rms_norm_eps);
            offload_func(cur);
            ggml_v3_set_name(cur, "rms_norm_0");

            // cur = cur*attention_norm(broadcasted)
            cur = ggml_v3_mul(ctx0, cur, model.layers[il].attention_norm);
            offload_func(cur);
            ggml_v3_set_name(cur, "attention_norm_0");
        }

        // self-attention
        {
            // compute Q and K and RoPE them
            struct ggml_v3_tensor * tmpk = ggml_v3_mul_mat(ctx0, model.layers[il].wk, cur);
            offload_func_kq(tmpk);
            ggml_v3_set_name(tmpk, "tmpk");

            struct ggml_v3_tensor * tmpq = ggml_v3_mul_mat(ctx0, model.layers[il].wq, cur);
            offload_func_kq(tmpq);
            ggml_v3_set_name(tmpq, "tmpq");

            struct ggml_v3_tensor * KQ_pos = ggml_v3_new_tensor_1d(ctx0, GGML_V3_TYPE_I32, n_tokens);
            ggml_v3_set_name(KQ_pos, "KQ_pos");

#ifdef LLAMA_V3_USE_ALLOCATOR
            offload_func_kq(KQ_pos); //don't offload rope for cublas, its broken now since ring buffer was added
            ggml_v3_allocr_alloc(lctx.alloc, KQ_pos);
            if (!ggml_v3_allocr_is_measure(lctx.alloc)) {
               int * data = (int *) KQ_pos->data;
                for (int i = 0; i < N; ++i) {
                    data[i] = n_past + i;
                }
            }
#else
            {
                int * data = (int *) KQ_pos->data;
                for (int i = 0; i < N; ++i) {
                    data[i] = n_past + i;
                }
            }
#endif

            struct ggml_v3_tensor *Kcur = ggml_v3_rope_custom_inplace(ctx0, ggml_v3_reshape_3d(ctx0, tmpk, n_embd_head, n_head_kv, N), KQ_pos, n_embd_head, 0, 0, 0, freq_base, freq_scale, 0, 1, 32, 1);
            offload_func_kq(Kcur);
            ggml_v3_set_name(Kcur, "Kcur");

            struct ggml_v3_tensor *Qcur = ggml_v3_rope_custom_inplace(ctx0, ggml_v3_reshape_3d(ctx0, tmpq, n_embd_head, n_head, N), KQ_pos, n_embd_head, 0, 0, 0, freq_base, freq_scale, 0, 1, 32, 1);
            offload_func_kq(Qcur);
            ggml_v3_set_name(Qcur, "Qcur");

            // store key and value to memory
            {
                // compute the transposed [N, n_embd] V matrix

                struct ggml_v3_tensor * tmpv = ggml_v3_mul_mat(ctx0, model.layers[il].wv, cur);
                offload_func_v(tmpv);
                ggml_v3_set_name(tmpv, "tmpv");

                struct ggml_v3_tensor * Vcur = ggml_v3_transpose(ctx0, ggml_v3_reshape_2d(ctx0, tmpv, n_embd_gqa, N));
                offload_func_v(Vcur);
                ggml_v3_set_name(Vcur, "Vcur");

                struct ggml_v3_tensor * k = ggml_v3_view_1d(ctx0, kv_self.k, N*n_embd_gqa, (ggml_v3_element_size(kv_self.k)*n_embd_gqa)*(il*n_ctx + n_past));
                offload_func_kq(k);
                ggml_v3_set_name(k, "k");

                struct ggml_v3_tensor * v = ggml_v3_view_2d(ctx0, kv_self.v, N, n_embd_gqa,
                        (   n_ctx)*ggml_v3_element_size(kv_self.v),
                        (il*n_ctx)*ggml_v3_element_size(kv_self.v)*n_embd_gqa + n_past*ggml_v3_element_size(kv_self.v));
                offload_func_v(v);
                ggml_v3_set_name(v, "v");

                // important: storing RoPE-ed version of K in the KV cache!
                ggml_v3_build_forward_expand(gf, ggml_v3_cpy(ctx0, Kcur, k));
                ggml_v3_build_forward_expand(gf, ggml_v3_cpy(ctx0, Vcur, v));
            }

            struct ggml_v3_tensor * Q =
                ggml_v3_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);
            offload_func_kq(Q);
            ggml_v3_set_name(Q, "Q");

            struct ggml_v3_tensor * K =
                ggml_v3_view_3d(ctx0, kv_self.k,
                        n_embd_head, n_past + N, n_head_kv,
                        ggml_v3_element_size(kv_self.k)*n_embd_gqa,
                        ggml_v3_element_size(kv_self.k)*n_embd_head,
                        ggml_v3_element_size(kv_self.k)*n_embd_gqa*n_ctx*il);
            offload_func_kq(K);
            ggml_v3_set_name(K, "K");

            // K * Q
            struct ggml_v3_tensor * KQ = ggml_v3_mul_mat(ctx0, K, Q);
            offload_func_kq(KQ);
            ggml_v3_set_name(KQ, "KQ");

            // KQ_scaled = KQ / sqrt(n_embd_head)
            // KQ_scaled shape [n_past + N, N, n_head, 1]
            struct ggml_v3_tensor * KQ_scaled = ggml_v3_scale_inplace(ctx0, KQ, KQ_scale_float);
            offload_func_kq(KQ_scaled);
            ggml_v3_set_name(KQ_scaled, "KQ_scaled");

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_v3_tensor * KQ_masked = ggml_v3_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);
            offload_func_kq(KQ_masked);
            ggml_v3_set_name(KQ_masked, "KQ_masked");

            // KQ = soft_max(KQ_masked)
            struct ggml_v3_tensor * KQ_soft_max = ggml_v3_soft_max_inplace(ctx0, KQ_masked);
            offload_func_v(KQ_soft_max);
            ggml_v3_set_name(KQ_soft_max, "KQ_soft_max");

            // split cached V into n_head heads
            struct ggml_v3_tensor * V =
                ggml_v3_view_3d(ctx0, kv_self.v,
                        n_past + N, n_embd_head, n_head_kv,
                        ggml_v3_element_size(kv_self.v)*n_ctx,
                        ggml_v3_element_size(kv_self.v)*n_ctx*n_embd_head,
                        ggml_v3_element_size(kv_self.v)*n_ctx*n_embd_gqa*il);
            offload_func_v(V);
            ggml_v3_set_name(V, "V");

#if 1
            struct ggml_v3_tensor * KQV = ggml_v3_mul_mat(ctx0, V, KQ_soft_max);
            offload_func_v(KQV);
            ggml_v3_set_name(KQV, "KQV");
#else
            // make V contiguous in memory to speed up the matmul, however we waste time on the copy
            // on M1 this is faster for the perplexity computation, but ~5% slower for the single-token generation
            // is there a better way?
            struct ggml_v3_tensor * V_cont = ggml_v3_cpy(ctx0, V, ggml_v3_new_tensor_3d(ctx0, kv_self.v->type, n_past + N, n_embd_head, n_head));
            struct ggml_v3_tensor * KQV = ggml_v3_mul_mat(ctx0, V_cont, KQ_soft_max);
#endif

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_v3_tensor * KQV_merged = ggml_v3_permute(ctx0, KQV, 0, 2, 1, 3);
            offload_func_v(KQV_merged);
            ggml_v3_set_name(KQV_merged, "KQV_merged");

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_v3_cpy(ctx0,
                    KQV_merged,
                    ggml_v3_new_tensor_2d(ctx0, GGML_V3_TYPE_F32, n_embd, N));
            offload_func_v(cur);
            ggml_v3_set_name(cur, "KQV_merged_contiguous");

            // projection (no bias)
            cur = ggml_v3_mul_mat(ctx0,
                    model.layers[il].wo,
                    cur);
            offload_func(cur);
            ggml_v3_set_name(cur, "result_wo");
        }

        lctx.use_buf(ctx0, 1);

        struct ggml_v3_tensor * inpFF = ggml_v3_add(ctx0, cur, inpSA);
        offload_func(inpFF);
        ggml_v3_set_name(inpFF, "inpFF");

        // feed-forward network
        {
            // norm
            {
                cur = ggml_v3_rms_norm(ctx0, inpFF, rms_norm_eps);
                offload_func(cur);
                ggml_v3_set_name(cur, "rms_norm_1");

                // cur = cur*ffn_norm(broadcasted)
                cur = ggml_v3_mul(ctx0, cur, model.layers[il].ffn_norm);
                offload_func(cur);
                ggml_v3_set_name(cur, "ffn_norm");
            }

            struct ggml_v3_tensor * tmp = ggml_v3_mul_mat(ctx0,
                    model.layers[il].w3,
                    cur);
            offload_func(tmp);
            ggml_v3_set_name(tmp, "result_w3");

            cur = ggml_v3_mul_mat(ctx0,
                    model.layers[il].w1,
                    cur);
            offload_func(cur);
            ggml_v3_set_name(cur, "result_w1");

            // SILU activation
            cur = ggml_v3_silu(ctx0, cur);
            offload_func(cur);
            ggml_v3_set_name(cur, "silu");

            cur = ggml_v3_mul(ctx0, cur, tmp);
            offload_func(cur);
            ggml_v3_set_name(cur, "silu_x_result_w3");

            cur = ggml_v3_mul_mat(ctx0,
                    model.layers[il].w2,
                    cur);
            offload_func(cur);
            ggml_v3_set_name(cur, "result_w2");
        }

        cur = ggml_v3_add(ctx0, cur, inpFF);
        offload_func(cur);
        ggml_v3_set_name(cur, "inpFF_+_result_w2");

        // input for next layer
        inpL = cur;
    }

    lctx.use_buf(ctx0, 0);

    // norm
    {
        cur = ggml_v3_rms_norm(ctx0, inpL, rms_norm_eps);
        offload_func_nr(cur);
        ggml_v3_set_name(cur, "rms_norm_2");

        // cur = cur*norm(broadcasted)
        cur = ggml_v3_mul(ctx0, cur, model.norm);
        // offload_func_nr(cur); // TODO CPU + GPU mirrored backend
        ggml_v3_set_name(cur, "result_norm");
    }

    // lm_head
    cur = ggml_v3_mul_mat(ctx0, model.output, cur);
    ggml_v3_set_name(cur, "result_output");

    lctx.use_buf(ctx0, -1);

    // logits -> probs
    //cur = ggml_v3_soft_max_inplace(ctx0, cur);

    ggml_v3_build_forward_expand(gf, cur);

    if (mem_per_token == 0) {
        mem_per_token = ggml_v3_used_mem(ctx0)/N;
    }

#if 0
    LLAMA_V3_LOG_INFO("\n%s: used_mem: eval ctx %.3f MB, scratch %.3f MB %.3f MB, work buf %.3f MB, n_past = %d, N = %d\n", __func__,
            ggml_v3_used_mem(ctx0)/1024.0/1024.0,
            lctx.get_buf_max_mem(0)/1024.0/1024.0,
            lctx.get_buf_max_mem(1)/1024.0/1024.0,
            lctx.work_buffer.size()/1024.0/1024.0,
            n_past, N);
#endif

    ggml_v3_free(ctx0);

    return gf;
}

// evaluate the transformer
//
//   - lctx:      llama context
//   - tokens:    new batch of tokens to process
//   - embd       embeddings input
//   - n_tokens   number of tokens
//   - n_past:    the context size so far
//   - n_threads: number of threads to use
//
static bool llama_v3_eval_internal(
         llama_v3_context & lctx,
     const llama_v3_token * tokens,
           const float * embd,
                   int   n_tokens,
                   int   n_past,
                   int   n_threads,
            const char * cgraph_fname) {

    LLAMA_V3_ASSERT((!tokens && embd) || (tokens && !embd));

    LLAMA_V3_ASSERT(n_tokens > 0);
    LLAMA_V3_ASSERT(n_past >= 0);
    LLAMA_V3_ASSERT(n_threads > 0);
    // TODO: keep the values of n_batch and n_ctx
    // LLAMA_V3_ASSERT(n_tokens <= n_batch);
    // LLAMA_V3_ASSERT(n_past + n_tokens <= n_ctx);

    const int64_t t_start_us = ggml_v3_time_us();



    const int N = n_tokens;

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;

    const auto & kv_self = lctx.kv_self;

    LLAMA_V3_ASSERT(!!kv_self.ctx);

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_vocab     = hparams.n_vocab;

#ifdef LLAMA_V3_USE_ALLOCATOR
    ggml_v3_allocr_reset(lctx.alloc);
#endif

    ggml_v3_cgraph * gf = llama_v3_build_graph(lctx, tokens, embd, n_tokens, n_past);

#ifdef LLAMA_V3_USE_ALLOCATOR
    ggml_v3_allocr_alloc_graph(lctx.alloc, gf);
#endif

    // LLAMA_V3_LOG_INFO("graph build time: %.3f ms (%d nodes, %d leafs)\n", (ggml_v3_time_us() - t_start_us)/1000.0, gf->n_nodes, gf->n_leafs);

    // for big prompts, if BLAS is enabled, it is better to use only one thread
    // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
    n_threads = N >= 32 && ggml_v3_cpu_has_blas() && !ggml_v3_cpu_has_gpublas() ? 1 : n_threads;

    struct ggml_v3_tensor * res = gf->nodes[gf->n_nodes - 1];
    struct ggml_v3_tensor * embeddings = gf->nodes[gf->n_nodes - 2];

    LLAMA_V3_ASSERT(strcmp(res->name, "result_output") == 0);
    LLAMA_V3_ASSERT(strcmp(embeddings->name, "result_norm") == 0);


    llv3_graph_compute_helper(lctx.work_buffer, gf, n_threads);


    // update kv token count
    lctx.kv_self.n = n_past + N;

    if (cgraph_fname) {
        ggml_v3_graph_export(gf, cgraph_fname);
    }

#ifdef GGML_V3_PERF
    // print timing information per ggml operation (for debugging purposes)
    // requires GGML_V3_PERF to be defined
    ggml_v3_graph_print(gf);
#endif

    // plot the computation graph in dot format (for debugging purposes)
    //if (n_past%100 == 0) {
    //    ggml_v3_graph_dump_dot(gf, NULL, "llama.dot");
    //}

    // extract logits
    {
        auto & logits_out = lctx.logits;

        if (lctx.logits_all) {
            logits_out.resize(n_vocab * N);
            memcpy(logits_out.data(), (float *) ggml_v3_get_data(res), sizeof(float)*n_vocab*N);
        } else {
            // return result for just the last token
            logits_out.resize(n_vocab);
            memcpy(logits_out.data(), (float *) ggml_v3_get_data(res) + (n_vocab*(N-1)), sizeof(float)*n_vocab);
        }
    }

    // extract embeddings
    if (!lctx.embedding.empty()) {
        auto & embedding_out = lctx.embedding;

        embedding_out.resize(n_embd);
        memcpy(embedding_out.data(), (float *) ggml_v3_get_data(embeddings) + (n_embd*(N - 1)), sizeof(float)*n_embd);
    }

    // measure the performance only for the single-token evals
    if (N == 1) {
        lctx.t_eval_us += ggml_v3_time_us() - t_start_us;
        lctx.n_eval++;
    }
    else if (N > 1) {
        lctx.t_p_eval_us += ggml_v3_time_us() - t_start_us;
        lctx.n_p_eval += N;
    }

    return true;
}

//
// tokenizer
//

static size_t utf8_len3(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

struct llama_v3_sp_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

static_assert(std::is_trivially_copyable<llama_v3_sp_symbol>::value, "llama_v3_sp_symbol is not trivially copyable");

struct llama_v3_sp_bigram {
    struct comparator {
        bool operator()(llama_v3_sp_bigram & l, llama_v3_sp_bigram & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llama_v3_sp_bigram>;
    using queue = std::priority_queue<llama_v3_sp_bigram, queue_storage, comparator>;
    llama_v3_sp_symbol::index left;
    llama_v3_sp_symbol::index right;
    float score;
    size_t size;
};

// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
struct llama_v3_tokenizer {
    llama_v3_tokenizer(const llama_v3_vocab & vocab): vocab_(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_v3_vocab::id> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llama_v3_sp_symbol sym;
            size_t char_len = std::min(text.size() - offs, utf8_len3(text[offs]));
            sym.text = text.c_str() + offs;
            sym.n = char_len;
            offs += char_len;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols_.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols_.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue_.empty()) {
            auto bigram = work_queue_.top();
            work_queue_.pop();

            auto & left_sym = symbols_[bigram.left];
            auto & right_sym = symbols_[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //LLAMA_V3_LOG_INFO("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols_[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols_[i].next) {
            auto & symbol = symbols_[i];
            auto token = vocab_.token_to_id.find(std::string(symbol.text, symbol.n));

            if (token == vocab_.token_to_id.end()) {
                // output any symbols that did not form tokens as bytes.
                for (int j = 0; j < (int) symbol.n; ++j) {
                    // NOTE: old version, before #2420 - not sure what are the implications of this
                    //llama_v3_vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
                    llama_v3_vocab::id token_id = vocab_.token_to_id.at(std::string(1, symbol.text[j]));
                    output.push_back(token_id);
                }
            } else {
                output.push_back((*token).second);
            }
        }
    }

private:
    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
        auto token = vocab_.token_to_id.find(text);

        if (token == vocab_.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size()) {
            return;
        }

        const auto &tok_score = vocab_.id_to_token[(*token).second];

        llama_v3_sp_bigram bigram;
        bigram.left = left;
        bigram.right = right;
        bigram.score = tok_score.score;
        bigram.size = text.size();
        work_queue_.push(bigram);
    }

    const llama_v3_vocab & vocab_;
    std::vector<llama_v3_sp_symbol> symbols_;
    llama_v3_sp_bigram::queue work_queue_;
};

std::vector<llama_token> llama_v3_tokenize(
        struct llama_v3_context * ctx,
           const std::string & text,
                        bool   add_bos) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + add_bos;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_v3_tokenize(ctx, text.c_str(), result.data(), result.size(), add_bos);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_v3_tokenize(ctx, text.c_str(), result.data(), result.size(), add_bos);
        GGML_V3_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

static std::vector<llama_v3_vocab::id> llama_v3_tokenize(const llama_v3_vocab & vocab, const std::string & text, bool bos) {
    llama_v3_tokenizer tokenizer(vocab);
    std::vector<llama_v3_vocab::id> output;

    if (text.empty()) {
        return output;
    }

    if (bos) {
        output.push_back(llama_v3_token_bos());
    }

    tokenizer.tokenize(text, output);
    return output;
}

//
// grammar - internal
//

struct llama_v3_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct llama_v3_grammar {
    const std::vector<std::vector<llama_v3_grammar_element>>   rules;
    std::vector<std::vector<const llama_v3_grammar_element *>> stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    llama_v3_partial_utf8                                      partial_utf8;
};

struct llama_v3_grammar_candidate {
    size_t               index;
    const uint32_t     * code_points;
    llama_v3_partial_utf8   partial_utf8;
};

// Decodes a UTF-8 string which may end in an incomplete sequence. Adds a terminating 0 for use as
// pointer. If an invalid sequence is encountered, returns `llama_v3_partial_utf8.n_remain == -1`.
std::pair<std::vector<uint32_t>, llama_v3_partial_utf8> decode_utf8(
        const char         * src,
        llama_v3_partial_utf8   partial_start) {
    static const int      lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4 };
    const char          * pos      = src;
    std::vector<uint32_t> code_points;
    uint32_t              value    = partial_start.value;
    int                   n_remain = partial_start.n_remain;

    // continue previous decode, if applicable
    while (*pos != 0 && n_remain > 0) {
        uint8_t next_byte = static_cast<uint8_t>(*pos);
        if ((next_byte >> 6) != 2) {
            // invalid sequence, abort
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), llama_v3_partial_utf8{ 0, -1 });
        }
        value = (value << 6) + (next_byte & 0x3F);
        ++pos;
        --n_remain;
    }

    if (partial_start.n_remain > 0 && n_remain == 0) {
        code_points.push_back(value);
    }

    // decode any subsequent utf-8 sequences, which may end in an incomplete one
    while (*pos != 0) {
        uint8_t  first_byte = static_cast<uint8_t>(*pos);
        uint8_t  highbits   = first_byte >> 4;
                 n_remain   = lookup[highbits] - 1;

        if (n_remain < 0) {
            // invalid sequence, abort
            code_points.clear();
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), llama_v3_partial_utf8{ 0, n_remain });
        }

        uint8_t  mask       = (1 << (7 - n_remain)) - 1;
                 value      = first_byte & mask;
        ++pos;
        while (*pos != 0 && n_remain > 0) {
            value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
            ++pos;
            --n_remain;
        }
        if (n_remain == 0) {
            code_points.push_back(value);
        }
    }
    code_points.push_back(0);

    return std::make_pair(std::move(code_points), llama_v3_partial_utf8{ value, n_remain });
}

// returns true iff pos points to the end of one of the definitions of a rule
static bool llama_v3_grammar_is_end_of_sequence(const llama_v3_grammar_element * pos) {
    switch (pos->type) {
        case LLAMA_V3_GRETYPE_END: return true;
        case LLAMA_V3_GRETYPE_ALT: return true;
        default:                return false;
    }
}

// returns true iff chr satisfies the char range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static std::pair<bool, const llama_v3_grammar_element *> llama_v3_grammar_match_char(
        const llama_v3_grammar_element * pos,
        const uint32_t                chr) {

    bool found            = false;
    bool is_positive_char = pos->type == LLAMA_V3_GRETYPE_CHAR;
    LLAMA_V3_ASSERT(is_positive_char || pos->type == LLAMA_V3_GRETYPE_CHAR_NOT);

    do {
        if (pos[1].type == LLAMA_V3_GRETYPE_CHAR_RNG_UPPER) {
            // inclusive range, e.g. [a-z]
            found = found || (pos->value <= chr && chr <= pos[1].value);
            pos += 2;
        } else {
            // exact char match, e.g. [a] or "a"
            found = found || pos->value == chr;
            pos += 1;
        }
    } while (pos->type == LLAMA_V3_GRETYPE_CHAR_ALT);

    return std::make_pair(found == is_positive_char, pos);
}

// returns true iff some continuation of the given partial UTF-8 sequence could satisfy the char
// range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static bool llama_v3_grammar_match_partial_char(
        const llama_v3_grammar_element * pos,
        const llama_v3_partial_utf8      partial_utf8) {

    bool is_positive_char = pos->type == LLAMA_V3_GRETYPE_CHAR;
    LLAMA_V3_ASSERT(is_positive_char || pos->type == LLAMA_V3_GRETYPE_CHAR_NOT);

    uint32_t partial_value = partial_utf8.value;
    int      n_remain      = partial_utf8.n_remain;

    // invalid sequence or 7-bit char split across 2 bytes (overlong)
    if (n_remain < 0 || (n_remain == 1 && partial_value < 2)) {
        return false;
    }

    // range of possible code points this partial UTF-8 sequence could complete to
    uint32_t low  = partial_value << (n_remain * 6);
    uint32_t high = low | ((1 << (n_remain * 6)) - 1);

    if (low == 0) {
        if (n_remain == 2) {
            low = 1 << 11;
        } else if (n_remain == 3) {
            low = 1 << 16;
        }
    }

    do {
        if (pos[1].type == LLAMA_V3_GRETYPE_CHAR_RNG_UPPER) {
            // inclusive range, e.g. [a-z]
            if (pos->value <= high && low <= pos[1].value) {
                return is_positive_char;
            }
            pos += 2;
        } else {
            // exact char match, e.g. [a] or "a"
            if (low <= pos->value && pos->value <= high) {
                return is_positive_char;
            }
            pos += 1;
        }
    } while (pos->type == LLAMA_V3_GRETYPE_CHAR_ALT);

    return !is_positive_char;
}


// transforms a grammar pushdown stack into N possible stacks, all ending
// at a character range (terminal element)
static void llama_v3_grammar_advance_stack(
        const std::vector<std::vector<llama_v3_grammar_element>>   & rules,
        const std::vector<const llama_v3_grammar_element *>        & stack,
        std::vector<std::vector<const llama_v3_grammar_element *>> & new_stacks) {

    if (stack.empty()) {
        new_stacks.push_back(stack);
        return;
    }

    const llama_v3_grammar_element * pos = stack.back();

    switch (pos->type) {
        case LLAMA_V3_GRETYPE_RULE_REF: {
            const size_t                  rule_id = static_cast<size_t>(pos->value);
            const llama_v3_grammar_element * subpos  = rules[rule_id].data();
            do {
                // init new stack without the top (pos)
                std::vector<const llama_v3_grammar_element *> new_stack(stack.begin(), stack.end() - 1);
                if (!llama_v3_grammar_is_end_of_sequence(pos + 1)) {
                    // if this rule ref is followed by another element, add that to stack
                    new_stack.push_back(pos + 1);
                }
                if (!llama_v3_grammar_is_end_of_sequence(subpos)) {
                    // if alternate is nonempty, add to stack
                    new_stack.push_back(subpos);
                }
                llama_v3_grammar_advance_stack(rules, new_stack, new_stacks);
                while (!llama_v3_grammar_is_end_of_sequence(subpos)) {
                    // scan to end of alternate def
                    subpos++;
                }
                if (subpos->type == LLAMA_V3_GRETYPE_ALT) {
                    // there's another alternate def of this rule to process
                    subpos++;
                } else {
                    break;
                }
            } while (true);
            break;
        }
        case LLAMA_V3_GRETYPE_CHAR:
        case LLAMA_V3_GRETYPE_CHAR_NOT:
            new_stacks.push_back(stack);
            break;
        default:
            // end of alternate (LLAMA_V3_GRETYPE_END, LLAMA_V3_GRETYPE_ALT) or middle of char range
            // (LLAMA_V3_GRETYPE_CHAR_ALT, LLAMA_V3_GRETYPE_CHAR_RNG_UPPER); stack should never be left on
            // those
            LLAMA_V3_ASSERT(false);
    }
}

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `llama_v3_grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
static std::vector<std::vector<const llama_v3_grammar_element *>> llama_v3_grammar_accept(
        const std::vector<std::vector<llama_v3_grammar_element>>         & rules,
        const std::vector<std::vector<const llama_v3_grammar_element *>> & stacks,
        const uint32_t                                                  chr) {

    std::vector<std::vector<const llama_v3_grammar_element *>> new_stacks;

    for (const auto & stack : stacks) {
        if (stack.empty()) {
            continue;
        }

        auto match = llama_v3_grammar_match_char(stack.back(), chr);
        if (match.first) {
            const llama_v3_grammar_element * pos = match.second;

            // update top of stack to next element, if any
            std::vector<const llama_v3_grammar_element *> new_stack(stack.begin(), stack.end() - 1);
            if (!llama_v3_grammar_is_end_of_sequence(pos)) {
                new_stack.push_back(pos);
            }
            llama_v3_grammar_advance_stack(rules, new_stack, new_stacks);
        }
    }

    return new_stacks;
}

static std::vector<llama_v3_grammar_candidate> llama_v3_grammar_reject_candidates(
        const std::vector<std::vector<llama_v3_grammar_element>>         & rules,
        const std::vector<std::vector<const llama_v3_grammar_element *>> & stacks,
        const std::vector<llama_v3_grammar_candidate>                    & candidates);

static std::vector<llama_v3_grammar_candidate> llama_v3_grammar_reject_candidates_for_stack(
        const std::vector<std::vector<llama_v3_grammar_element>> & rules,
        const std::vector<const llama_v3_grammar_element *>      & stack,
        const std::vector<llama_v3_grammar_candidate>            & candidates) {

    std::vector<llama_v3_grammar_candidate> rejects;

    if (stack.empty()) {
        for (auto tok : candidates) {
            if (*tok.code_points != 0 || tok.partial_utf8.n_remain != 0) {
                rejects.push_back(tok);
            }
        }
        return rejects;
    }

    const llama_v3_grammar_element * stack_pos = stack.back();

    std::vector<llama_v3_grammar_candidate> next_candidates;
    for (auto tok : candidates) {
        if (*tok.code_points == 0) {
            // reached end of full codepoints in token, reject iff it ended in a partial sequence
            // that cannot satisfy this position in grammar
            if (tok.partial_utf8.n_remain != 0 &&
                    !llama_v3_grammar_match_partial_char(stack_pos, tok.partial_utf8)) {
                rejects.push_back(tok);
            }
        } else if (llama_v3_grammar_match_char(stack_pos, *tok.code_points).first) {
            next_candidates.push_back({ tok.index, tok.code_points + 1, tok.partial_utf8 });
        } else {
            rejects.push_back(tok);
        }
    }

    auto stack_pos_after = llama_v3_grammar_match_char(stack_pos, 0).second;

    // update top of stack to next element, if any
    std::vector<const llama_v3_grammar_element *> stack_after(stack.begin(), stack.end() - 1);
    if (!llama_v3_grammar_is_end_of_sequence(stack_pos_after)) {
        stack_after.push_back(stack_pos_after);
    }
    std::vector<std::vector<const llama_v3_grammar_element *>> next_stacks;
    llama_v3_grammar_advance_stack(rules, stack_after, next_stacks);

    auto next_rejects = llama_v3_grammar_reject_candidates(rules, next_stacks, next_candidates);
    for (auto tok : next_rejects) {
        rejects.push_back({ tok.index, tok.code_points - 1, tok.partial_utf8 });
    }

    return rejects;
}

static std::vector<llama_v3_grammar_candidate> llama_v3_grammar_reject_candidates(
        const std::vector<std::vector<llama_v3_grammar_element>>         & rules,
        const std::vector<std::vector<const llama_v3_grammar_element *>> & stacks,
        const std::vector<llama_v3_grammar_candidate>                    & candidates) {
    LLAMA_V3_ASSERT(!stacks.empty()); // REVIEW

    if (candidates.empty()) {
        return std::vector<llama_v3_grammar_candidate>();
    }

    auto rejects = llama_v3_grammar_reject_candidates_for_stack(rules, stacks.front(), candidates);

    for (size_t i = 1, size = stacks.size(); i < size; ++i) {
        rejects = llama_v3_grammar_reject_candidates_for_stack(rules, stacks[i], rejects);
    }
    return rejects;
}

//
// grammar - external
//

struct llama_v3_grammar * llama_v3_grammar_init(
            const llama_v3_grammar_element ** rules,
                                 size_t    n_rules,
                                 size_t    start_rule_index) {
    const llama_v3_grammar_element * pos;

    // copy rule definitions into vectors
    std::vector<std::vector<llama_v3_grammar_element>> vec_rules(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        for (pos = rules[i]; pos->type != LLAMA_V3_GRETYPE_END; pos++) {
            vec_rules[i].push_back(*pos);
        }
        vec_rules[i].push_back({LLAMA_V3_GRETYPE_END, 0});
    }

    // loop over alternates of start rule to build initial stacks
    std::vector<std::vector<const llama_v3_grammar_element *>> stacks;
    pos = rules[start_rule_index];
    do {
        std::vector<const llama_v3_grammar_element *> stack;
        if (!llama_v3_grammar_is_end_of_sequence(pos)) {
            // if alternate is nonempty, add to stack
            stack.push_back(pos);
        }
        llama_v3_grammar_advance_stack(vec_rules, stack, stacks);
        while (!llama_v3_grammar_is_end_of_sequence(pos)) {
            // scan to end of alternate def
            pos++;
        }
        if (pos->type == LLAMA_V3_GRETYPE_ALT) {
            // there's another alternate def of this rule to process
            pos++;
        } else {
            break;
        }
    } while (true);

    return new llama_v3_grammar{ std::move(vec_rules), std::move(stacks), {} };
}

void llama_v3_grammar_free(struct llama_v3_grammar * grammar) {
    delete grammar;
}

//
// sampling
//

void llama_v3_sample_softmax(struct llama_v3_context * ctx, llama_v3_token_data_array * candidates) {
    assert(candidates->size > 0);

    const int64_t t_start_sample_us = ggml_v3_time_us();

    // Sort the logits in descending order
    if (!candidates->sorted) {
        std::sort(candidates->data, candidates->data + candidates->size, [](const llama_v3_token_data & a, const llama_v3_token_data & b) {
            return a.logit > b.logit;
        });
        candidates->sorted = true;
    }

    float max_l = candidates->data[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        float p = expf(candidates->data[i].logit - max_l);
        candidates->data[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].p /= cum_sum;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
    }
}

void llama_v3_sample_top_k(struct llama_v3_context * ctx, llama_v3_token_data_array * candidates, int k, size_t min_keep) {
    const int64_t t_start_sample_us = ggml_v3_time_us();

    k = std::max(k, (int) min_keep);
    k = std::min(k, (int) candidates->size);

    // Sort scores in descending order
    if (!candidates->sorted) {
        auto comp = [](const llama_v3_token_data & a, const llama_v3_token_data & b) {
            return a.logit > b.logit;
        };
        if (k == (int) candidates->size) {
            std::sort(candidates->data, candidates->data + candidates->size, comp);
        } else {
            std::partial_sort(candidates->data, candidates->data + k, candidates->data + candidates->size, comp);
        }
        candidates->sorted = true;
    }
    candidates->size = k;

    if (ctx) {
        ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
    }
}

void llama_v3_sample_top_p(struct llama_v3_context * ctx, llama_v3_token_data_array * candidates, float p, size_t min_keep) {
    if (p >= 1.0f) {
        return;
    }

    llama_v3_sample_softmax(ctx, candidates);

    const int64_t t_start_sample_us = ggml_v3_time_us();

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;

    for (size_t i = 0; i < candidates->size; ++i) {
        cum_sum += candidates->data[i].p;

        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        // we set the last index to i+1 to indicate that the current iterate should be included in the set
        if (cum_sum >= p && i + 1 >= min_keep) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the top-p tokens
    candidates->size = last_idx;

    if (ctx) {
        ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
    }
}

void llama_v3_sample_tail_free(struct llama_v3_context * ctx, llama_v3_token_data_array * candidates, float z, size_t min_keep) {
    if (z >= 1.0f || candidates->size <= 2) {
        return;
    }

    llama_v3_sample_softmax(nullptr, candidates);
    const int64_t t_start_sample_us = ggml_v3_time_us();

    // Compute the first and second derivatives
    std::vector<float> first_derivatives(candidates->size - 1);
    std::vector<float> second_derivatives(candidates->size - 2);

    for (size_t i = 0; i < first_derivatives.size(); ++i) {
        first_derivatives[i] = candidates->data[i].p - candidates->data[i + 1].p;
    }
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = first_derivatives[i] - first_derivatives[i + 1];
    }

    // Calculate absolute value of second derivatives
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = abs(second_derivatives[i]);
    }

    // Normalize the second derivatives
    {
        const float second_derivatives_sum = std::accumulate(second_derivatives.begin(), second_derivatives.end(), 0.0f);

        if (second_derivatives_sum > 1e-6f) {
            for (float & value : second_derivatives) {
                value /= second_derivatives_sum;
            }
        } else {
            for (float & value : second_derivatives) {
                value = 1.0f / second_derivatives.size();
            }
        }
    }

    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        cum_sum += second_derivatives[i];

        // Check if the running sum is greater than z or if we have kept at least min_keep tokens
        if (cum_sum > z && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the tokens above the tail location
    candidates->size = last_idx;

    if (ctx) {
        ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
    }
}


void llama_v3_sample_typical(struct llama_v3_context * ctx, llama_v3_token_data_array * candidates, float p, size_t min_keep) {
    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (p >= 1.0f) {
        return;
    }

    // Compute the softmax of logits and calculate entropy
    llama_v3_sample_softmax(nullptr, candidates);

    const int64_t t_start_sample_us = ggml_v3_time_us();

    float entropy = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        if(candidates->data[i].p>0)
        {
            entropy += -candidates->data[i].p * logf(candidates->data[i].p);
        }
    }

    // Compute the absolute difference between negative log probability and entropy for each candidate
    std::vector<float> shifted_scores;
    for (size_t i = 0; i < candidates->size; ++i) {
        float shifted_score = fabsf(-logf(candidates->data[i].p) - entropy);
        shifted_scores.push_back(shifted_score);
    }

    // Sort tokens based on the shifted_scores and their corresponding indices
    std::vector<size_t> indices(candidates->size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return shifted_scores[a] < shifted_scores[b];
    });

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = indices.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += candidates->data[idx].p;

        // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
        if (cum_sum > p && i >= min_keep - 1) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<llama_v3_token_data> new_candidates;
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        new_candidates.push_back(candidates->data[idx]);
    }

    // Replace the data in candidates with the new_candidates data
    std::copy(new_candidates.begin(), new_candidates.end(), candidates->data);
    candidates->size = new_candidates.size();

    if (ctx) {
        ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
    }
}

void llama_v3_sample_temperature(struct llama_v3_context * ctx, llama_v3_token_data_array * candidates_p, float temp) {
    const int64_t t_start_sample_us = ggml_v3_time_us();

    for (size_t i = 0; i < candidates_p->size; ++i) {
        candidates_p->data[i].logit /= temp;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
    }
}

void llama_v3_sample_repetition_penalty(struct llama_v3_context * ctx, llama_v3_token_data_array * candidates, const llama_v3_token * last_tokens, size_t last_tokens_size, float penalty) {
    if (last_tokens_size == 0 || penalty == 1.0f) {
        return;
    }

    const int64_t t_start_sample_us = ggml_v3_time_us();

    for (size_t i = 0; i < candidates->size; ++i) {
        const auto * token_iter = std::find(last_tokens, last_tokens + last_tokens_size, candidates->data[i].id);
        if (token_iter == last_tokens + last_tokens_size) {
            continue;
        }

        // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
        // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
        if (candidates->data[i].logit <= 0) {
            candidates->data[i].logit *= penalty;
        } else {
            candidates->data[i].logit /= penalty;
        }
    }

    candidates->sorted = false;

    if (ctx) {
        ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
    }
}

void llama_v3_sample_frequency_and_presence_penalties(struct llama_v3_context * ctx, llama_v3_token_data_array * candidates, const llama_v3_token * last_tokens_p, size_t last_tokens_size, float alpha_frequency, float alpha_presence) {
    if (last_tokens_size == 0 || (alpha_frequency == 0.0f && alpha_presence == 0.0f)) {
        return;
    }

    const int64_t t_start_sample_us = ggml_v3_time_us();

    // Create a frequency map to count occurrences of each token in last_tokens
    std::unordered_map<llama_v3_token, int> token_count;
    for (size_t i = 0; i < last_tokens_size; ++i) {
        token_count[last_tokens_p[i]]++;
    }

    // Apply frequency and presence penalties to the candidates
    for (size_t i = 0; i < candidates->size; ++i) {
        auto token_iter = token_count.find(candidates->data[i].id);
        if (token_iter == token_count.end()) {
            continue;
        }

        int count = token_iter->second;
        candidates->data[i].logit -= float(count) * alpha_frequency + float(count > 0) * alpha_presence;
    }

    candidates->sorted = false;

    if (ctx) {
        ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
    }
}

void llama_v3_sample_grammar(struct llama_v3_context * ctx, llama_v3_token_data_array * candidates, const struct llama_v3_grammar * grammar) {
    assert(ctx);
    const int64_t t_start_sample_us = ggml_v3_time_us();

    bool allow_eos = false;
    for (const auto & stack : grammar->stacks) {
        if (stack.empty()) {
            allow_eos = true;
            break;
        }
    }

    const llama_v3_token eos = llama_v3_token_eos();

    std::vector<std::pair<std::vector<uint32_t>, llama_v3_partial_utf8>> candidates_decoded;
    std::vector<llama_v3_grammar_candidate>                              candidates_grammar;

    for (size_t i = 0; i < candidates->size; ++i) {
        const llama_v3_token id  = candidates->data[i].id;
        const char *      str = llama_v3_token_to_str(ctx, id);
        if (id == eos) {
            if (!allow_eos) {
                candidates->data[i].logit = -INFINITY;
            }
        } else if (*str == 0) {
            candidates->data[i].logit = -INFINITY;
        } else {
            candidates_decoded.push_back(decode_utf8(str, grammar->partial_utf8));
            candidates_grammar.push_back({
                i, candidates_decoded.back().first.data(), candidates_decoded.back().second
            });
        }
    }

    const auto rejects =
        llama_v3_grammar_reject_candidates(grammar->rules, grammar->stacks, candidates_grammar);
    for (auto & reject : rejects) {
        candidates->data[reject.index].logit = -INFINITY;
    }

    ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
}

static void llama_v3_log_softmax(float * array, size_t size) {
    float max_l = *std::max_element(array, array + size);
    float sum = 0.f;
    for (size_t i = 0; i < size; ++i) {
        float p = expf(array[i] - max_l);
        sum += p;
        array[i] = p;
    }

    for (size_t i = 0; i < size; ++i) {
        array[i] = logf(array[i] / sum);
    }
}

void llama_v3_sample_classifier_free_guidance(
          struct llama_v3_context * ctx,
        llama_v3_token_data_array * candidates,
          struct llama_v3_context * guidance_ctx,
                         float   scale) {
    int64_t t_start_sample_us = ggml_v3_time_us();

    assert(ctx);
    auto n_vocab = llama_v3_n_vocab(ctx);
    assert(n_vocab == (int)candidates->size);
    assert(!candidates->sorted);

    std::vector<float> logits_base;
    logits_base.reserve(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        logits_base.push_back(candidates->data[i].logit);
    }
    llama_v3_log_softmax(logits_base.data(), candidates->size);

    float* logits_guidance = llama_v3_get_logits(guidance_ctx);
    llama_v3_log_softmax(logits_guidance, n_vocab);

    for (int i = 0; i < n_vocab; ++i) {
        float logit_guidance = logits_guidance[i];
        float logit_base = logits_base[i];
        candidates->data[i].logit = scale * (logit_base - logit_guidance) + logit_guidance;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
    }
}

llama_v3_token llama_v3_sample_token_mirostat(struct llama_v3_context * ctx, llama_v3_token_data_array * candidates, float tau, float eta, int m, float * mu) {
    assert(ctx);
    auto N = float(llama_v3_n_vocab(ctx));
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_v3_time_us();

    llama_v3_sample_softmax(nullptr, candidates);

    // Estimate s_hat using the most probable m tokens
    float s_hat = 0.0;
    float sum_ti_bi = 0.0;
    float sum_ti_sq = 0.0;
    for (size_t i = 0; i < size_t(m - 1) && i < candidates->size - 1; ++i) {
        float t_i = logf(float(i + 2) / float(i + 1));
        float b_i = logf(candidates->data[i].p / candidates->data[i + 1].p);
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
    }
    s_hat = sum_ti_bi / sum_ti_sq;

    // Compute k from the estimated s_hat and target surprise value
    float epsilon_hat = s_hat - 1;
    float k = powf((epsilon_hat * powf(2, *mu)) / (1 - powf(N, -epsilon_hat)), 1 / s_hat);

    // Sample the next word X using top-k sampling
    llama_v3_sample_top_k(nullptr, candidates, int(k), 1);
    if (ctx) {
        ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
    }
    llama_v3_token X = llama_v3_sample_token(ctx, candidates);
    t_start_sample_us = ggml_v3_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_v3_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (ctx) {
        ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
    }
    return X;
}

llama_v3_token llama_v3_sample_token_mirostat_v2(struct llama_v3_context * ctx, llama_v3_token_data_array * candidates, float tau, float eta, float * mu) {
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_v3_time_us();

    llama_v3_sample_softmax(ctx, candidates);

    // Truncate the words with surprise values greater than mu
    candidates->size = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_v3_token_data & candidate) {
        return -log2f(candidate.p) > *mu;
    }));

    if (candidates->size == 0) {
        candidates->size = 1;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
    }

    // Normalize the probabilities of the remaining words
    llama_v3_sample_softmax(ctx, candidates);

    // Sample the next word X from the remaining words
    llama_v3_token X = llama_v3_sample_token(ctx, candidates);
    t_start_sample_us = ggml_v3_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_v3_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (ctx) {
        ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
    }
    return X;
}

llama_v3_token llama_v3_sample_token_greedy(struct llama_v3_context * ctx, llama_v3_token_data_array * candidates) {
    const int64_t t_start_sample_us = ggml_v3_time_us();

    // Find max element
    auto * max_iter = std::max_element(candidates->data, candidates->data + candidates->size, [](const llama_v3_token_data & a, const llama_v3_token_data & b) {
        return a.logit < b.logit;
    });

    llama_v3_token result = max_iter->id;
    if (ctx) {
        ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
        ctx->n_sample++;
    }
    return result;
}

llama_v3_token llama_v3_sample_token(struct llama_v3_context * ctx, llama_v3_token_data_array * candidates) {
    assert(ctx);
    const int64_t t_start_sample_us = ggml_v3_time_us();
    llama_v3_sample_softmax(nullptr, candidates);

    std::vector<float> probs;
    probs.reserve(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        probs.push_back(candidates->data[i].p);
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    auto & rng = ctx->rng;
    int idx = dist(rng);

    llama_v3_token result = candidates->data[idx].id;

    ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
    ctx->n_sample++;
    return result;
}

void llama_v3_grammar_accept_token(struct llama_v3_context * ctx, struct llama_v3_grammar * grammar, llama_v3_token token) {
    const int64_t t_start_sample_us = ggml_v3_time_us();

    if (token == llama_v3_token_eos()) {
        for (const auto & stack : grammar->stacks) {
            if (stack.empty()) {
                return;
            }
        }
        LLAMA_V3_ASSERT(false);
    }

    const char * str = llama_v3_token_to_str(ctx, token);

    // Note terminating 0 in decoded string
    const auto   decoded     = decode_utf8(str, grammar->partial_utf8);
    const auto & code_points = decoded.first;
    for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
        grammar->stacks = llama_v3_grammar_accept(grammar->rules, grammar->stacks, *it);
    }
    grammar->partial_utf8 = decoded.second;
    LLAMA_V3_ASSERT(!grammar->stacks.empty());

    ctx->t_sample_us += ggml_v3_time_us() - t_start_sample_us;
}

//
// quantization
//

static void llama_v3_convert_tensor_internal(const llama_v3_load_tensor & tensor, llama_v3_buffer & output, const int nelements, const int nthread) {
    if (output.size < nelements * sizeof(float)) {
        output.resize(nelements * sizeof(float));
    }
    float * f32_output = (float *) output.addr;

    ggml_v3_type_traits_t qtype;
    if (ggml_v3_is_quantized(tensor.type)) {
        qtype = ggml_v3_internal_get_type_traits(tensor.type);
        if (qtype.to_float == NULL) {
            throw std::runtime_error(format_old("type %s unsupported for integer quantization: no dequantization available", ggml_v3_type_name(tensor.type)));
        }
    } else if (tensor.type != GGML_V3_TYPE_F16) {
        throw std::runtime_error(format_old("cannot dequantize/convert tensor type %s", ggml_v3_type_name(tensor.type)));
    }

    if (nthread < 2) {
        if (tensor.type == GGML_V3_TYPE_F16) {
            ggml_v3_fp16_to_fp32_row((ggml_v3_fp16_t *)tensor.data, f32_output, nelements);
        } else if (ggml_v3_is_quantized(tensor.type)) {
            qtype.to_float(tensor.data, f32_output, nelements);
        } else {
            LLAMA_V3_ASSERT(false); // unreachable
        }
        return;
    }

    auto block_size = tensor.type == GGML_V3_TYPE_F16 ? 1 : (size_t)ggml_v3_blck_size(tensor.type);
    auto block_size_bytes = ggml_v3_type_size(tensor.type);

    LLAMA_V3_ASSERT(nelements % block_size == 0);
    auto nblocks = nelements / block_size;
    auto blocks_per_thread = nblocks / nthread;
    auto spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    std::vector<std::thread> workers;
    for (auto tnum = 0, in_buff_offs = 0, out_buff_offs = 0; tnum < nthread; tnum++) {
        auto thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        auto thr_elems = thr_blocks * block_size; // number of elements for this thread
        auto thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_v3_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_V3_TYPE_F16) {
                ggml_v3_fp16_to_fp32_row((ggml_v3_fp16_t *)inbuf, outbuf, nels);
            } else {
                qtype.to_float(inbuf, outbuf, nels);
            }
        };
        workers.push_back(std::thread(compute, tensor.type, tensor.data + in_buff_offs, f32_output + out_buff_offs, thr_elems));
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & worker : workers) {
        worker.join();
    }

}

static void llama_v3_model_quantize_internal(const std::string & fname_inp, const std::string & fname_out, const llama_v3_model_quantize_params * params) {
    ggml_v3_type quantized_type;
    llama_v3_ftype ftype = params->ftype;
    int nthread = params->nthread;

    switch (params->ftype) {
        case LLAMA_V3_FTYPE_MOSTLY_Q4_0: quantized_type = GGML_V3_TYPE_Q4_0; break;
        case LLAMA_V3_FTYPE_MOSTLY_Q4_1: quantized_type = GGML_V3_TYPE_Q4_1; break;
        case LLAMA_V3_FTYPE_MOSTLY_Q5_0: quantized_type = GGML_V3_TYPE_Q5_0; break;
        case LLAMA_V3_FTYPE_MOSTLY_Q5_1: quantized_type = GGML_V3_TYPE_Q5_1; break;
        case LLAMA_V3_FTYPE_MOSTLY_Q8_0: quantized_type = GGML_V3_TYPE_Q8_0; break;
        case LLAMA_V3_FTYPE_MOSTLY_F16:  quantized_type = GGML_V3_TYPE_F16;  break;
        case LLAMA_V3_FTYPE_ALL_F32:     quantized_type = GGML_V3_TYPE_F32;  break;

#ifdef GGML_USE_K_QUANTS
        // K-quants
        case LLAMA_V3_FTYPE_MOSTLY_Q2_K:   quantized_type = GGML_V3_TYPE_Q2_K; break;
        case LLAMA_V3_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_V3_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_V3_FTYPE_MOSTLY_Q3_K_L: quantized_type = GGML_V3_TYPE_Q3_K; break;
        case LLAMA_V3_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_V3_FTYPE_MOSTLY_Q4_K_M: quantized_type = GGML_V3_TYPE_Q4_K; break;
        case LLAMA_V3_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_V3_FTYPE_MOSTLY_Q5_K_M: quantized_type = GGML_V3_TYPE_Q5_K; break;
        case LLAMA_V3_FTYPE_MOSTLY_Q6_K:   quantized_type = GGML_V3_TYPE_Q6_K; break;
#endif
        default: throw std::runtime_error(format_old("invalid output file type %d\n", ftype));
    }

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    std::unique_ptr<llama_v3_model_loader> model_loader(new llama_v3_model_loader(fname_inp, /*use_mmap*/ false));
    llama_v3_file_saver file_saver(fname_out.c_str(), model_loader->file_loader.get(), params->ftype);

#ifdef GGML_USE_K_QUANTS
    int n_attention_wv    = 0;
    int n_feed_forward_w2 = 0;
    for (auto& tensor : model_loader->tensors_map.tensors) {
        if (tensor.name.find("attention.wv.weight") != std::string::npos) {
            ++n_attention_wv;
        }
        else if (tensor.name.find("feed_forward.w2.weight") != std::string::npos) {
            ++n_feed_forward_w2;
        }
    }

    int i_attention_wv = 0;
    int i_feed_forward_w2 = 0;
#endif

    size_t total_size_org = 0;
    size_t total_size_new = 0;
    std::vector<int64_t> hist_all(1 << 4, 0);

    std::vector<std::thread> workers;
    std::mutex mutex;

    auto use_more_bits = [] (int i_layer, int num_layers) -> bool {
        return i_layer < num_layers/8 || i_layer >= 7*num_layers/8 || (i_layer - num_layers/8)%3 == 2;
    };

    size_t idx = 0;
    for (llama_v3_load_tensor & tensor : model_loader->tensors_map.tensors) {
        llama_v3_buffer read_data;
        read_data.resize(tensor.size);
        tensor.data = read_data.addr;
        model_loader->load_data_for(tensor);

        LLAMA_V3_LOG_INFO("[%4zu/%4zu] %36s - %16s, type = %6s, ",
               ++idx, model_loader->tensors_map.tensors.size(),
               tensor.name.c_str(), llama_v3_format_tensor_shape(tensor.ne).c_str(),
               ggml_v3_type_name(tensor.type));

        // This used to be a regex, but <regex> has an extreme cost to compile times.
        bool quantize = tensor.name.rfind("weight") == tensor.name.size() - 6; // ends with 'weight'?

        // quantize only 2D tensors
        quantize &= (tensor.ne.size() == 2);
        quantize &= params->quantize_output_tensor || tensor.name != "output.weight";
        quantize &= quantized_type != tensor.type;

        enum ggml_v3_type new_type;
        void * new_data;
        size_t new_size;
        llama_v3_buffer work;

        if (!quantize) {
            new_type = tensor.type;
            new_data = tensor.data;
            new_size = tensor.size;
            LLAMA_V3_LOG_INFO("size = %8.3f MB\n", tensor.size/1024.0/1024.0);
        } else {
            new_type = quantized_type;
#ifdef GGML_USE_K_QUANTS
            if (tensor.name == "output.weight") {
                int nx = tensor.ne.at(0);
                int ny = tensor.ne.at(1);
                if (nx % QK_K == 0 && ny % QK_K == 0) {
                    new_type = GGML_V3_TYPE_Q6_K;
                }
            } else if (tensor.name.find("attention.wv.weight") != std::string::npos) {
                if      (ftype == LLAMA_V3_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_V3_FTYPE_MOSTLY_Q2_K) new_type = GGML_V3_TYPE_Q4_K;
                else if (ftype == LLAMA_V3_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_V3_TYPE_Q5_K;
                else if ((ftype == LLAMA_V3_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_V3_FTYPE_MOSTLY_Q5_K_M) &&
                        use_more_bits(i_attention_wv, n_attention_wv)) new_type = GGML_V3_TYPE_Q6_K;
                else if (QK_K == 64 && (ftype == LLAMA_V3_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_V3_FTYPE_MOSTLY_Q3_K_S) &&
                        (i_attention_wv < n_attention_wv/8 || i_attention_wv >= 7*n_attention_wv/8)) new_type = GGML_V3_TYPE_Q6_K;
                ++i_attention_wv;
            } else if (tensor.name.find("feed_forward.w2.weight") != std::string::npos) {
                if      (ftype == LLAMA_V3_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_V3_FTYPE_MOSTLY_Q2_K) new_type = GGML_V3_TYPE_Q4_K;
                else if (ftype == LLAMA_V3_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_V3_TYPE_Q5_K;
                else if ((ftype == LLAMA_V3_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_V3_FTYPE_MOSTLY_Q5_K_M) &&
                         use_more_bits(i_feed_forward_w2, n_feed_forward_w2)) new_type = GGML_V3_TYPE_Q6_K;
                //else if (ftype == LLAMA_V3_FTYPE_MOSTLY_Q4_K_S && i_feed_forward_w2 < n_feed_forward_w2/8) new_type = GGML_V3_TYPE_Q6_K;
                ++i_feed_forward_w2;
            } else if (tensor.name.find("attention.wo.weight") != std::string::npos) {
                if      (ftype == LLAMA_V3_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_V3_FTYPE_MOSTLY_Q2_K) new_type = GGML_V3_TYPE_Q4_K;
                else if (ftype == LLAMA_V3_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_V3_TYPE_Q5_K;
            }
            bool convert_incompatible_tensor = false;
            if (new_type == GGML_V3_TYPE_Q2_K || new_type == GGML_V3_TYPE_Q3_K || new_type == GGML_V3_TYPE_Q4_K ||
                new_type == GGML_V3_TYPE_Q5_K || new_type == GGML_V3_TYPE_Q6_K) {
                int nx = tensor.ne.at(0);
                int ny = tensor.ne.at(1);
                if (nx % QK_K != 0 || ny % QK_K != 0) {
                    LLAMA_V3_LOG_INFO("\n\nTensor sizes %d x %d are not divisible by %d, required for k-quants.\n",nx,ny,QK_K);
                    convert_incompatible_tensor = true;
                }
            }
            if (convert_incompatible_tensor) {
                if (tensor.name == "output.weight") {
                    new_type = GGML_V3_TYPE_F16; //fall back to F16 instead of just failing.
                    LLAMA_V3_LOG_WARN("F16 will be used for this tensor instead.\n");
                } else if (tensor.name == "tok_embeddings.weight") {
                    new_type = GGML_V3_TYPE_Q4_0; //fall back to Q4_0 instead of just failing.
                    LLAMA_V3_LOG_WARN("Q4_0 will be used for this tensor instead.\n");
                } else {
                    throw std::runtime_error("Unsupported tensor size encountered\n");
                }
            }
#endif

            float * f32_data;
            size_t nelements = tensor.ne.at(0) * tensor.ne.at(1);
            llama_v3_buffer f32_conv_buf;

            if (tensor.type == GGML_V3_TYPE_F32) {
                f32_data = (float *) tensor.data;
            } else if (ggml_v3_is_quantized(tensor.type) && !params->allow_requantize) {
                throw std::runtime_error(format_old("requantizing from type %s is disabled", ggml_v3_type_name(tensor.type)));
            } else {
                llama_v3_convert_tensor_internal(tensor, f32_conv_buf, nelements, nthread);
                f32_data = (float *) f32_conv_buf.addr;
            }

            LLAMA_V3_LOG_INFO("quantizing to %s .. ", ggml_v3_type_name(new_type));
            fflush(stdout);

            work.resize(nelements * 4); // upper bound on size
            new_data = work.addr;
            std::vector<int64_t> hist_cur(1 << 4, 0);

            int chunk_size = 32 * 512;
            const int nchunk = (nelements + chunk_size - 1)/chunk_size;
            const int nthread_use = nthread > 1 ? std::max(1, std::min(nthread, nchunk)) : 1;
            if (nthread_use < 2) {
                new_size = ggml_v3_quantize_chunk(new_type, f32_data, new_data, 0, nelements, hist_cur.data());
            } else {
                size_t counter = 0;
                new_size = 0;
                auto compute = [&mutex, &counter, &hist_cur, &new_size, new_type, f32_data, new_data, nelements, chunk_size] () {
                    std::vector<int64_t> local_hist;
                    size_t local_size = 0;
                    while (true) {
                        std::unique_lock<std::mutex> lock(mutex);
                        size_t first = counter; counter += chunk_size;
                        if (first >= nelements) {
                            if (!local_hist.empty()) {
                                for (int j=0; j<int(local_hist.size()); ++j) {
                                    hist_cur[j] += local_hist[j];
                                }
                                new_size += local_size;
                            }
                            break;
                        }
                        lock.unlock();
                        size_t last = std::min(nelements, first + chunk_size);
                        if (local_hist.empty()) {
                            local_hist.resize(hist_cur.size(), 0);
                        }
                        local_size += ggml_v3_quantize_chunk(new_type, f32_data, new_data, first, last - first, local_hist.data());
                    }
                };
                if ((int) workers.size() < nthread_use - 1) {
                    workers.resize(nthread_use - 1);
                }
                for (int it = 0; it < nthread_use - 1; ++it) {
                    workers[it] = std::thread(compute);
                }
                compute();
                for (int it = 0; it < nthread_use - 1; ++it) {
                    workers[it].join();
                }
            }

            LLAMA_V3_LOG_INFO("size = %8.2f MB -> %8.2f MB | hist: ", tensor.size/1024.0/1024.0, new_size/1024.0/1024.0);
            int64_t tot_count = 0;
            for (size_t i = 0; i < hist_cur.size(); i++) {
                hist_all[i] += hist_cur[i];
                tot_count += hist_cur[i];
            }

            if (tot_count > 0) {
                for (size_t i = 0; i < hist_cur.size(); i++) {
                    LLAMA_V3_LOG_INFO("%5.3f ", hist_cur[i] / float(nelements));
                }
            }
            LLAMA_V3_LOG_INFO("\n");
        }
        total_size_org += tensor.size;
        total_size_new += new_size;
        file_saver.write_tensor(tensor, new_type, new_data, new_size);
    }

    LLAMA_V3_LOG_INFO("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    LLAMA_V3_LOG_INFO("%s: quant size  = %8.2f MB\n", __func__, total_size_new/1024.0/1024.0);

    {
        int64_t sum_all = 0;
        for (size_t i = 0; i < hist_all.size(); i++) {
            sum_all += hist_all[i];
        }

        if (sum_all > 0) {
            LLAMA_V3_LOG_INFO("%s: hist: ", __func__);
            for (size_t i = 0; i < hist_all.size(); i++) {
                LLAMA_V3_LOG_INFO("%5.3f ", hist_all[i] / float(sum_all));
            }
            LLAMA_V3_LOG_INFO("\n");
        }
    }
}



//
// interface implementation
//

struct llama_v3_model * llama_v3_load_model_from_file(
                             const char * path_model,
            struct llama_v3_context_params   params) {
    ggml_v3_time_init();

    llama_v3_model * model = new llama_v3_model;

    ggml_v3_type memory_type = params.f16_kv ? GGML_V3_TYPE_F16 : GGML_V3_TYPE_F32;

    if (!llama_v3_model_load(path_model, *model, model->vocab, params.n_ctx, params.n_batch, params.n_gqa, params.rms_norm_eps, params.n_gpu_layers,
                params.main_gpu, params.tensor_split, params.mul_mat_q, params.rope_freq_base, params.rope_freq_scale,params.low_vram,
                memory_type, params.use_mmap, params.use_mlock, params.vocab_only, params.progress_callback,
                params.progress_callback_user_data)) {
        LLAMA_V3_LOG_ERROR("%s: failed to load model\n", __func__);
        delete model;
        return nullptr;
    }

    return model;
}

void llama_v3_free_model(struct llama_v3_model * model) {
    delete model;
}

struct llama_v3_context * llama_v3_new_context_with_model(
                 struct llama_v3_model * model,
        struct llama_v3_context_params   params) {

    if (!model) {
        return nullptr;
    }

    llama_v3_context * ctx = new llama_v3_context(*model);

    if (params.seed == LLAMA_V3_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    size_t blasbatchmul = get_blas_batch_mul3(params.n_batch);

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_V3_LOG_INFO(".");
                if (percentage >= 100) {
                    LLAMA_V3_LOG_INFO("\n");
                }
            }
        };
    }

    ctx->rng = std::mt19937(params.seed);
    ctx->logits_all = params.logits_all;

    ggml_v3_type memory_type = params.f16_kv ? GGML_V3_TYPE_F16 : GGML_V3_TYPE_F32;

    // reserve memory for context buffers
    if (!params.vocab_only) {
        if (!kv_cache_init(ctx->model.hparams, ctx->kv_self, memory_type, ctx->model.hparams.n_ctx, params.n_gpu_layers)) {
            LLAMA_V3_LOG_ERROR("%s: kv_cache_init() failed for self-attention cache\n", __func__);
            llama_v3_free(ctx);
            return nullptr;
        }

        {
            const size_t memory_size = ggml_v3_nbytes(ctx->kv_self.k) + ggml_v3_nbytes(ctx->kv_self.v);
            LLAMA_V3_LOG_INFO("%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
        }

        const auto & hparams = ctx->model.hparams;

        // resized during inference
        if (params.logits_all) {
            ctx->logits.reserve(hparams.n_ctx*hparams.n_vocab);
        } else {
            ctx->logits.reserve(hparams.n_vocab);
        }

        if (params.embedding){
            ctx->embedding.resize(hparams.n_embd);
        }

#ifdef LLAMA_V3_USE_ALLOCATOR
        {
            static const size_t tensor_alignment = 32;
            // the compute buffer is used to store the tensor and graph structs, while the allocator buffer is used for the tensor data
            ctx->buf_compute.resize(ggml_v3_tensor_overhead()*GGML_V3_MAX_NODES + ggml_v3_graph_overhead());

            // create measure allocator
            ctx->alloc = ggml_v3_allocr_new_measure(tensor_alignment);

            // build worst-case graph
            int n_tokens = std::min((int)hparams.n_ctx, params.n_batch);
            int n_past = hparams.n_ctx - n_tokens;
            llama_v3_token token = llama_v3_token_bos(); // not actually used by llama_v3_build_graph, but required to choose between token and embedding inputs graph
            ggml_v3_cgraph * gf = llama_v3_build_graph(*ctx, &token, NULL, n_tokens, n_past);

            // measure memory requirements for the graph
            size_t alloc_size = ggml_v3_allocr_alloc_graph(ctx->alloc, gf) + tensor_alignment;

            LLAMA_V3_LOG_INFO("%s: compute buffer total size = %7.2f MB\n", __func__, (ctx->buf_compute.size + alloc_size) / 1024.0 / 1024.0);

            // debug - for comparison with scratch buffer
            //size_t prev_req =
            //    MEM_REQ_SCRATCH0_3(hparams.n_ctx).at(ctx->model.type) +
            //    MEM_REQ_SCRATCH1_3().at(ctx->model.type) +
            //    MEM_REQ_EVAL_3().at(ctx->model.type);
            //LLAMA_V3_LOG_INFO("%s: (debug) equivalent with scratch buffer = %7.2f MB\n", __func__, prev_req / 1024.0 / 1024.0);

            // recreate allocator with exact memory requirements
            ggml_v3_allocr_free(ctx->alloc);

            ctx->buf_alloc.resize(alloc_size);
            ctx->alloc = ggml_v3_allocr_new(ctx->buf_alloc.addr, ctx->buf_alloc.size, tensor_alignment);

        }
#else
        ctx->buf_compute.resize(blasbatchmul*MEM_REQ_EVAL_3().at(ctx->model.type) + ggml_v3_graph_overhead());
#endif

#ifdef LLAMA_V3_USE_SCRATCH
        ctx->buf_scratch[0].resize(blasbatchmul*MEM_REQ_SCRATCH0_3(hparams.n_ctx).at(ctx->model.type));
        ctx->buf_scratch[1].resize(blasbatchmul*MEM_REQ_SCRATCH1_3().at(ctx->model.type));
#endif
    }

    return ctx;
}

struct llama_v3_context * llama_v3_init_from_file(
                             const char * path_model,
            struct llama_v3_context_params   params) {

    struct llama_v3_model * model = llama_v3_load_model_from_file(path_model, params);
    if (!model) {
        return nullptr;
    }
    struct llama_v3_context * ctx = llama_v3_new_context_with_model(model, params);
    ctx->model_owner = true;
    return ctx;
}

void llama_v3_free(struct llama_v3_context * ctx) {
    delete ctx;
}

int llama_v3_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_v3_model_quantize_params *params) {
    try {
        llama_v3_model_quantize_internal(fname_inp, fname_out, params);
        return 0;
    } catch (const std::exception & err) {
        LLAMA_V3_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }
}

int llama_v3_apply_lora_from_file_internal(const struct llama_v3_model & model, const char * path_lora, const char * path_base_model, int n_threads) {
    LLAMA_V3_LOG_INFO("%s: applying lora adapter from '%s' - please wait ...\n", __func__, path_lora);

    const int64_t t_start_lora_us = ggml_v3_time_us();

    auto fin = std::ifstream(path_lora, std::ios::binary);
    if (!fin) {
        LLAMA_V3_LOG_ERROR("%s: failed to open '%s'\n", __func__, path_lora);
        return 1;
    }

    // verify magic and version
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != LLAMA_V3_FILE_MAGIC_GGLA) {
            LLAMA_V3_LOG_ERROR("%s: bad file magic\n", __func__);
            return 1;
        }
        uint32_t format_version;
        fin.read((char *) &format_version, sizeof(format_version));

        if (format_version != 1) {
            LLAMA_V3_LOG_ERROR("%s: unsupported file version\n", __func__ );
            return 1;
        }
    }

    int32_t lora_r;
    int32_t lora_alpha;
    fin.read((char *) &lora_r, sizeof(lora_r));
    fin.read((char *) &lora_alpha, sizeof(lora_alpha));
    float scaling = (float)lora_alpha / (float)lora_r;

    LLAMA_V3_LOG_INFO("%s: r = %d, alpha = %d, scaling = %.2f\n", __func__, lora_r, lora_alpha, scaling);


    // create a temporary ggml context to store the lora tensors
    // todo: calculate size from biggest possible tensor
    std::vector<uint8_t> lora_buf(1024ull * 1024ull * 1024ull);
    struct ggml_v3_init_params params;
    params.mem_size   = lora_buf.size();
    params.mem_buffer = lora_buf.data();
    params.no_alloc   = false;

    ggml_v3_context * lora_ctx = ggml_v3_init(params);
    std::unordered_map<std::string, struct ggml_v3_tensor *> lora_tensors;

    // create a name -> tensor map of the model to accelerate lookups
    std::unordered_map<std::string, struct ggml_v3_tensor*> model_tensors;
    for (const auto & kv: model.tensors_by_name) {
        model_tensors.insert(kv);
    }


    // load base model
    std::unique_ptr<llama_v3_model_loader> model_loader;
    ggml_v3_context * base_ctx = NULL;
    llama_v3_buffer base_buf;
    if (path_base_model) {
        LLAMA_V3_LOG_INFO("%s: loading base model from '%s'\n", __func__, path_base_model);
        model_loader.reset(new llama_v3_model_loader(path_base_model, /*use_mmap*/ true));

        size_t ctx_size;
        size_t mmapped_size;
        model_loader->calc_sizes(&ctx_size, &mmapped_size);
        base_buf.resize(ctx_size);

        ggml_v3_init_params base_params;
        base_params.mem_size   = base_buf.size;
        base_params.mem_buffer = base_buf.addr;
        base_params.no_alloc   = model_loader->use_mmap;

        base_ctx = ggml_v3_init(base_params);

        model_loader->ggml_v3_ctx = base_ctx;

        // maybe this should in llama_v3_model_loader
        if (model_loader->use_mmap) {
            model_loader->mapping.reset(new llama_v3_mmap(&model_loader->file_loader->file, /* prefetch */ 0, ggml_v3_is_numa()));
        }
    }

    // read tensors and apply
    bool warned = false;
    int n_tensors = 0;

    std::vector<uint8_t> work_buffer;

    while (true) {
        int32_t n_dims;
        int32_t length;
        int32_t ftype;

        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        fin.read(reinterpret_cast<char *>(&length), sizeof(length));
        fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));
        if (fin.eof()) {
            break;
        }

        int32_t ne[2] = { 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        }

        std::string name;
        {
            char buf[1024];
            fin.read(buf, length);
            name = std::string(buf, length);
        }

        // check for lora suffix and get the type of tensor
        const std::string lora_suffix = ".lora";
        size_t pos = name.rfind(lora_suffix);
        if (pos == std::string::npos) {
            LLAMA_V3_LOG_ERROR("%s: error: '%s' is not a lora tensor\n", __func__, name.c_str());
            return 1;
        }

        std::string lora_type = name.substr(pos + lora_suffix.length());
        std::string base_name = name;
        base_name.erase(pos);
        // LLAMA_V3_LOG_INFO("%s: %s => %s (lora type %s) \n", __func__, name.c_str(),base_name.c_str(), lora_type.c_str());

        if (model_tensors.find(base_name) == model_tensors.end()) {
            LLAMA_V3_LOG_ERROR("%s: unknown tensor '%s' in lora adapter\n", __func__, name.data());
            return 1;
        }

        // create ggml tensor
        ggml_v3_type wtype;
        switch (ftype) {
            case 0: wtype = GGML_V3_TYPE_F32;  break;
            case 1: wtype = GGML_V3_TYPE_F16;  break;
            default:
                    {
                        LLAMA_V3_LOG_ERROR("%s: invalid tensor data type '%d'\n",
                                __func__, ftype);
                        return false;
                    }
        }
        ggml_v3_tensor * lora_tensor;
        if (n_dims == 2) {
            lora_tensor = ggml_v3_new_tensor_2d(lora_ctx, wtype, ne[0], ne[1]);
        }
        else {
            LLAMA_V3_LOG_ERROR("%s: unsupported tensor dimension %d\n", __func__, n_dims);
            return 1;
        }
        ggml_v3_set_name(lora_tensor, "lora_tensor");

        // load tensor data
        size_t offset = fin.tellg();
        size_t tensor_data_size = ggml_v3_nbytes(lora_tensor);
        offset = (offset + 31) & -32;
        fin.seekg(offset);
        fin.read((char*)lora_tensor->data, tensor_data_size);

        lora_tensors[name] = lora_tensor;

        // check if we have both A and B tensors and apply
        if (lora_tensors.find(base_name + ".loraA") != lora_tensors.end() &&
            lora_tensors.find(base_name + ".loraB") != lora_tensors.end()) {

            ggml_v3_tensor * dest_t = model_tensors[base_name];

            offload_func_v3_t offload_func = llama_v3_nop;
            offload_func_v3_t offload_func_force_inplace = llama_v3_nop;

#if defined(GGML_USE_CUDA) || defined(GGML_USE_CLBLAST)
            if (dest_t->backend == GGML_V3_BACKEND_GPU || dest_t->backend == GGML_V3_BACKEND_GPU_SPLIT) {
                if (dest_t->type != GGML_V3_TYPE_F16) {
                    printf("\nError: the simultaneous use of LoRAs and GPU acceleration is only supported for f16 models\n");
                    throw std::runtime_error(format_old(
                        "%s: error: lora failed", __func__));
                }
#if defined(GGML_USE_CUDA)
                offload_func = ggml_v3_cuda_assign_buffers;
                offload_func_force_inplace = ggml_v3_cuda_assign_buffers_force_inplace;
#endif
            }
#endif // GGML_USE_CUDA

            ggml_v3_tensor * base_t;
            if (model_loader) {
                // load from base model
                if (model_loader->tensors_map.name_to_idx.find(base_name) == model_loader->tensors_map.name_to_idx.end()) {
                    LLAMA_V3_LOG_ERROR("%s: error: tensor '%s' not found in base model\n", __func__, base_name.c_str());
                    return 1;
                }
                size_t idx = model_loader->tensors_map.name_to_idx[base_name];
                llama_v3_load_tensor & lt = model_loader->tensors_map.tensors[idx];
                base_t = model_loader->get_tensor(base_name, { (uint32_t)dest_t->ne[0], (uint32_t)dest_t->ne[1] }, GGML_V3_BACKEND_CPU);
                lt.data = (uint8_t *) lt.ggml_v3_tensor->data;
                model_loader->load_data_for(lt);
                lt.ggml_v3_tensor->data = lt.data;
            }
            else {
                base_t = dest_t;
            }

            if (ggml_v3_is_quantized(base_t->type)) {
                if (!warned) {
                    LLAMA_V3_LOG_WARN("%s: warning: using a lora adapter with a quantized model may result in poor quality, "
                                   "use a f16 or f32 base model with --lora-base\n", __func__);
                    warned = true;
                }
            }

            ggml_v3_tensor * loraA = lora_tensors[base_name + ".loraA"];
            GGML_V3_ASSERT(loraA->type == GGML_V3_TYPE_F32);
            ggml_v3_set_name(loraA, "loraA");

            ggml_v3_tensor * loraB = lora_tensors[base_name + ".loraB"];
            GGML_V3_ASSERT(loraB->type == GGML_V3_TYPE_F32);
            ggml_v3_set_name(loraB, "loraB");

            if (base_t->ne[0] != loraA->ne[1] || base_t->ne[1] != loraB->ne[1]) {
                LLAMA_V3_LOG_ERROR("%s: incompatible tensor dimensions (%" PRId64 " and %" PRId64 ");"
                                " are you sure that this adapter is for this model?\n", __func__, base_t->ne[0], loraA->ne[1]);
                return 1;
            }

            // w = w + BA*s
            ggml_v3_tensor * BA = ggml_v3_mul_mat(lora_ctx, loraA, loraB);
            offload_func(BA);
            ggml_v3_set_name(BA, "BA");

            if (scaling != 1.0f) {
                ggml_v3_tensor * scale_tensor = ggml_v3_new_f32(lora_ctx, scaling);
                ggml_v3_set_name(scale_tensor, "scale_tensor");

                BA = ggml_v3_scale_inplace(lora_ctx, BA, scaling);
                offload_func(BA);
                ggml_v3_set_name(BA, "BA_scaled");
            }

            ggml_v3_tensor * r;
            if (base_t == dest_t) {
                r = ggml_v3_add_inplace(lora_ctx, dest_t, BA);
                offload_func_force_inplace(r);
                ggml_v3_set_name(r, "r_add_inplace");
            }
            else {
                r = ggml_v3_add(lora_ctx, base_t, BA);
                offload_func(r);
                ggml_v3_set_name(r, "r_add");

                r = ggml_v3_cpy(lora_ctx, r, dest_t);
                offload_func(r);
                ggml_v3_set_name(r, "r_cpy");
            }

            struct ggml_v3_cgraph * gf = ggml_v3_new_graph(lora_ctx);
            ggml_v3_build_forward_expand(gf, r);

            llv3_graph_compute_helper(work_buffer, gf, n_threads);

            // we won't need these tensors again, reset the context to save memory
            ggml_v3_free(lora_ctx);
            lora_ctx = ggml_v3_init(params);
            lora_tensors.clear();

            n_tensors++;
            if (n_tensors % 4 == 0) {
                LLAMA_V3_LOG_INFO(".");
            }
        }
    }

    // TODO: this should be in a destructor, it will leak on failure
    ggml_v3_free(lora_ctx);
    if (base_ctx) {
        ggml_v3_free(base_ctx);
    }

    const int64_t t_lora_us = ggml_v3_time_us() - t_start_lora_us;
    LLAMA_V3_LOG_INFO(" done (%.2f ms)\n", t_lora_us / 1000.0);

    return 0;
}

int llama_v3_apply_lora_from_file(struct llama_v3_context * ctx, const char * path_lora, const char * path_base_model, int n_threads) {
    try {
        return llama_v3_apply_lora_from_file_internal(ctx->model, path_lora, path_base_model, n_threads);
    } catch (const std::exception & err) {
        LLAMA_V3_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());
        return 1;
    }
}

int llama_v3_model_apply_lora_from_file(const struct llama_v3_model * model, const char * path_lora, const char * path_base_model, int n_threads) {
    try {
        return llama_v3_apply_lora_from_file_internal(*model, path_lora, path_base_model, n_threads);
    } catch (const std::exception & err) {
        LLAMA_V3_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());
        return 1;
    }
}

int llama_v3_get_kv_cache_token_count(const struct llama_v3_context * ctx) {
    return ctx->kv_self.n;
}

#define LLAMA_V3_MAX_RNG_STATE (64*1024)

void llama_v3_set_rng_seed(struct llama_v3_context * ctx, uint32_t seed) {
    if (seed == LLAMA_V3_DEFAULT_SEED) {
        seed = time(NULL);
    }
    ctx->rng.seed(seed);
}

// Returns the *maximum* size of the state
size_t llama_v3_get_state_size(const struct llama_v3_context * ctx) {
    // we don't know size of rng until we actually serialize it. so reserve more than enough memory for its serialized state.
    // for reference, std::mt19937(1337) serializes to 6701 bytes.
    const size_t s_rng_size        = sizeof(size_t);
    const size_t s_rng             = LLAMA_V3_MAX_RNG_STATE;
    const size_t s_logits_capacity = sizeof(size_t);
    const size_t s_logits_size     = sizeof(size_t);
    const size_t s_logits          = ctx->logits.capacity() * sizeof(float);
    const size_t s_embedding_size  = sizeof(size_t);
    const size_t s_embedding       = ctx->embedding.size() * sizeof(float);
    const size_t s_kv_size         = sizeof(size_t);
    const size_t s_kv_ntok         = sizeof(int);
    const size_t s_kv              = ctx->kv_self.buf.size;

    const size_t s_total = (
        + s_rng_size
        + s_rng
        + s_logits_capacity
        + s_logits_size
        + s_logits
        + s_embedding_size
        + s_embedding
        + s_kv_size
        + s_kv_ntok
        + s_kv
    );

    return s_total;
}

/** copy state data into either a buffer or file depending on the passed in context
 *
 * file context:
 * llama_v3_file file("/path", "wb");
 * llama_v3_data_file_context data_ctx(&file);
 * llama_v3_copy_state_data(ctx, &data_ctx);
 *
 * buffer context:
 * std::vector<uint8_t> buf(max_size, 0);
 * llama_v3_data_buffer_context data_ctx(&buf.data());
 * llama_v3_copy_state_data(ctx, &data_ctx);
 *
*/
void llama_v3_copy_state_data_internal(struct llama_v3_context * ctx, llama_v3_data_context * data_ctx) {
    // copy rng
    {
        std::stringstream rng_ss;
        rng_ss << ctx->rng;

        const size_t rng_size = rng_ss.str().size();
        char rng_buf[LLAMA_V3_MAX_RNG_STATE];

        memset(&rng_buf[0], 0, LLAMA_V3_MAX_RNG_STATE);
        memcpy(&rng_buf[0], rng_ss.str().data(), rng_ss.str().size());

        data_ctx->write(&rng_size,   sizeof(rng_size));
        data_ctx->write(&rng_buf[0], LLAMA_V3_MAX_RNG_STATE);
    }

    // copy logits
    {
        const size_t logits_cap  = ctx->logits.capacity();
        const size_t logits_size = ctx->logits.size();

        data_ctx->write(&logits_cap,  sizeof(logits_cap));
        data_ctx->write(&logits_size, sizeof(logits_size));

        if (logits_size) {
            data_ctx->write(ctx->logits.data(), logits_size * sizeof(float));
        }

        // If there is a gap between the size and the capacity, write padding
        size_t padding_size = (logits_cap - logits_size) * sizeof(float);
        if (padding_size > 0) {
            std::vector<uint8_t> padding(padding_size, 0); // Create a buffer filled with zeros
            data_ctx->write(padding.data(), padding_size);
        }
    }

    // copy embeddings
    {
        const size_t embedding_size = ctx->embedding.size();

        data_ctx->write(&embedding_size, sizeof(embedding_size));

        if (embedding_size) {
            data_ctx->write(ctx->embedding.data(), embedding_size * sizeof(float));
        }
    }

    // copy kv cache
    {
        const auto & kv_self = ctx->kv_self;
        const auto & hparams = ctx->model.hparams;
        const int    n_layer = hparams.n_layer;
        const int    n_embd  = hparams.n_embd_gqa();
        const int    n_ctx   = hparams.n_ctx;

        const size_t kv_size = kv_self.buf.size;
        const int    kv_ntok = llama_v3_get_kv_cache_token_count(ctx);

        data_ctx->write(&kv_size, sizeof(kv_size));
        data_ctx->write(&kv_ntok, sizeof(kv_ntok));

        if (kv_size) {
            const size_t elt_size = ggml_v3_element_size(kv_self.k);

            ggml_v3_context * cpy_ctx = ggml_v3_init({ 4096, NULL, /* no_alloc */ true });
            ggml_v3_cgraph * gf = ggml_v3_new_graph(cpy_ctx);

            ggml_v3_tensor * kout3d = ggml_v3_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd, kv_ntok, n_layer);
            std::vector<uint8_t> kout3d_data(ggml_v3_nbytes(kout3d), 0);
            kout3d->data = kout3d_data.data();

            ggml_v3_tensor * vout3d = ggml_v3_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok, n_embd, n_layer);
            std::vector<uint8_t> vout3d_data(ggml_v3_nbytes(vout3d), 0);
            vout3d->data = vout3d_data.data();

            ggml_v3_tensor * k3d = ggml_v3_view_3d(cpy_ctx, kv_self.k,
                n_embd, kv_ntok, n_layer,
                elt_size*n_embd, elt_size*n_embd*n_ctx, 0);

            ggml_v3_tensor * v3d = ggml_v3_view_3d(cpy_ctx, kv_self.v,
                kv_ntok, n_embd, n_layer,
                elt_size*n_ctx, elt_size*n_ctx*n_embd, 0);

            ggml_v3_build_forward_expand(gf, ggml_v3_cpy(cpy_ctx, k3d, kout3d));
            ggml_v3_build_forward_expand(gf, ggml_v3_cpy(cpy_ctx, v3d, vout3d));
            llv3_graph_compute_helper(ctx->work_buffer, gf, /*n_threads*/ 1);

            ggml_v3_free(cpy_ctx);

            // our data is now in the kout3d_data and vout3d_data buffers
            // write them to file
            data_ctx->write(kout3d_data.data(), kout3d_data.size());
            data_ctx->write(vout3d_data.data(), vout3d_data.size());
        }
    }
}

size_t llama_v3_copy_state_data(struct llama_v3_context * ctx, uint8_t * dst) {
    llama_v3_data_buffer_context data_ctx(dst);
    llama_v3_copy_state_data_internal(ctx, &data_ctx);

    return data_ctx.get_size_written();
}

// Sets the state reading from the specified source address
size_t llama_v3_set_state_data(struct llama_v3_context * ctx, uint8_t * src) {
    uint8_t * inp = src;

    // set rng
    {
        size_t rng_size;
        char   rng_buf[LLAMA_V3_MAX_RNG_STATE];

        memcpy(&rng_size,   inp, sizeof(rng_size));    inp += sizeof(rng_size);
        memcpy(&rng_buf[0], inp, LLAMA_V3_MAX_RNG_STATE); inp += LLAMA_V3_MAX_RNG_STATE;

        std::stringstream rng_ss;
        rng_ss.str(std::string(&rng_buf[0], rng_size));
        rng_ss >> ctx->rng;

        LLAMA_V3_ASSERT(rng_ss.fail() == false);
    }

    // set logits
    {
        size_t logits_cap;
        size_t logits_size;

        memcpy(&logits_cap,  inp, sizeof(logits_cap));  inp += sizeof(logits_cap);
        memcpy(&logits_size, inp, sizeof(logits_size)); inp += sizeof(logits_size);

        LLAMA_V3_ASSERT(ctx->logits.capacity() == logits_cap);

        if (logits_size) {
            ctx->logits.resize(logits_size);
            memcpy(ctx->logits.data(), inp, logits_size * sizeof(float));
        }

        inp += logits_cap * sizeof(float);
    }

    // set embeddings
    {
        size_t embedding_size;

        memcpy(&embedding_size, inp, sizeof(embedding_size)); inp += sizeof(embedding_size);

        LLAMA_V3_ASSERT(ctx->embedding.capacity() == embedding_size);

        if (embedding_size) {
            memcpy(ctx->embedding.data(), inp, embedding_size * sizeof(float));
            inp += embedding_size * sizeof(float);
        }
    }

    // set kv cache
    {
        const auto & kv_self = ctx->kv_self;
        const auto & hparams = ctx->model.hparams;
        const int    n_layer = hparams.n_layer;
        const int    n_embd  = hparams.n_embd_gqa();
        const int    n_ctx   = hparams.n_ctx;

        size_t kv_size;
        int kv_ntok;

        memcpy(&kv_size, inp, sizeof(kv_size)); inp += sizeof(kv_size);
        memcpy(&kv_ntok, inp, sizeof(kv_ntok)); inp += sizeof(kv_ntok);

        if (kv_size) {
            LLAMA_V3_ASSERT(kv_self.buf.size == kv_size);

            const size_t elt_size = ggml_v3_element_size(kv_self.k);

            ggml_v3_context * cpy_ctx = ggml_v3_init({ 4096, NULL, /* no_alloc */ true });
            ggml_v3_cgraph * gf = ggml_v3_new_graph(cpy_ctx);

            ggml_v3_tensor * kin3d = ggml_v3_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd, kv_ntok, n_layer);
            kin3d->data = (void *) inp;
            inp += ggml_v3_nbytes(kin3d);

            ggml_v3_tensor * vin3d = ggml_v3_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok, n_embd, n_layer);
            vin3d->data = (void *) inp;
            inp += ggml_v3_nbytes(vin3d);

            ggml_v3_tensor * k3d = ggml_v3_view_3d(cpy_ctx, kv_self.k,
                n_embd, kv_ntok, n_layer,
                elt_size*n_embd, elt_size*n_embd*n_ctx, 0);

            ggml_v3_tensor * v3d = ggml_v3_view_3d(cpy_ctx, kv_self.v,
                kv_ntok, n_embd, n_layer,
                elt_size*n_ctx, elt_size*n_ctx*n_embd, 0);

            ggml_v3_build_forward_expand(gf, ggml_v3_cpy(cpy_ctx, kin3d, k3d));
            ggml_v3_build_forward_expand(gf, ggml_v3_cpy(cpy_ctx, vin3d, v3d));
            llv3_graph_compute_helper(ctx->work_buffer, gf, /*n_threads*/ 1);

            ggml_v3_free(cpy_ctx);
        }

        ctx->kv_self.n = kv_ntok;
    }

    const size_t nread    = inp - src;
    const size_t max_size = llama_v3_get_state_size(ctx);

    LLAMA_V3_ASSERT(nread <= max_size);

    return nread;
}

static bool llama_v3_load_session_file_internal(struct llama_v3_context * ctx, const char * path_session, llama_v3_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_v3_file file(path_session, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_V3_SESSION_MAGIC || version != LLAMA_V3_SESSION_VERSION) {
            LLAMA_V3_LOG_ERROR("%s : unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }

        llama_v3_hparams session_hparams;
        file.read_raw(&session_hparams, sizeof(llama_v3_hparams));

        if (session_hparams != ctx->model.hparams) {
            LLAMA_V3_LOG_INFO("%s : model hparams didn't match from session file!\n", __func__);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_V3_LOG_ERROR("%s : token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_v3_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size - file.tell();
        const size_t n_state_size_max = llama_v3_get_state_size(ctx);

        if (n_state_size_cur > n_state_size_max) {
            LLAMA_V3_LOG_ERROR("%s : the state size in session file is too big! max %zu, got %zu\n", __func__, n_state_size_max, n_state_size_cur);
            return false;
        }

        std::vector<uint8_t> state_data(n_state_size_max);
        file.read_raw(state_data.data(), n_state_size_cur);

        llama_v3_set_state_data(ctx, state_data.data());
    }

    return true;
}

bool llama_v3_load_session_file(struct llama_v3_context * ctx, const char * path_session, llama_v3_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    try {
        return llama_v3_load_session_file_internal(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_V3_LOG_ERROR("error loading session file: %s\n", err.what());
        return false;
    }
}

bool llama_v3_save_session_file(struct llama_v3_context * ctx, const char * path_session, const llama_v3_token * tokens, size_t n_token_count) {
    llama_v3_file file(path_session, "wb");

    file.write_u32(LLAMA_V3_SESSION_MAGIC);
    file.write_u32(LLAMA_V3_SESSION_VERSION);

    file.write_raw(&ctx->model.hparams, sizeof(llama_v3_hparams));

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_v3_token) * n_token_count);

    // save the context state using stream saving
    llama_v3_data_file_context data_ctx(&file);
    llama_v3_copy_state_data_internal(ctx, &data_ctx);

    return true;
}

int llama_v3_eval(
        struct llama_v3_context * ctx,
           const llama_v3_token * tokens,
                         int   n_tokens,
                         int   n_past,
                         int   n_threads) {
    if (!llama_v3_eval_internal(*ctx, tokens, nullptr, n_tokens, n_past, n_threads, nullptr)) {
        LLAMA_V3_LOG_ERROR("%s: failed to eval\n", __func__);
        return 1;
    }

    // get a more accurate load time, upon first eval
    // TODO: fix this
    if (!ctx->has_evaluated_once) {
        ctx->t_load_us = ggml_v3_time_us() - ctx->t_start_us;
        ctx->has_evaluated_once = true;
    }

    return 0;
}


int llama_v3_eval_embd(
            struct llama_v3_context * ctx,
                     const float * embd,
                             int   n_tokens,
                             int   n_past,
                             int   n_threads) {
    if (!llama_v3_eval_internal(*ctx, nullptr, embd, n_tokens, n_past, n_threads, nullptr)) {
        LLAMA_V3_LOG_ERROR("%s: failed to eval\n", __func__);
        return 1;
    }

    // get a more accurate load time, upon first eval
    // TODO: fix this
    if (!ctx->has_evaluated_once) {
        ctx->t_load_us = ggml_v3_time_us() - ctx->t_start_us;
        ctx->has_evaluated_once = true;
    }

    return 0;
}

int llama_v3_eval_export(struct llama_v3_context * ctx, const char * fname) {
    const int n_batch = 1;
    const int n_ctx   = 512 - n_batch;

    const std::vector<llama_v3_token> tmp(n_batch, llama_v3_token_bos());

    if (!llama_v3_eval_internal(*ctx, tmp.data(), nullptr, tmp.size(), n_ctx, 1, fname)) {
        LLAMA_V3_LOG_ERROR("%s: failed to eval\n", __func__);
        return 1;
    }

    return 0;
}

int llama_v3_tokenize_with_model(
    const struct llama_v3_model * model,
                  const char * text,
                 llama_v3_token * tokens,
                         int   n_max_tokens,
                        bool   add_bos) {
    auto res = llama_v3_tokenize(model->vocab, text, add_bos);

    if (n_max_tokens < (int) res.size()) {
        LLAMA_V3_LOG_ERROR("%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

int llama_v3_tokenize(
        struct llama_v3_context * ctx,
                  const char * text,
                 llama_v3_token * tokens,
                         int   n_max_tokens,
                        bool   add_bos) {
    return llama_v3_tokenize_with_model(&ctx->model, text, tokens, n_max_tokens, add_bos);
}

int llama_v3_n_vocab_from_model(const struct llama_v3_model * model) {
    return model->vocab.id_to_token.size();
}

int llama_v3_n_ctx_from_model(const struct llama_v3_model * model) {
    return model->hparams.n_ctx;
}

int llama_v3_n_embd_from_model(const struct llama_v3_model * model) {
    return model->hparams.n_embd;
}

int llama_v3_n_vocab(const struct llama_v3_context * ctx) {
    return ctx->model.vocab.id_to_token.size();
}

int llama_v3_n_ctx(const struct llama_v3_context * ctx) {
    return ctx->model.hparams.n_ctx;
}

int llama_v3_n_embd(const struct llama_v3_context * ctx) {
    return ctx->model.hparams.n_embd;
}

int llama_v3_model_type(const struct llama_v3_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "LLaMA %s %s", llama_v3_model_type_name(model->type), llama_v3_ftype_name(model->hparams.ftype));
}

int llama_v3_get_vocab_from_model(
        const struct llama_v3_model * model,
        const char * * strings,
        float  * scores,
        int capacity) {
    int n = std::min(capacity, (int) model->vocab.id_to_token.size());
    for (int i = 0; i<n; ++i) {
        strings[i] = model->vocab.id_to_token[i].tok.c_str();
        scores[i]  = model->vocab.id_to_token[i].score;
    }
    return n;
}

int llama_v3_get_vocab(
        const struct llama_v3_context * ctx,
        const char * * strings,
        float  * scores,
        int capacity) {
    return llama_v3_get_vocab_from_model(&ctx->model, strings, scores, capacity);
}

float * llama_v3_get_logits(struct llama_v3_context * ctx) {
    return ctx->logits.data();
}

float * llama_v3_get_embeddings(struct llama_v3_context * ctx) {
    return ctx->embedding.data();
}

const char * llama_v3_token_to_str_with_model(const struct llama_v3_model * model, llama_v3_token token) {
    if (token >= llama_v3_n_vocab_from_model(model)) {
        return nullptr;
    }

    return model->vocab.id_to_token[token].tok.c_str();
}

const char * llama_v3_token_to_str(const struct llama_v3_context * ctx, llama_v3_token token) {
    return llama_v3_token_to_str_with_model(&ctx->model, token);
}

llama_v3_token llama_v3_token_bos() {
    return 1;
}

llama_v3_token llama_v3_token_eos() {
    return 2;
}

llama_v3_token llama_v3_token_nl() {
    return 13;
}

struct llama_v3_timings llama_v3_get_timings(struct llama_v3_context * ctx) {
    struct llama_v3_timings result = {
        /*.t_start_ms  =*/ 1e-3 * ctx->t_start_us,
        /*.t_end_ms    =*/ 1.00 * ggml_v3_time_ms(),
        /*.t_load_ms   =*/ 1e-3 * ctx->t_load_us,
        /*.t_sample_ms =*/ 1e-3 * ctx->t_sample_us,
        /*.t_p_eval_ms =*/ 1e-3 * ctx->t_p_eval_us,
        /*.t_eval_ms   =*/ 1e-3 * ctx->t_eval_us,

        /*.n_sample =*/ std::max(1, ctx->n_sample),
        /*.n_p_eval =*/ std::max(1, ctx->n_p_eval),
        /*.n_eval   =*/ std::max(1, ctx->n_eval),
    };

    return result;
}

void llama_v3_print_timings(struct llama_v3_context * ctx) {
    const llama_v3_timings timings = llama_v3_get_timings(ctx);

    LLAMA_V3_LOG_INFO("\n");
    LLAMA_V3_LOG_INFO("%s:        load time = %8.2f ms\n", __func__, timings.t_load_ms);
    LLAMA_V3_LOG_INFO("%s:      sample time = %8.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_sample_ms, timings.n_sample, timings.t_sample_ms / timings.n_sample, 1e3 / timings.t_sample_ms * timings.n_sample);
    LLAMA_V3_LOG_INFO("%s: prompt eval time = %8.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_p_eval_ms, timings.n_p_eval, timings.t_p_eval_ms / timings.n_p_eval, 1e3 / timings.t_p_eval_ms * timings.n_p_eval);
    LLAMA_V3_LOG_INFO("%s:        eval time = %8.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_eval_ms, timings.n_eval, timings.t_eval_ms / timings.n_eval, 1e3 / timings.t_eval_ms * timings.n_eval);
    LLAMA_V3_LOG_INFO("%s:       total time = %8.2f ms\n", __func__, (timings.t_end_ms - timings.t_start_ms));
}

void llama_v3_reset_timings(struct llama_v3_context * ctx) {
    ctx->t_start_us = ggml_v3_time_us();
    ctx->t_sample_us = ctx->n_sample = 0;
    ctx->t_eval_us   = ctx->n_eval   = 0;
    ctx->t_p_eval_us = ctx->n_p_eval = 0;
}

const char * llama_v3_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "         + std::to_string(ggml_v3_cpu_has_avx())         + " | ";
    s += "AVX2 = "        + std::to_string(ggml_v3_cpu_has_avx2())        + " | ";
    s += "AVX512 = "      + std::to_string(ggml_v3_cpu_has_avx512())      + " | ";
    s += "AVX512_VBMI = " + std::to_string(ggml_v3_cpu_has_avx512_vbmi()) + " | ";
    s += "AVX512_VNNI = " + std::to_string(ggml_v3_cpu_has_avx512_vnni()) + " | ";
    s += "FMA = "         + std::to_string(ggml_v3_cpu_has_fma())         + " | ";
    s += "NEON = "        + std::to_string(ggml_v3_cpu_has_neon())        + " | ";
    s += "ARM_FMA = "     + std::to_string(ggml_v3_cpu_has_arm_fma())     + " | ";
    s += "F16C = "        + std::to_string(ggml_v3_cpu_has_f16c())        + " | ";
    s += "FP16_VA = "     + std::to_string(ggml_v3_cpu_has_fp16_va())     + " | ";
    s += "WASM_SIMD = "   + std::to_string(ggml_v3_cpu_has_wasm_simd())   + " | ";
    s += "BLAS = "        + std::to_string(ggml_v3_cpu_has_blas())        + " | ";
    s += "SSE3 = "        + std::to_string(ggml_v3_cpu_has_sse3())        + " | ";
    s += "VSX = "         + std::to_string(ggml_v3_cpu_has_vsx())         + " | ";

    return s.c_str();
}

// For internal test use
const std::vector<std::pair<std::string, struct ggml_v3_tensor *>>& llama_v3_internal_get_tensor_map(struct llama_v3_context * ctx) {
    return ctx->model.tensors_by_name;
}


void llama_v3_log_set(llama_v3_log_callback log_callback, void * user_data) {
    llv3_g_state.log_callback = log_callback ? log_callback : llama_v3_log_callback_default;
    llv3_g_state.log_callback_user_data = user_data;
}

#if defined(_MSC_VER) && !defined(vsnprintf)
#define vsnprintf _vsnprintf
#endif

static void llama_v3_log_internal_v(llama_v3_log_level level, const char * format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        llv3_g_state.log_callback(level, buffer, llv3_g_state.log_callback_user_data);
    } else {
        char* buffer2 = new char[len+1];
        vsnprintf(buffer2, len+1, format, args_copy);
        buffer2[len] = 0;
        llv3_g_state.log_callback(level, buffer2, llv3_g_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args_copy);
}

static void llama_v3_log_internal(llama_v3_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    llama_v3_log_internal_v(level, format, args);
    va_end(args);
}

static void llama_v3_log_callback_default(llama_v3_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

//// stuff this here since it's just some obsolete junk ////
static std::vector<uint8_t> kcpp_compute_buf;
void kcpp_graph_compute_helper(struct ggml_v3_cgraph *graph, int n_threads)
{
    struct ggml_v3_cplan plan = ggml_v3_graph_plan(graph, n_threads);
    if (plan.work_size > 0)
    {
        kcpp_compute_buf.resize(plan.work_size);
        plan.work_data = kcpp_compute_buf.data();
    }
    ggml_v3_graph_compute(graph, &plan);
}