#pragma once
#include <cstdint>

const int tensor_split_max = 16;
const int images_max = 8;
const int logprobs_max = 5;

// match kobold's sampler list and order
enum samplers
{
    KCPP_SAMPLER_TOP_K=0,
    KCPP_SAMPLER_TOP_A=1,
    KCPP_SAMPLER_TOP_P=2,
    KCPP_SAMPLER_TFS=3,
    KCPP_SAMPLER_TYP=4,
    KCPP_SAMPLER_TEMP=5,
    KCPP_SAMPLER_REP_PEN=6,
    KCPP_SAMPLER_MAX
};
enum stop_reason
{
    INVALID=-1,
    OUT_OF_TOKENS=0,
    EOS_TOKEN_HIT=1,
    CUSTOM_STOPPER=2,
};
struct logit_bias {
    int32_t token_id;
    float bias;
};
struct load_model_inputs
{
    const int threads = 0;
    const int blasthreads = 0;
    const int max_context_length = 0;
    const bool low_vram = 0;
    const bool use_mmq = 0;
    const bool use_rowsplit = 0;
    const char * executable_path = nullptr;
    const char * model_filename = nullptr;
    const char * lora_filename = nullptr;
    const char * lora_base = nullptr;
    const char * draftmodel_filename = nullptr;
    const int draft_amount = 8;
    const int draft_gpulayers = 999;
    const float draft_gpusplit[tensor_split_max] = {};
    const char * mmproj_filename = nullptr;
    const int visionmaxres = 2048;
    const bool use_mmap = false;
    const bool use_mlock = false;
    const bool use_smartcontext = false;
    const bool use_contextshift = false;
    const bool use_fastforward = false;
    const int clblast_info = 0;
    const int cublas_info = 0;
    const char * vulkan_info = nullptr;
    const int blasbatchsize = 512;
    const int forceversion = 0;
    const int gpulayers = 0;
    const float rope_freq_scale = 1.0f;
    const float rope_freq_base = 10000.0f;
    const int moe_experts = -1;
    const bool flash_attention = false;
    const float tensor_split[tensor_split_max] = {};
    const int quant_k = 0;
    const int quant_v = 0;
    const bool quiet = false;
    const int debugmode = 0;
};
struct generation_inputs
{
    const int seed = 0;
    const char * prompt = nullptr;
    const char * memory = nullptr;
    const char * images[images_max] = {};
    const int max_context_length = 0;
    const int max_length = 0;
    const float temperature = 0.0f;
    const int top_k = 0;
    const float top_a = 0.0f;
    const float top_p = 0.0f;
    const float min_p = 0.0f;
    const float typical_p = 0;
    const float tfs = 0;
    const float nsigma = 0.0f;
    const float rep_pen = 0;
    const int rep_pen_range = 0;
    const float rep_pen_slope = 1.0f;
    const float presence_penalty = 0.0f;
    const int mirostat = 0;
    const float mirostat_eta = 0.0f;
    const float mirostat_tau = 0.0f;
    const float xtc_threshold = 0.0f;
    const float xtc_probability = 0.0f;
    const int xtc_round = 0;
    const float xtc_divide = 1.0f;
    const samplers sampler_order[KCPP_SAMPLER_MAX] = {};
    const int sampler_len = 0;
    const bool allow_eos_token = false;
    const bool bypass_eos_token = false;
    const bool render_special = false;
    const bool stream_sse = false;
    const char * grammar = nullptr;
    const bool grammar_retain_state = false;
    const float dynatemp_range = 0.0f;
    const float dynatemp_exponent = 1.0f;
    const float smoothing_factor = 0.0f;
    const float dry_multiplier = 0.0f;
    const float dry_base = 0.0f;
    const int dry_allowed_length = 0;
    const int dry_penalty_last_n = 0;
    const int dry_sequence_breakers_len = 0;
    const char ** dry_sequence_breakers = nullptr;
    const int stop_sequence_len = 0;
    const char ** stop_sequence = nullptr;
    const int logit_biases_len = 0;
    const logit_bias * logit_biases = nullptr;
    const int banned_tokens_len = 0;
    const char ** banned_tokens = nullptr;
};
struct generation_outputs
{
    int status = -1;
    int stopreason = stop_reason::INVALID;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    const char * text; //response will now be stored in c++ allocated memory
};
struct token_count_outputs
{
    int count = 0;
    int * ids; //we'll just use shared memory for this one, bit of a hack
};

struct logprob_item {
    int option_count;
    const char * selected_token;
    float selected_logprob;
    const char * tokens[logprobs_max];
    float * logprobs = nullptr;
};
struct last_logprobs_outputs {
    int count = 0;
    logprob_item * logprob_items = nullptr;
};

struct sd_load_model_inputs
{
    const char * model_filename = nullptr;
    const char * executable_path = nullptr;
    const int clblast_info = 0;
    const int cublas_info = 0;
    const char * vulkan_info = nullptr;
    const int threads = 0;
    const int quant = 0;
    const bool taesd = false;
    const bool notile = false;
    const char * t5xxl_filename = nullptr;
    const char * clipl_filename = nullptr;
    const char * clipg_filename = nullptr;
    const char * vae_filename = nullptr;
    const char * lora_filename = nullptr;
    const float lora_multiplier = 1.0f;
    const bool quiet = false;
    const int debugmode = 0;
};
struct sd_generation_inputs
{
    const char * prompt = nullptr;
    const char * negative_prompt = nullptr;
    const char * init_images = "";
    const float denoising_strength = 0.0f;
    const float cfg_scale = 0.0f;
    const int sample_steps = 0;
    const int width = 0;
    const int height = 0;
    const int seed = 0;
    const char * sample_method = nullptr;
    const int clip_skip = -1;
};
struct sd_generation_outputs
{
    int status = -1;
    const char * data = "";
};

struct whisper_load_model_inputs
{
    const char * model_filename = nullptr;
    const char * executable_path = nullptr;
    const int clblast_info = 0;
    const int cublas_info = 0;
    const char * vulkan_info = nullptr;
    const bool quiet = false;
    const int debugmode = 0;
};
struct whisper_generation_inputs
{
    const char * prompt = nullptr;
    const char * audio_data = nullptr;
    const bool suppress_non_speech = false;
    const char * langcode = nullptr;
};
struct whisper_generation_outputs
{
    int status = -1;
    const char * text = "";
};

struct tts_load_model_inputs
{
    const int threads = 4;
    const char * ttc_model_filename = nullptr;
    const char * cts_model_filename = nullptr;
    const char * executable_path = nullptr;
    const int clblast_info = 0;
    const int cublas_info = 0;
    const char * vulkan_info = nullptr;
    const int gpulayers = 0;
    const bool flash_attention = false;
    const int ttsmaxlen = 4096;
    const bool quiet = false;
    const int debugmode = 0;
};
struct tts_generation_inputs
{
    const char * prompt = nullptr;
    const int speaker_seed = 0;
    const int audio_seed = 0;
};
struct tts_generation_outputs
{
    int status = -1;
    const char * data = "";
};

extern std::string executable_path;
extern std::string lora_filename;
extern std::string lora_base;
extern std::string mmproj_filename;
extern std::string draftmodel_filename;
extern std::vector<std::string> generated_tokens;
extern bool generation_finished;
extern float last_eval_time;
extern float last_process_time;
extern int last_token_count;
extern int last_seed;
extern int total_gens;
extern int total_img_gens;
extern stop_reason last_stop_reason;
