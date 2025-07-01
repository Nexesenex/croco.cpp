#ifndef __STABLE_DIFFUSION_H__
#define __STABLE_DIFFUSION_H__

#if defined(_WIN32) || defined(__CYGWIN__)
#ifndef SD_BUILD_SHARED_LIB
#define SD_API
#else
#ifdef SD_BUILD_DLL
#define SD_API __declspec(dllexport)
#else
#define SD_API __declspec(dllimport)
#endif
#endif
#else
#if __GNUC__ >= 4
#define SD_API __attribute__((visibility("default")))
#else
#define SD_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum rng_type_t {
    STD_DEFAULT_RNG,
    CUDA_RNG
};

enum sample_method_t {
    EULER_A,
    EULER,
    HEUN,
    DPM2,
    DPMPP2S_A,
    DPMPP2M,
    DPMPP2Mv2,
    IPNDM,
    IPNDM_V,
    LCM,
    DDIM_TRAILING,
    TCD,
    N_SAMPLE_METHODS
};

enum schedule_t {
    DEFAULT,
    DISCRETE,
    KARRAS,
    EXPONENTIAL,
    AYS,
    GITS,
    N_SCHEDULES
};

// same as enum ggml_type
enum sd_type_t {
    SD_TYPE_F32     = 0,
    SD_TYPE_F16     = 1,
    SD_TYPE_Q4_0    = 2,
    SD_TYPE_Q4_1    = 3,
    // SD_TYPE_Q4_2 = 4, support has been removed
    // SD_TYPE_Q4_3 = 5, support has been removed
    SD_TYPE_Q5_0    = 6,
    SD_TYPE_Q5_1    = 7,
    SD_TYPE_Q8_0    = 8,
    SD_TYPE_Q8_1    = 9,
    SD_TYPE_Q2_K    = 10,
    SD_TYPE_Q3_K    = 11,
    SD_TYPE_Q4_K    = 12,
    SD_TYPE_Q5_K    = 13,
    SD_TYPE_Q6_K    = 14,
    SD_TYPE_Q8_K    = 15,
    SD_TYPE_IQ2_XXS = 16,
    SD_TYPE_IQ2_XS  = 17,
    SD_TYPE_IQ3_XXS = 18,
    SD_TYPE_IQ1_S   = 19,
    SD_TYPE_IQ4_NL  = 20,
    SD_TYPE_IQ3_S   = 21,
    SD_TYPE_IQ2_S   = 22,
    SD_TYPE_IQ4_XS  = 23,
    SD_TYPE_I8      = 24,
    SD_TYPE_I16     = 25,
    SD_TYPE_I32     = 26,
    SD_TYPE_I64     = 27,
    SD_TYPE_F64     = 28,
    SD_TYPE_IQ1_M   = 29,
    SD_TYPE_BF16    = 30,
    SD_TYPE_Q4_0_4_4 = 31,
    SD_TYPE_Q4_0_4_8 = 32,
    SD_TYPE_Q4_0_8_8 = 33,
    SD_TYPE_TQ1_0   = 34,
    SD_TYPE_TQ2_0   = 35,
    SD_TYPE_IQ4_NL_4_4 = 36,
    // SD_TYPE_IQ4_NL_4_8 = 37,
    // SD_TYPE_IQ4_NL_8_8 = 38,
    // SD_TYPE_COUNT = 39,
    //
    SD_TYPE_I2_S    = 86,
    //
    SD_TYPE_Q8_0_X4 = 97,
    SD_TYPE_Q8_1_X4 = 98,
    SD_TYPE_Q8_2_X4 = 99,
    SD_TYPE_Q6_0    = 133,
    SD_TYPE_IQ1_BN  = 134,
    SD_TYPE_IQ2_BN  = 135,
    SD_TYPE_Q8_K64  = 136,
    SD_TYPE_IQ2_K   = 137,
    SD_TYPE_IQ3_K   = 138,
    SD_TYPE_IQ4_K   = 139,
    SD_TYPE_IQ5_K   = 140,
    SD_TYPE_IQ6_K   = 141,
    // depricated: SD_TYPE_IQ2_TN  = 142,
    // depricated: SD_TYPE_IQ1_TN  = 143,
    SD_TYPE_IQ4_KS  = 144,
    SD_TYPE_IQ2_KS  = 145,
    SD_TYPE_IQ4_KSS = 146,
    SD_TYPE_Q8_K16   = 147,
    SD_TYPE_Q8_K32   = 148,
    SD_TYPE_Q8_KR8   = 149,
    SD_TYPE_Q8_K128  = 150,
    SD_TYPE_Q8_KV    = 151,
    SD_TYPE_IQ5_KS   = 152,
    SD_TYPE_IQ2_KT   = 153,
    SD_TYPE_IQ3_KT   = 154,
    SD_TYPE_IQ4_KT   = 155,

    SD_TYPE_IQ3_KS_V1   = 196,

    SD_TYPE_Q4_0_R8   = 202,
    SD_TYPE_Q5_0_R4   = 206,
    SD_TYPE_Q8_0_R8   = 208,
    SD_TYPE_Q2_K_R4   = 210,
    SD_TYPE_Q3_K_R4   = 211,
    SD_TYPE_Q4_K_R4   = 212,
    SD_TYPE_Q5_K_R4   = 213,
    SD_TYPE_Q6_K_R4   = 214,
    SD_TYPE_IQ2_XXS_R4= 216,
    SD_TYPE_IQ2_XS_R4 = 217,
    SD_TYPE_IQ3_XXS_R4= 218,
    SD_TYPE_IQ1_S_R4  = 219,
    SD_TYPE_IQ4_NL_R4 = 220,
    SD_TYPE_IQ3_S_R4  = 221,
    SD_TYPE_IQ2_S_R4  = 222,
    SD_TYPE_IQ4_XS_R8 = 223,
    SD_TYPE_IQ1_M_R4  = 229,
    SD_TYPE_BF16_R16  = 230,
    SD_TYPE_Q6_0_R4   = 233,
    SD_TYPE_IQ2_BN_R4 = 335,
    SD_TYPE_IQ2_K_R4  = 337,
    SD_TYPE_IQ3_K_R4  = 338,
    SD_TYPE_IQ4_K_R4  = 339,
    SD_TYPE_IQ5_K_R4  = 340,
    SD_TYPE_IQ4_KS_R4 = 344,
    SD_TYPE_IQ5_KS_R4 = 352,
    SD_TYPE_Q8_KV_R8  = 398,
    SD_TYPE_Q8_K_R8   = 399,

    SD_TYPE_COUNT,
};

SD_API const char* sd_type_name(enum sd_type_t type);

enum sd_log_level_t {
    SD_LOG_DEBUG,
    SD_LOG_INFO,
    SD_LOG_WARN,
    SD_LOG_ERROR
};

typedef void (*sd_log_cb_t)(enum sd_log_level_t level, const char* text, void* data);
typedef void (*sd_progress_cb_t)(int step, int steps, float time, void* data);

SD_API void sd_set_log_callback(sd_log_cb_t sd_log_cb, void* data);
SD_API void sd_set_progress_callback(sd_progress_cb_t cb, void* data);
SD_API int32_t sd_get_num_physical_cores();
SD_API const char* sd_get_system_info();

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
} sd_image_t;

typedef struct sd_ctx_t sd_ctx_t;

SD_API void set_sd_vae_tiling(sd_ctx_t* ctx, bool tiling);
SD_API int get_loaded_sd_version(sd_ctx_t* ctx);
SD_API bool sd_loaded_chroma();

SD_API sd_ctx_t* new_sd_ctx(const char* model_path,
                            const char* clip_l_path,
                            const char* clip_g_path,
                            const char* t5xxl_path,
                            const char* diffusion_model_path,
                            const char* vae_path,
                            const char* taesd_path,
                            const char* control_net_path_c_str,
                            const char* lora_model_dir,
                            const char* embed_dir_c_str,
                            const char* stacked_id_embed_dir_c_str,
                            bool vae_decode_only,
                            bool vae_tiling,
                            bool free_params_immediately,
                            int n_threads,
                            enum sd_type_t wtype,
                            enum rng_type_t rng_type,
                            enum schedule_t s,
                            bool keep_clip_on_cpu,
                            bool keep_control_net_cpu,
                            bool keep_vae_on_cpu,
                            bool diffusion_flash_attn,
                            bool chroma_use_dit_mask,
                            bool chroma_use_t5_mask,
                            int chroma_t5_mask_pad);

SD_API void free_sd_ctx(sd_ctx_t* sd_ctx);

SD_API sd_image_t* txt2img(sd_ctx_t* sd_ctx,
                           const char* prompt,
                           const char* negative_prompt,
                           int clip_skip,
                           float cfg_scale,
                           float guidance,
                           float eta,
                           int width,
                           int height,
                           enum sample_method_t sample_method,
                           int sample_steps,
                           int64_t seed,
                           int batch_count,
                           const sd_image_t* control_cond,
                           float control_strength,
                           float style_strength,
                           bool normalize_input,
                           const char* input_id_images_path,
                           sd_image_t* kontext_imgs,
                           int kontext_img_count,
                           int* skip_layers,
                           size_t skip_layers_count,
                           float slg_scale,
                           float skip_layer_start,
                           float skip_layer_end,
                           const std::vector<sd_image_t*> photomaker_references);

SD_API sd_image_t* img2img(sd_ctx_t* sd_ctx,
                           sd_image_t init_image,
                           sd_image_t mask_image,
                           const char* prompt,
                           const char* negative_prompt,
                           int clip_skip,
                           float cfg_scale,
                           float guidance,
                           float eta,
                           int width,
                           int height,
                           enum sample_method_t sample_method,
                           int sample_steps,
                           float strength,
                           int64_t seed,
                           int batch_count,
                           const sd_image_t* control_cond,
                           float control_strength,
                           float style_strength,
                           bool normalize_input,
                           const char* input_id_images_path,
                           sd_image_t* kontext_imgs,
                           int kontext_img_count,
                           int* skip_layers,
                           size_t skip_layers_count,
                           float slg_scale,
                           float skip_layer_start,
                           float skip_layer_end,
                           const std::vector<sd_image_t*> photomaker_references);

SD_API sd_image_t* img2vid(sd_ctx_t* sd_ctx,
                           sd_image_t init_image,
                           int width,
                           int height,
                           int video_frames,
                           int motion_bucket_id,
                           int fps,
                           float augmentation_level,
                           float min_cfg,
                           float cfg_scale,
                           enum sample_method_t sample_method,
                           int sample_steps,
                           float strength,
                           int64_t seed);

SD_API sd_image_t* edit(sd_ctx_t* sd_ctx,
                        sd_image_t* ref_images,
                        int ref_images_count,
                        const char* prompt,
                        const char* negative_prompt,
                        int clip_skip,
                        float cfg_scale,
                        float guidance,
                        float eta,
                        int width,
                        int height,
                        enum sample_method_t sample_method,
                        int sample_steps,
                        float strength,
                        int64_t seed,
                        int batch_count,
                        const sd_image_t* control_cond,
                        float control_strength,
                        float style_strength,
                        bool normalize_input,
                        int* skip_layers,
                        size_t skip_layers_count,
                        float slg_scale,
                        float skip_layer_start,
                        float skip_layer_end);

typedef struct upscaler_ctx_t upscaler_ctx_t;

SD_API upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path,
                                        int n_threads);
SD_API void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx);

SD_API sd_image_t upscale(upscaler_ctx_t* upscaler_ctx, sd_image_t input_image, uint32_t upscale_factor);

SD_API bool convert(const char* input_path, const char* vae_path, const char* output_path, enum sd_type_t output_type);

SD_API uint8_t* preprocess_canny(uint8_t* img,
                                 int width,
                                 int height,
                                 float high_threshold,
                                 float low_threshold,
                                 float weak,
                                 float strong,
                                 bool inverse);

#ifdef __cplusplus
}
#endif

#endif  // __STABLE_DIFFUSION_H__
