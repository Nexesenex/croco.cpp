#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// #include "iqk/iqk_quantize.h"

#include <stdint.h>
#include <stddef.h>

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_1_ref(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_0_ref(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_1_ref(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_1_ref(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q6_0_ref(const float * GGML_RESTRICT x, block_q6_0 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_q2_K_ref(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q3_K_ref(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_K_ref(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_K_ref(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_K64_ref(const float * GGML_RESTRICT x, block_q8_K64 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_tq1_0_ref(const float * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tq2_0_ref(const float * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_iq1_s_ref(const float * GGML_RESTRICT x, block_iq2_xxs * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq1_m_ref(const float * GGML_RESTRICT x, block_iq2_xs * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq2_xxs_ref(const float * GGML_RESTRICT x, block_iq2_xxs * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq2_xs_ref (const float * GGML_RESTRICT x, block_iq2_xs  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq3_xxs_ref(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_nl_ref (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_xs_ref (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq3_s_ref  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq2_s_ref  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);


// GGML_API void quantize_row_iq2_ks_ref(const float * GGML_RESTRICT x, block_iq2_ks * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq2_k_ref (const float * GGML_RESTRICT x, block_iq2_k  * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq3_k_ref (const float * GGML_RESTRICT x, block_iq3_k * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq4_kss_ref(const float * GGML_RESTRICT x, block_iq4_kss  * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq4_ks_ref (const float * GGML_RESTRICT x, block_iq4_ks  * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq4_k_ref  (const float * GGML_RESTRICT x, block_iq4_k   * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq5_k_ref  (const float * GGML_RESTRICT x, block_iq5_k   * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq5_ks_ref  (const float * GGML_RESTRICT x, block_iq5_ks   * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq3_ks_v1_ref  (const float * GGML_RESTRICT x, block_iq3_ks_v1   * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq6_k_ref  (const float * GGML_RESTRICT x, block_iq6_k   * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq2_kt_ref (const float * GGML_RESTRICT x, block_iq2_kt   * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq3_kt_ref (const float * GGML_RESTRICT x, block_iq3_kt   * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq4_kt_ref (const float * GGML_RESTRICT x, block_iq4_kt   * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq1_bn_ref(const float * GGML_RESTRICT x, block_iq1_bn * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq2_bn_ref(const float * GGML_RESTRICT x, block_iq2_bn * GGML_RESTRICT y, int64_t k);


// GGML_API void quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_q4_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_q5_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_q5_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_q8_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_q6_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

// GGML_API void quantize_row_q2_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_q3_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_q4_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_q5_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_q8_K64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);


// GGML_API void quantize_row_iq2_xxs(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq2_xs (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq3_xxs(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq4_nl (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq4_xs (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq3_s  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq2_s  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_tq1_0 (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_tq2_0 (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);


// GGML_API void quantize_row_iq2_ks(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq2_k (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq3_k (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq4_kss (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq4_ks  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq4_k (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq5_k (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq5_ks (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq3_ks_v1 (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq6_k (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq2_kt (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq3_kt (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq4_kt (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq1_bn (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
// GGML_API void quantize_row_iq2_bn (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);


// Dequantization
GGML_API void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_1(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_1(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q6_0(const block_q6_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_tq1_0(const block_tq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tq2_0(const block_tq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_xs (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_s  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_xxs(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq1_s  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq1_m  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_nl (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_xs (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_s  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);


// GGML_API void dequantize_row_iq2_ks(const block_iq2_ks * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
// GGML_API void dequantize_row_iq2_k (const block_iq2_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
// GGML_API void dequantize_row_iq3_k (const block_iq3_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
// GGML_API void dequantize_row_iq4_kss (const block_iq4_kss  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
// GGML_API void dequantize_row_iq4_ks (const block_iq4_ks  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
// GGML_API void dequantize_row_iq4_k  (const block_iq4_k   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
// GGML_API void dequantize_row_iq5_k  (const block_iq5_k   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
// GGML_API void dequantize_row_iq5_ks  (const block_iq5_ks   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
// GGML_API void dequantize_row_iq3_ks_v1  (const block_iq3_ks_v1   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
// GGML_API void dequantize_row_iq6_k  (const block_iq6_k   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
// GGML_API void dequantize_row_iq2_kt (const block_iq2_kt   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
// GGML_API void dequantize_row_iq3_kt (const block_iq3_kt   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
// GGML_API void dequantize_row_iq4_kt (const block_iq4_kt   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
// GGML_API void dequantize_row_iq1_bn (const block_iq1_bn * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
// GGML_API void dequantize_row_iq2_bn (const block_iq2_bn * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);


// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq1_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq1_m  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_nl (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);


// GGML_API size_t quantize_iq2_ks (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// GGML_API size_t quantize_iq2_k (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// GGML_API size_t quantize_iq3_k  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// GGML_API size_t quantize_iq4_kss (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// GGML_API size_t quantize_iq4_ks (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// GGML_API size_t quantize_iq4_k (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// GGML_API size_t quantize_iq5_k  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// GGML_API size_t quantize_iq5_ks  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// GGML_API size_t quantize_iq3_ks_v1  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// GGML_API size_t quantize_iq6_k  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// GGML_API size_t quantize_iq2_kt (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// GGML_API size_t quantize_iq3_kt (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// GGML_API size_t quantize_iq4_kt (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// GGML_API size_t quantize_iq1_bn (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// GGML_API size_t quantize_iq2_bn (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);


GGML_API size_t quantize_q2_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q3_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q6_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q8_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q6_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void iq2xs_init_impl(enum ggml_type type);
GGML_API void iq2xs_free_impl(enum ggml_type type);
GGML_API void iq3xs_init_impl(int grid_size);
GGML_API void iq3xs_free_impl(int grid_size);

GGML_API void iq1s_process_1block(int block_size, const float * xb, const float * weight, int8_t * L,
        float * the_scale, uint16_t * the_index, int * the_shift, float * pairs, float * sumx, float * sumw);
GGML_API void iq1m_process_1block(const float * xb, const float * weight, int8_t * L,
        float * the_scale, uint16_t * the_index, int * the_shift, float * pairs);

#ifdef __cplusplus
}
#endif
