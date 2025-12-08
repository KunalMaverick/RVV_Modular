#ifndef UTIL_MODULAR_H
#define UTIL_MODULAR_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <riscv_vector.h>
#include "uint_mod_arith_rvv.h"

// ============================================================================
// 1) ALLOCATION
// ============================================================================

static inline uint64_t *allocate_tensor_1d_uint64(size_t n)
{
    return (uint64_t *)malloc(n * sizeof(uint64_t));
}

// ============================================================================
// 2) PURE MODULAR ARITHMETIC (RVV VERSION)
// ============================================================================

// ---------------------- Add (vector + vector) ----------------------
static inline void add_vv_mod_u64(size_t n,
                                  const uint64_t *a,
                                  const uint64_t *b,
                                  uint64_t modulus,
                                  uint64_t *c)
{
    size_t vl;
    while (n > 0) {
        vl = __riscv_vsetvl_e64m8(n);

        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl);
        vuint64m8_t vb = __riscv_vle64_v_u64m8(b, vl);
        vuint64m8_t vs = __riscv_vadd_vv_u64m8(va, vb, vl);

        vbool8_t mask = __riscv_vmsgeu_vx_u64m8_b8(vs, modulus, vl);
        vuint64m8_t vs2 = __riscv_vsub_vx_u64m8(vs, modulus, vl);
        vuint64m8_t vc = __riscv_vmerge_vvm_u64m8(vs, vs2, mask, vl);

        __riscv_vse64_v_u64m8(c, vc, vl);

        n -= vl; a += vl; b += vl; c += vl;
    }
}

// ---------------------- Add (vector + scalar) ----------------------
static inline void add_vx_mod_u64(size_t n,
                                  const uint64_t *a,
                                  uint64_t b,
                                  uint64_t modulus,
                                  uint64_t *c)
{
    size_t vl;
    while (n > 0) {
        vl = __riscv_vsetvl_e64m8(n);

        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl);
        vuint64m8_t vs = __riscv_vadd_vx_u64m8(va, b, vl);

        vbool8_t mask = __riscv_vmsgeu_vx_u64m8_b8(vs, modulus, vl);
        vuint64m8_t vs2 = __riscv_vsub_vx_u64m8(vs, modulus, vl);
        vuint64m8_t vc = __riscv_vmerge_vvm_u64m8(vs, vs2, mask, vl);

        __riscv_vse64_v_u64m8(c, vc, vl);

        n -= vl; a += vl; c += vl;
    }
}

// ---------------------- Sub (vector - vector) ----------------------
static inline void sub_vv_mod_u64(size_t n,
                                  const uint64_t *a,
                                  const uint64_t *b,
                                  uint64_t modulus,
                                  uint64_t *c)
{
    size_t vl;
    while (n > 0) {
        vl = __riscv_vsetvl_e64m8(n);

        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl);
        vuint64m8_t vb = __riscv_vle64_v_u64m8(b, vl);
        vuint64m8_t vd = __riscv_vsub_vv_u64m8(va, vb, vl);

        vbool8_t mask = __riscv_vmsltu_vv_u64m8_b8(va, vb, vl);
        vuint64m8_t vd2 = __riscv_vadd_vx_u64m8(vd, modulus, vl);
        vuint64m8_t vc = __riscv_vmerge_vvm_u64m8(vd, vd2, mask, vl);

        __riscv_vse64_v_u64m8(c, vc, vl);

        n -= vl; a += vl; b += vl; c += vl;
    }
}

// ---------------------- Sub (vector - scalar) ----------------------
static inline void sub_vx_mod_u64(size_t n,
                                  const uint64_t *a,
                                  uint64_t b,
                                  uint64_t modulus,
                                  uint64_t *c)
{
    size_t vl;
    while (n > 0) {
        vl = __riscv_vsetvl_e64m8(n);

        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl);
        vuint64m8_t vd = __riscv_vsub_vx_u64m8(va, b, vl);

        vbool8_t mask = __riscv_vmsltu_vx_u64m8_b8(va, b, vl);
        vuint64m8_t vd2 = __riscv_vadd_vx_u64m8(vd, modulus, vl);
        vuint64m8_t vc = __riscv_vmerge_vvm_u64m8(vd, vd2, mask, vl);

        __riscv_vse64_v_u64m8(c, vc, vl);

        n -= vl; a += vl; c += vl;
    }
}

// ---------------------- Negation ----------------------
static inline void neg_v_mod_u64(size_t n,
                                 const uint64_t *a,
                                 uint64_t modulus,
                                 uint64_t *c)
{
    size_t vl;
    while (n > 0) {
        vl = __riscv_vsetvl_e64m8(n);

        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl);
        vuint64m8_t vt = __riscv_vrsub_vx_u64m8(va, modulus, vl);

        vbool8_t mask = __riscv_vmsne_vx_u64m8_b8(va, 0, vl);
        vuint64m8_t v0 = __riscv_vmv_v_x_u64m8(0, vl);
        vuint64m8_t vc = __riscv_vmerge_vvm_u64m8(v0, vt, mask, vl);

        __riscv_vse64_v_u64m8(c, vc, vl);

        n -= vl; a += vl; c += vl;
    }
}

// ---------------------- Multiplication with Barrett ----------------------
static inline void mul_vv_mod_u64(
    size_t n, const uint64_t *a, const uint64_t *b,
    uint64_t modulus, const uint64_t *mu,
    uint64_t *ab_lo, uint64_t *ab_hi, uint64_t *c)
{
    size_t vl;
    for (size_t i = 0; i < n; ) {
        vl = __riscv_vsetvl_e64m8(n - i);
        vuint64m8_t va = __riscv_vle64_v_u64m8(a + i, vl);
        vuint64m8_t vb = __riscv_vle64_v_u64m8(b + i, vl);

        // compute lo, hi
        vuint64m8_t lo = __riscv_vmul_vv_u64m8(va, vb, vl);
        vuint64m8_t hi = __riscv_vmulhu_vv_u64m8(va, vb, vl);

        // interleave so barrett_reduce reads correctly
        __riscv_vse64_v_u64m8(ab_lo + i, lo, vl);
        __riscv_vse64_v_u64m8(ab_hi + i, hi, vl);

        i += vl;
    }

    // IMPORTANT: pass mu as non-const pointer
    barrett_reduce_rvv(ab_lo, ab_hi, c, modulus, (uint64_t*)mu, n);
}

// ---------------------- Multiplication (vector * scalar) with Barrett ----------------------
static inline void mul_vx_mod_u64(
    size_t n, const uint64_t *a, uint64_t b,
    uint64_t modulus, const uint64_t *mu,
    uint64_t *ab_lo, uint64_t *ab_hi, uint64_t *c)
{
    size_t vl;
    for (size_t i = 0; i < n; ) {
        vl = __riscv_vsetvl_e64m8(n - i);
        vuint64m8_t va = __riscv_vle64_v_u64m8(a + i, vl);

        // compute lo, hi (vector * scalar)
        vuint64m8_t lo = __riscv_vmul_vx_u64m8(va, b, vl);
        vuint64m8_t hi = __riscv_vmulhu_vx_u64m8(va, b, vl);

        // store them so barrett_reduce sees paired hi/lo
        __riscv_vse64_v_u64m8(ab_lo + i, lo, vl);
        __riscv_vse64_v_u64m8(ab_hi + i, hi, vl);

        i += vl;
    }

    // IMPORTANT: cast away const to satisfy the library
    barrett_reduce_rvv(ab_lo, ab_hi, c, modulus, (uint64_t *)mu, n);
}

// ============================================================================
// MATRIX UTILITIES (allocation + random fill)
// ============================================================================

// Allocate contiguous dim x dim matrix (row-major layout)
static inline uint64_t *allocate_matrix_u64(size_t dim)
{
    uint64_t *ptr = (uint64_t *)malloc(dim * dim * sizeof(uint64_t));
    if (!ptr) {
        fprintf(stderr, "Matrix allocation failed for %zu x %zu\n", dim, dim);
        exit(1);
    }
    return ptr;
}

// Fill matrix with random 64-bit values (no modulo)
static inline void fill_matrix_rand_u64(uint64_t *M, size_t dim)
{
    size_t total = dim * dim;
    for (size_t i = 0; i < total; i++) {
        M[i] = ((uint64_t)rand() << 32) ^ (uint64_t)rand();
    }
}



// ============================================================================
// RVV MATRIX MULTIPLICATION
// A: row-major (dim x dim)
// B_t: row-major transpose of B (so B_t[j*dim + k] = B[k*dim + j])
// C: row-major (dim x dim)
// ============================================================================
static inline void matmul_u64_rvv(const uint64_t *A,
                                  const uint64_t *B_t,
                                  uint64_t *C,
                                  size_t dim)
{
    for (size_t i = 0; i < dim; i++) {
        const uint64_t *rowA_base = A + i * dim;

        for (size_t j = 0; j < dim; j++) {
            const uint64_t *rowBt_base = B_t + j * dim;

            uint64_t sum = 0;
            size_t k = 0;

            while (k < dim) {
                size_t vl = __riscv_vsetvl_e64m8(dim - k);

                // A[i][k .. k+vl-1]
                vuint64m8_t va = __riscv_vle64_v_u64m8(rowA_base + k, vl);

                // B_t[j][k .. k+vl-1] == B[k..][j]
                vuint64m8_t vb = __riscv_vle64_v_u64m8(rowBt_base + k, vl);

                // elementwise product
                vuint64m8_t vprod = __riscv_vmul_vv_u64m8(va, vb, vl);

                // horizontal reduction sum(vprod[0..vl-1]) → scalar
                vuint64m1_t vsum_m1 = __riscv_vmv_s_x_u64m1(0, vl);
                vsum_m1 = __riscv_vredsum_vs_u64m8_u64m1(vprod, vsum_m1, vl);
                uint64_t partial = __riscv_vmv_x_s_u64m1_u64(vsum_m1);

                sum += partial;
                k += vl;
            }

            C[i * dim + j] = sum;
        }
    }
}

static inline void matmul_mod_u64_rvv(
    const uint64_t *A,
    const uint64_t *B_t,        // B pre-transposed
    uint64_t *C,
    size_t dim,
    uint64_t modulus,
    const uint64_t *mu,         // Barrett constant
    uint64_t *tmp_lo,           // size dim
    uint64_t *tmp_hi,           // size dim
    uint64_t *tmp_out           // size dim
)
{
    for (size_t i = 0; i < dim; i++) {

        const uint64_t *rowA = A + i * dim;

        for (size_t j = 0; j < dim; j++) {

            const uint64_t *rowBt = B_t + j * dim;

            // ----------------------------------------------
            // Compute vector: tmp_out[k] = (A[i][k] * B[k][j]) mod modulus
            // ----------------------------------------------
            mul_vv_mod_u64(
                dim,
                rowA,
                rowBt,
                modulus,
                mu,
                tmp_lo,
                tmp_hi,
                tmp_out
            );

            // ----------------------------------------------
            // Reduce tmp_out[k] into a scalar sum mod modulus
            // SEAL-style modular accumulation
            // ----------------------------------------------

            uint64_t sum = 0;

            for (size_t k = 0; k < dim; k++) {
                // sum = (sum + tmp_out[k]) mod modulus
                uint64_t s = sum + tmp_out[k];
                if (s >= modulus) s -= modulus;
                sum = s;
            }

            C[i*dim + j] = sum;
        }
    }
}




void matmul_u64_256x256(const uint64_t A[256][256], 
                        const uint64_t B[256][256], 
                        uint64_t C[256][256]) {
    // Zero out C matrix first
    for (size_t i = 0; i < 256; i++) {
        for (size_t j = 0; j < 256; j++) {
            C[i][j] = 0;
        }
    }
    
    // C[i][j] = sum(A[i][k] * B[k][j]) for k=0..255
    for (size_t i = 0; i < 256; i++) {
        for (size_t j = 0; j < 256; j++) {
            uint64_t sum = 0;
            size_t k = 0;
            
            // Vectorized dot product of row A[i] and column B[:,j]
            while (k < 256) {
                size_t vl = __riscv_vsetvl_e64m8(256 - k);
                
                // Load A[i][k:k+vl] - contiguous row access
                vuint64m8_t va = __riscv_vle64_v_u64m8(&A[i][k], vl);
                
                // Load B[k:k+vl][j] - strided column access (stride = 256)
                vuint64m8_t vb = __riscv_vlse64_v_u64m8(&B[k][j], 256 * sizeof(uint64_t), vl);
                
                // Multiply: va * vb
                vuint64m8_t vprod = __riscv_vmul_vv_u64m8(va, vb, vl);
                
                // Reduce sum - extract scalar sum from vector
                vuint64m1_t vsum_m1 = __riscv_vmv_s_x_u64m1(0, vl);  // Initialize scalar to 0
                vsum_m1 = __riscv_vredsum_vs_u64m8_u64m1(vprod, vsum_m1, vl);
                uint64_t partial_sum = __riscv_vmv_x_s_u64m1_u64(vsum_m1);
                
                sum += partial_sum;
                k += vl;
            }
            
            C[i][j] = sum;
        }
    }
}

// Instrumented modular matmul: inspect one output (inspect_i, inspect_j)
// - tmp_lo/tmp_hi/tmp_out must be size 'dim' and provided by caller
// - max_print_k limits how many k entries to print (set small e.g. 8 or 16)
static inline void matmul_mod_u64_rvv_inspect(
    const uint64_t *A,
    const uint64_t *B_t,        // B pre-transposed
    uint64_t *C,
    size_t dim,
    uint64_t modulus,
    const uint64_t *mu,         // Barrett constant
    uint64_t *tmp_lo,           // size dim
    uint64_t *tmp_hi,           // size dim
    uint64_t *tmp_out,          // size dim
    size_t inspect_i,           // which row to inspect
    size_t inspect_j,           // which col to inspect
    size_t max_print_k          // how many k values to print
)
{
    for (size_t i = 0; i < dim; i++) {

        const uint64_t *rowA = A + i * dim;

        for (size_t j = 0; j < dim; j++) {

            const uint64_t *rowBt = B_t + j * dim;

            // Compute vector: tmp_lo/tmp_hi for all k (vectorized)
            mul_vv_mod_u64(
                dim,
                rowA,
                rowBt,
                modulus,
                mu,
                tmp_lo,
                tmp_hi,
                tmp_out
            );

            // If this is the inspected output, print internal info
            int want_inspect = (i == inspect_i && j == inspect_j) ? 1 : 0;
            if (want_inspect) {
                size_t limit = max_print_k < dim ? max_print_k : dim;
                printf("\n=== INSPECTING C[%zu,%zu] (printing first %zu k entries) ===\n",
                       i, j, limit);
                printf("k |    A[i,k]    B[k,j]    lo (low64)     hi (high64)   tmp_out (mod)\n");
                printf("-----------------------------------------------------------------------\n");
                for (size_t k = 0; k < limit; k++) {
                    printf("%2zu | %10lu  %10lu  %12lu  %12lu  %12lu\n",
                           k, rowA[k], rowBt[k], tmp_lo[k], tmp_hi[k], tmp_out[k]);
                }
            }

            // Now reduce tmp_out[k] into a scalar sum mod modulus (SEAL style)
            uint64_t sum = 0;
            if (!want_inspect) {
                // Fast path: no extra prints
                for (size_t k = 0; k < dim; k++) {
                    uint64_t s = sum + tmp_out[k];
                    if (s >= modulus) s -= modulus;
                    sum = s;
                }
            } else {
                // Inspecting accumulation step-by-step
                printf("\nAccumulation trace for C[%zu,%zu]:\n", i, j);
                printf("step | tmp_out[k] | sum_before -> sum_after\n");
                printf("-----------------------------------------\n");
                for (size_t k = 0; k < dim; k++) {
                    uint64_t before = sum;
                    uint64_t s = sum + tmp_out[k];
                    uint64_t after = (s >= modulus) ? (s - modulus) : s;
                    // Print only first max_print_k steps, but compute all
                    if (k < max_print_k) {
                        printf("%4zu | %10lu | %8lu -> %8lu\n", k, tmp_out[k], before, after);
                    } else if (k == max_print_k) {
                        printf(" ... (skipping remaining steps for brevity) ...\n");
                    }
                    sum = after;
                }
                printf("Final C[%zu,%zu] = %lu\n", i, j, sum);
            }

            C[i*dim + j] = sum;
        }
    }
}



#endif
