// util.h
// Common utilities with modular arithmetic support
// Structure: Allocation → Operations → Deallocation

#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <riscv_vector.h>
#include "rvv_arith.h" 

// ==================== MEMORY ALLOCATION ====================

static inline float* allocate_tensor_1d_float(size_t n) {
    float* tensor = (float*)malloc(n * sizeof(float));
    if (tensor == NULL) {
        fprintf(stderr, "[alloc_float] Failed to allocate tensor of size %zu\n", n);
        return NULL;
    }
    return tensor;
}

static inline int* allocate_tensor_1d_int(size_t n) {
    int* tensor = (int*)malloc(n * sizeof(int));
    if (tensor == NULL) {
        fprintf(stderr, "[alloc_int] Failed to allocate tensor of size %zu\n", n);
        return NULL;
    }
    return tensor;
}

static inline uint32_t* allocate_tensor_1d_uint32(size_t n) {
    uint32_t* tensor = (uint32_t*)malloc(n * sizeof(uint32_t));
    if (tensor == NULL) {
        fprintf(stderr, "[alloc_uint32] Failed to allocate tensor of size %zu\n", n);
        return NULL;
    }
    return tensor;
}

static inline uint64_t* allocate_tensor_1d_uint64(size_t n) {
    uint64_t* tensor = (uint64_t*)malloc(n * sizeof(uint64_t));
    if (tensor == NULL) {
        fprintf(stderr, "[alloc_uint64] Failed to allocate tensor of size %zu\n", n);
        return NULL;
    }
    return tensor;
}

// ==================== VECTOR OPERATIONS ====================

// -------------------- Random Fill Functions --------------------

static inline void fill_float_tensor_rvv_rand(float *tensor, size_t n) {
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        float tmp[vl];
        for (size_t j = 0; j < vl; j++) {
            tmp[j] = (float)rand();
        }
        vfloat32m8_t vdata = __riscv_vle32_v_f32m8(tmp, vl);
        __riscv_vse32_v_f32m8(tensor + i, vdata, vl);
        i += vl;
    }
}

static inline void fill_int_tensor_rvv_rand(int *tensor, size_t n) {
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        int tmp[vl];
        for (size_t j = 0; j < vl; j++) {
            tmp[j] = (int)rand();
        }
        vint32m8_t vdata = __riscv_vle32_v_i32m8(tmp, vl);
        __riscv_vse32_v_i32m8(tensor + i, vdata, vl);
        i += vl;
    }
}

static inline void fill_u32_tensor_rvv_rand(uint32_t *tensor, size_t n) {
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        uint32_t tmp[vl];
        for (size_t j = 0; j < vl; j++) {
            tmp[j] = (uint32_t)rand();
        }
        vuint32m8_t vdata = __riscv_vle32_v_u32m8(tmp, vl);
        __riscv_vse32_v_u32m8(tensor + i, vdata, vl);
        i += vl;
    }
}

static inline void fill_u64_tensor_rvv_rand(uint64_t *tensor, size_t n) {
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e64m8(n - i);
        uint64_t tmp[vl];
        for (size_t j = 0; j < vl; j++) {
            tmp[j] = (uint64_t)rand();
        }
        vuint64m8_t vdata = __riscv_vle64_v_u64m8(tmp, vl);
        __riscv_vse64_v_u64m8(tensor + i, vdata, vl);
        i += vl;
    }
}

// -------------------- Basic Arithmetic: int32 --------------------

void add_vv_int32(size_t n, const int32_t *a, const int32_t *b, int32_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, b += vl, c += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vint32m8_t va = __riscv_vle32_v_i32m8(a, vl);
        vint32m8_t vb = __riscv_vle32_v_i32m8(b, vl);
        vint32m8_t vc = __riscv_vadd_vv_i32m8(va, vb, vl);
        __riscv_vse32_v_i32m8(c, vc, vl);
    }
}

void add_vx_int32(size_t n, const int32_t *a, int32_t b, int32_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, c += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vint32m8_t va = __riscv_vle32_v_i32m8(a, vl);
        vint32m8_t vc = __riscv_vadd_vx_i32m8(va, b, vl);
        __riscv_vse32_v_i32m8(c, vc, vl);
    }
}

void sub_vv_int32(size_t n, const int32_t *a, const int32_t *b, int32_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, b += vl, c += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vint32m8_t va = __riscv_vle32_v_i32m8(a, vl);
        vint32m8_t vb = __riscv_vle32_v_i32m8(b, vl);
        vint32m8_t vc = __riscv_vsub_vv_i32m8(va, vb, vl);
        __riscv_vse32_v_i32m8(c, vc, vl);
    }
}

void sub_vx_int32(size_t n, const int32_t *a, int32_t b, int32_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, c += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vint32m8_t va = __riscv_vle32_v_i32m8(a, vl);
        vint32m8_t vc = __riscv_vsub_vx_i32m8(va, b, vl);
        __riscv_vse32_v_i32m8(c, vc, vl);
    }
}

// -------------------- Basic Arithmetic: uint64 --------------------

void add_vv_u64(size_t n, const uint64_t *a, const uint64_t *b, uint64_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, b += vl, c += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl);
        vuint64m8_t vb = __riscv_vle64_v_u64m8(b, vl);
        vuint64m8_t vc = __riscv_vadd_vv_u64m8(va, vb, vl);
        __riscv_vse64_v_u64m8(c, vc, vl);
    }
}

void add_vx_u64(size_t n, const uint64_t *a, uint64_t b, uint64_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, c += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl);
        vuint64m8_t vc = __riscv_vadd_vx_u64m8(va, b, vl);
        __riscv_vse64_v_u64m8(c, vc, vl);
    }
}

void sub_vv_u64(size_t n, const uint64_t *a, const uint64_t *b, uint64_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, b += vl, c += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl);
        vuint64m8_t vb = __riscv_vle64_v_u64m8(b, vl);
        vuint64m8_t vc = __riscv_vsub_vv_u64m8(va, vb, vl);
        __riscv_vse64_v_u64m8(c, vc, vl);
    }
}

void sub_vx_u64(size_t n, const uint64_t *a, uint64_t b, uint64_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, c += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl);
        vuint64m8_t vc = __riscv_vsub_vx_u64m8(va, b, vl);
        __riscv_vse64_v_u64m8(c, vc, vl);
    }
}

void neg_v_uint64(size_t n, const uint64_t *a, uint64_t *b) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, b += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        vuint64m8_t va_u = __riscv_vle64_v_u64m8(a, vl);
        vint64m8_t va_i = __riscv_vreinterpret_v_u64m8_i64m8(va_u);
        vint64m8_t vb_i = __riscv_vneg_v_i64m8(va_i, vl);
        vuint64m8_t vb_u = __riscv_vreinterpret_v_i64m8_u64m8(vb_i);
        __riscv_vse64_v_u64m8(b, vb_u, vl);
    }
}

// -------------------- Modular Arithmetic: uint32 --------------------

/**
 * Modular Addition (uint32): c = (a + b) mod m
 * Formula: c = a + b - m * [a + b >= m]
 * Precondition: a, b < m and (a + b) < 2*m
 */
void add_vv_mod_u32(size_t n, const uint32_t *a, const uint32_t *b, 
                    uint32_t modulus, uint32_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, b += vl, c += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        
        vuint32m8_t va = __riscv_vle32_v_u32m8(a, vl);
        vuint32m8_t vb = __riscv_vle32_v_u32m8(b, vl);
        vuint32m8_t vsum = __riscv_vadd_vv_u32m8(va, vb, vl);
        
        vbool4_t mask = __riscv_vmsgeu_vx_u32m8_b4(vsum, modulus, vl);
        vuint32m8_t vsum_minus_mod = __riscv_vsub_vx_u32m8(vsum, modulus, vl);
        vuint32m8_t vc = __riscv_vmerge_vvm_u32m8(vsum, vsum_minus_mod, mask, vl);
        
        __riscv_vse32_v_u32m8(c, vc, vl);
    }
}

void add_vx_mod_u32(size_t n, const uint32_t *a, uint32_t b, 
                    uint32_t modulus, uint32_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, c += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        
        vuint32m8_t va = __riscv_vle32_v_u32m8(a, vl);
        vuint32m8_t vsum = __riscv_vadd_vx_u32m8(va, b, vl);
        
        vbool4_t mask = __riscv_vmsgeu_vx_u32m8_b4(vsum, modulus, vl);
        vuint32m8_t vsum_minus_mod = __riscv_vsub_vx_u32m8(vsum, modulus, vl);
        vuint32m8_t vc = __riscv_vmerge_vvm_u32m8(vsum, vsum_minus_mod, mask, vl);
        
        __riscv_vse32_v_u32m8(c, vc, vl);
    }
}

/**
 * Modular Subtraction (uint32): c = (a - b) mod m
 * Formula: c = a - b + m * [a < b]
 * Precondition: a, b < m
 */
void sub_vv_mod_u32(size_t n, const uint32_t *a, const uint32_t *b, 
                    uint32_t modulus, uint32_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, b += vl, c += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        
        vuint32m8_t va = __riscv_vle32_v_u32m8(a, vl);
        vuint32m8_t vb = __riscv_vle32_v_u32m8(b, vl);
        vuint32m8_t vdiff = __riscv_vsub_vv_u32m8(va, vb, vl);
        
        vbool4_t mask = __riscv_vmsltu_vv_u32m8_b4(va, vb, vl);
        vuint32m8_t vdiff_plus_mod = __riscv_vadd_vx_u32m8(vdiff, modulus, vl);
        vuint32m8_t vc = __riscv_vmerge_vvm_u32m8(vdiff, vdiff_plus_mod, mask, vl);
        
        __riscv_vse32_v_u32m8(c, vc, vl);
    }
}

void sub_vx_mod_u32(size_t n, const uint32_t *a, uint32_t b, 
                    uint32_t modulus, uint32_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, c += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        
        vuint32m8_t va = __riscv_vle32_v_u32m8(a, vl);
        vuint32m8_t vdiff = __riscv_vsub_vx_u32m8(va, b, vl);
        
        vbool4_t mask = __riscv_vmsltu_vx_u32m8_b4(va, b, vl);
        vuint32m8_t vdiff_plus_mod = __riscv_vadd_vx_u32m8(vdiff, modulus, vl);
        vuint32m8_t vc = __riscv_vmerge_vvm_u32m8(vdiff, vdiff_plus_mod, mask, vl);
        
        __riscv_vse32_v_u32m8(c, vc, vl);
    }
}

// -------------------- Modular Arithmetic: uint64 --------------------

/**
 * Modular Addition (uint64): c = (a + b) mod m
 * Formula: c = a + b - m * [a + b >= m]
 * Precondition: a, b < m and (a + b) < 2*m
 */
void add_vv_mod_u64(size_t n, const uint64_t *a, const uint64_t *b, 
                    uint64_t modulus, uint64_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, b += vl, c += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        
        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl);
        vuint64m8_t vb = __riscv_vle64_v_u64m8(b, vl);
        vuint64m8_t vsum = __riscv_vadd_vv_u64m8(va, vb, vl);
        
        vbool8_t mask = __riscv_vmsgeu_vx_u64m8_b8(vsum, modulus, vl);
        vuint64m8_t vsum_minus_mod = __riscv_vsub_vx_u64m8(vsum, modulus, vl);
        vuint64m8_t vc = __riscv_vmerge_vvm_u64m8(vsum, vsum_minus_mod, mask, vl);
        
        __riscv_vse64_v_u64m8(c, vc, vl);
    }
}

void add_vx_mod_u64(size_t n, const uint64_t *a, uint64_t b, 
                    uint64_t modulus, uint64_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, c += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        
        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl);
        vuint64m8_t vsum = __riscv_vadd_vx_u64m8(va, b, vl);
        
        vbool8_t mask = __riscv_vmsgeu_vx_u64m8_b8(vsum, modulus, vl);
        vuint64m8_t vsum_minus_mod = __riscv_vsub_vx_u64m8(vsum, modulus, vl);
        vuint64m8_t vc = __riscv_vmerge_vvm_u64m8(vsum, vsum_minus_mod, mask, vl);
        
        __riscv_vse64_v_u64m8(c, vc, vl);
    }
}

/**
 * Modular Subtraction (uint64): c = (a - b) mod m
 * Formula: c = a - b + m * [a < b]
 * Precondition: a, b < m
 */
void sub_vv_mod_u64(size_t n, const uint64_t *a, const uint64_t *b, 
                    uint64_t modulus, uint64_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, b += vl, c += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        
        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl);
        vuint64m8_t vb = __riscv_vle64_v_u64m8(b, vl);
        vuint64m8_t vdiff = __riscv_vsub_vv_u64m8(va, vb, vl);
        
        vbool8_t mask = __riscv_vmsltu_vv_u64m8_b8(va, vb, vl);
        vuint64m8_t vdiff_plus_mod = __riscv_vadd_vx_u64m8(vdiff, modulus, vl);
        vuint64m8_t vc = __riscv_vmerge_vvm_u64m8(vdiff, vdiff_plus_mod, mask, vl);
        
        __riscv_vse64_v_u64m8(c, vc, vl);
    }
}

void sub_vx_mod_u64(size_t n, const uint64_t *a, uint64_t b, 
                    uint64_t modulus, uint64_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, c += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        
        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl);
        vuint64m8_t vdiff = __riscv_vsub_vx_u64m8(va, b, vl);
        
        vbool8_t mask = __riscv_vmsltu_vx_u64m8_b8(va, b, vl);
        vuint64m8_t vdiff_plus_mod = __riscv_vadd_vx_u64m8(vdiff, modulus, vl);
        vuint64m8_t vc = __riscv_vmerge_vvm_u64m8(vdiff, vdiff_plus_mod, mask, vl);
        
        __riscv_vse64_v_u64m8(c, vc, vl);
    }
}

/**
 * Modular Negation (uint64): c = (-a) mod m
 * Formula: c = (m - a) * [a != 0]
 * Precondition: a < m
 */
void neg_v_mod_u64(size_t n, const uint64_t *a, uint64_t modulus, uint64_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, c += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        
        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl);
        vuint64m8_t vtemp = __riscv_vrsub_vx_u64m8(va, modulus, vl);
        
        vbool8_t mask = __riscv_vmsne_vx_u64m8_b8(va, 0, vl);
        vuint64m8_t vzero = __riscv_vmv_v_x_u64m8(0, vl);
        vuint64m8_t vc = __riscv_vmerge_vvm_u64m8(vzero, vtemp, mask, vl);
        
        __riscv_vse64_v_u64m8(c, vc, vl);
    }
}

// -------------------- Modular Multiplication (Barrett Reduction) --------------------

void mul_vv_mod_u64(size_t n, const uint64_t *a, const uint64_t *b, 
                    uint64_t modulus, const uint64_t *mu, uint64_t *c) {
    // Your existing barrett_reduce_rvv already handles this!
    // Just need to multiply first
    
    size_t vl;
    uint64_t *ab_lo = allocate_tensor_1d_uint64(n);
    uint64_t *ab_hi = allocate_tensor_1d_uint64(n);
    
    // Compute products
    for (size_t i = 0; i < n; ) {
        vl = __riscv_vsetvl_e64m8(n - i);
        vuint64m8_t va = __riscv_vle64_v_u64m8(a + i, vl);
        vuint64m8_t vb = __riscv_vle64_v_u64m8(b + i, vl);
        vuint64m8_t lo = __riscv_vmul_vv_u64m8(va, vb, vl);
        vuint64m8_t hi = __riscv_vmulhu_vv_u64m8(va, vb, vl);
        __riscv_vse64_v_u64m8(ab_lo + i, lo, vl);
        __riscv_vse64_v_u64m8(ab_hi + i, hi, vl);
        i += vl;
    }
    
    // Use your existing working Barrett reduction!
    barrett_reduce_rvv(ab_lo, ab_hi, c, modulus, (uint64_t*)mu, n);
    
    free(ab_lo);
    free(ab_hi);
}

// ==================== MEMORY DEALLOCATION ====================

static inline void free_tensor_safe(void** ptr) {
    if (ptr && *ptr) {
        printf("[DEBUG] Freeing tensor at address %p\n", *ptr);
        free(*ptr);
        *ptr = NULL;
        printf("[DEBUG] Tensor freed and pointer set to NULL\n");
    }
}