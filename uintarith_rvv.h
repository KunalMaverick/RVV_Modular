#ifndef UINTARITH_RVV_1_0_H
#define UINTARITH_RVV_1_0_H

#include <riscv_vector.h>
#include <stddef.h>

static inline void add_uint64_rvv(vuint64m1_t *a,
                                  vuint64m1_t *b,
                                  vuint64m1_t *result,
                                  vbool64_t *carry_out,
                                  size_t vl)
{
    *result = __riscv_vadd_vv_u64m1(*a, *b, vl);
    *carry_out = __riscv_vmsltu_vv_u64m1_b64(*result, *a, vl);
}

static inline void add_uint64_with_carry_rvv(vuint64m1_t *a,
                                             vuint64m1_t *b,
                                             vuint64m1_t *result,
                                             vbool64_t *carry_in,
                                             vbool64_t *carry_out,
                                             size_t vl)
{
    *result = __riscv_vadd_vv_u64m1(*a, *b, vl);
    *result = __riscv_vadc_vxm_u64m1_tu(*result, *result, 0, *carry_in, vl);
    *carry_out = __riscv_vmsltu_vv_u64m1_b64(*result, *a, vl);
}

static inline void multiply_uint64_rvv(vuint64m1_t *a,
                                       vuint64m1_t *b,
                                       vuint64m1_t *result_lo,
                                       vuint64m1_t *result_hi,
                                       size_t vl)
{
    *result_lo = __riscv_vmul_vv_u64m1(*a, *b, vl);
    *result_hi = __riscv_vmulhu_vv_u64m1(*a, *b, vl);
}

static inline void multiply_uint64_vx_rvv(vuint64m1_t *a,
                                          uint64_t b,
                                          vuint64m1_t *result_lo,
                                          vuint64m1_t *result_hi,
                                          size_t vl)
{
    *result_lo = __riscv_vmul_vx_u64m1(*a, b, vl);
    *result_hi = __riscv_vmulhu_vx_u64m1(*a, b, vl);
}

static inline void subtract_uint64_rvv(vuint64m1_t *a,
                                       vuint64m1_t *b,
                                       vuint64m1_t *result,
                                       vbool64_t *borrow_out,
                                       size_t vl)
{
    // TODO: implement
}

static inline void subtract_uint64_with_borrow_rvv(vuint64m1_t *a,
                                                  vuint64m1_t *b,
                                                  vuint64m1_t *result,
                                                  vbool64_t *borrow_in,
                                                  vbool64_t *borrow_out,
                                                  size_t vl)
{
    //TODO: implement 

}

static inline void shift_right_uint128_rvv(vuint64m1_t *input_lo,
                                           vuint64m1_t *input_hi,
                                           size_t shift_amount,
                                           size_t uint64_bit_size,
                                           vuint64m1_t *out_lo,
                                           vuint64m1_t *out_hi,
                                           size_t vl)
{
    if (shift_amount == 0) {
        *out_lo = *input_lo;
        *out_hi = *input_hi;
        return;
    }
    size_t neg_bit_shift_amount = uint64_bit_size - shift_amount;
    *out_lo = __riscv_vsrl_vx_u64m1(*input_lo, shift_amount, vl);
    vuint64m1_t temp_hi_for_lo = __riscv_vsll_vx_u64m1(*input_hi, neg_bit_shift_amount, vl);
    *out_lo = __riscv_vor_vv_u64m1(*out_lo, temp_hi_for_lo, vl);
    *out_hi = __riscv_vsrl_vx_u64m1(*input_hi, shift_amount, vl);
}

#endif // UINTARITH_RVV_1_0_H