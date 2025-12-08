#ifndef MOD_UINTARITH_RVV_1_0_H
#define MOD_UINTARITH_RVV_1_0_H

#include <stddef.h>        // For size_t
#include "uintarith_rvv.h" // Include the previous functions/inlines

static inline void barrett_reduce64_rvv(const uint64_t *ab_lo,
                                        uint64_t *result,
                                        uint64_t modulo,
                                        uint64_t *mu,
                                        size_t len)
{
    // TODO: Implement the Barrett reduction for 64-bit integers
}

static inline void barrett_reduce_rvv_helper(vuint64m1_t *ab_lo_vec,
                                             vuint64m1_t *ab_hi_vec,
                                             vuint64m1_t *result_vec,
                                             uint64_t modulo,
                                             uint64_t *mu,
                                             size_t vl)
{
    vuint64m1_t m1_lo, m1_hi;
    vuint64m1_t m2_lo, m2_hi;
    vuint64m1_t m3_lo, m3_hi;
    vuint64m1_t m4_lo, m4_hi;
    vuint64m1_t sum_mid1, sum_mid2;
    vbool64_t carry_mask;

    multiply_uint64_vx_rvv(ab_lo_vec, mu[0], &m1_lo, &m1_hi, vl);
    multiply_uint64_vx_rvv(ab_hi_vec, mu[0], &m2_lo, &m2_hi, vl);
    multiply_uint64_vx_rvv(ab_lo_vec, mu[1], &m3_lo, &m3_hi, vl);
    multiply_uint64_vx_rvv(ab_hi_vec, mu[1], &m4_lo, &m4_hi, vl);


    add_uint64_rvv(&m2_lo, &m3_lo, &sum_mid1, &carry_mask, vl);
    add_uint64_with_carry_rvv(&sum_mid1, &m1_hi, &sum_mid1, &carry_mask, &carry_mask, vl);

    add_uint64_with_carry_rvv(&m2_hi, &m3_hi, &sum_mid2, &carry_mask, &carry_mask, vl);
    add_uint64_with_carry_rvv(&sum_mid2, &m4_lo, &sum_mid2, &carry_mask, &carry_mask, vl); // sum_mid2 is the q

    m4_hi = __riscv_vadc_vxm_u64m1_tu(m4_hi, m4_hi, 0, carry_mask, vl);

    vuint64m1_t q_modulo = __riscv_vmul_vx_u64m1(sum_mid2, modulo, vl);
    vuint64m1_t r_vec = __riscv_vsub_vv_u64m1(*ab_lo_vec, q_modulo, vl);

    // first round of subtraction
    carry_mask = __riscv_vmsgeu_vx_u64m1_b64(r_vec, modulo, vl);
    r_vec = __riscv_vsub_vx_u64m1_tum(carry_mask, r_vec, r_vec, modulo, vl);

    // second round of subtraction
    // carry_mask = __riscv_vmsgeu_vx_u64m1_b64(r_vec, modulo, vl);
    // r_vec = __riscv_vsub_vx_u64m1_tum(carry_mask, r_vec, r_vec, modulo, vl);
    *result_vec = r_vec;
}

static inline void barrett_reduce_rvv(const uint64_t *ab_lo,
                                      const uint64_t *ab_hi,
                                      uint64_t *result,
                                      uint64_t modulo,
                                      uint64_t *mu,
                                      size_t len)
{
    size_t vl;
    for (size_t i = 0; i < len; i += vl)
    {
        vl = __riscv_vsetvl_e64m1(len - i);
        vuint64m1_t ab_lo_vec = __riscv_vle64_v_u64m1(ab_lo + i, vl);
        vuint64m1_t ab_hi_vec = __riscv_vle64_v_u64m1(ab_hi + i, vl);
        vuint64m1_t result_vec;
        barrett_reduce_rvv_helper(&ab_lo_vec, &ab_hi_vec, &result_vec, modulo, mu, vl);
        __riscv_vse64_v_u64m1(result + i, result_vec, vl);
    }
}

static inline void montgomery_core_rvv(vuint64m1_t *vab_lo,
                                       vuint64m1_t *vab_hi,
                                       vuint64m1_t *result,
                                       uint64_t modulo,
                                       uint64_t m_prime,
                                       uint64_t r_mask,
                                       uint64_t k_bits,
                                       size_t vl)
{

    // q = (ab_lo * m_prime) & r_mask
    vuint64m1_t q = __riscv_vmul_vx_u64m1(*vab_lo, m_prime, vl);
    q = __riscv_vand_vx_u64m1(q, r_mask, vl);

    // t = q * modulo + ab
    vuint64m1_t t_lo = __riscv_vmul_vx_u64m1(q, modulo, vl);
    vuint64m1_t t_hi = __riscv_vmulhu_vx_u64m1(q, modulo, vl);
    vbool64_t carry_mask;
    add_uint64_rvv(vab_lo, &t_lo, &t_lo, &carry_mask, vl);
    add_uint64_with_carry_rvv(vab_hi, &t_hi, &t_hi, &carry_mask, &carry_mask, vl);

    // t = t >> shift_amount
    vuint64m1_t t_shifted_lo, t_shifted_hi;
    shift_right_uint128_rvv(&t_lo, &t_hi, k_bits, 64, &t_shifted_lo, &t_shifted_hi, vl);


    // if t_shifted_lo >= modulo, then t_shifted_lo = t_shifted_lo - modulo - 2 times
    vbool64_t ge_mask = __riscv_vmsgeu_vx_u64m1_b64(t_shifted_lo, modulo, vl);
    t_shifted_lo= __riscv_vsub_vx_u64m1_tum(ge_mask, t_shifted_lo, t_shifted_lo, modulo, vl); 

    ge_mask = __riscv_vmsgeu_vx_u64m1_b64(t_shifted_lo, modulo, vl);
    t_shifted_lo = __riscv_vsub_vx_u64m1_tum(ge_mask, t_shifted_lo, t_shifted_lo, modulo, vl);

    *result = t_shifted_lo;
}

static inline void montgomery_transform_rvv(vuint64m1_t *a,
                                            vuint64m1_t *b,
                                            vuint64m1_t *result,
                                            uint64_t modulo,
                                            uint64_t m_prime,
                                            uint64_t r_mask,
                                            uint64_t k_bits,
                                            size_t vl)
{
    vuint64m1_t vab_hi = __riscv_vmulhu_vv_u64m1(*a, *b, vl);
    vuint64m1_t vab_lo = __riscv_vmul_vv_u64m1(*a, *b, vl);
    montgomery_core_rvv(&vab_lo, &vab_hi, result, modulo, m_prime, r_mask, k_bits, vl);
}

static inline void montgomery_reduction_rvv_with_transform_helper(vuint64m1_t *a_vec,
                                                                  vuint64m1_t *b_vec,
                                                                  vuint64m1_t *result_vec,
                                                                  uint64_t modulo,
                                                                  uint64_t m_prime,
                                                                  uint64_t r_square,
                                                                  uint64_t r_mask,
                                                                  uint64_t k_bits,
                                                                  size_t vl)
{

    vuint64m1_t am, bm;
    vuint64m1_t r_square_vec, one_vec;

    r_square_vec = __riscv_vmv_v_x_u64m1(r_square, vl);

    montgomery_transform_rvv(a_vec, &r_square_vec, &am, modulo, m_prime, r_mask, k_bits, vl);
    montgomery_transform_rvv(b_vec, &r_square_vec, &bm, modulo, m_prime, r_mask, k_bits, vl);

    montgomery_transform_rvv(&am, &bm, result_vec, modulo, m_prime, r_mask, k_bits, vl);

    one_vec = __riscv_vmv_v_x_u64m1(1UL, vl);
    montgomery_transform_rvv(result_vec, &one_vec, result_vec, modulo, m_prime, r_mask, k_bits, vl);
}

static inline void montgomery_reduction_with_transform_rvv(uint64_t *a,
                                                           uint64_t *b,
                                                           uint64_t *result,
                                                           uint64_t modulo,
                                                           uint64_t m_prime,
                                                           uint64_t r_square,
                                                           uint64_t r_mask,
                                                           uint64_t k_bits,
                                                           size_t len)
{
    size_t vl;
    for (size_t i = 0; i < len; i += vl)
    {
        vl = __riscv_vsetvl_e64m1(len - i);

        vuint64m1_t va = __riscv_vle64_v_u64m1(a + i, vl);
        vuint64m1_t vb = __riscv_vle64_v_u64m1(b + i, vl);
        vuint64m1_t result_vec;

        montgomery_reduction_rvv_with_transform_helper(&va, &vb, &result_vec, modulo, m_prime, r_square, r_mask, k_bits, vl);

        __riscv_vse64_v_u64m1(result + i, result_vec, vl);
    }
}

static inline void montgomery_reduction_wo_transform_rvv(uint64_t *a,
                                                         uint64_t *b,
                                                         uint64_t *result,
                                                         uint64_t modulo,
                                                         uint64_t m_prime,
                                                         uint64_t r_square,
                                                         uint64_t r_mask,
                                                         uint64_t k_bits,
                                                         size_t len)
{
    size_t vl;
    for (size_t i = 0; i < len; i += vl)
    {
        vl = __riscv_vsetvl_e64m1(len - i);

        vuint64m1_t va = __riscv_vle64_v_u64m1(a + i, vl);
        vuint64m1_t vb = __riscv_vle64_v_u64m1(b + i, vl);
        vuint64m1_t result_vec;

        // Directly use the montgomery reduction without transformation
        vuint64m1_t vab_lo = __riscv_vmul_vv_u64m1(va, vb, vl);
        vuint64m1_t vab_hi = __riscv_vmulhu_vv_u64m1(va, vb, vl);
        montgomery_core_rvv(&vab_lo, &vab_hi, &result_vec, modulo, m_prime, r_mask, k_bits, vl);

        __riscv_vse64_v_u64m1(result + i, result_vec, vl);
    }
}

#endif // MOD_UINTARITH_RVV_1_0_H