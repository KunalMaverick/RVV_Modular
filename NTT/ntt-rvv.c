// file name = ntt-rvv.c
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <riscv_vector.h>
#include "ntt-rvv.h"

typedef unsigned __int128 uint128_t;

static uint8_t initialized_mem = 1;
static uint64_t *aux_increment = NULL;
static uint64_t *auxiliary_array = NULL;

// =========================================================
// SAFE RVV MASKED REDUCTION WRAPPERS (RVV 0.7.1 ABI)
// =========================================================

static inline vuint64m1_t vsub_mod_mask(
    vbool64_t mask, vuint64m1_t v, vuint64m1_t mod, size_t vl)
{
    vuint64m1_t tmp = __riscv_vsub_vv_u64m1(v, mod, vl);
    return __riscv_vmerge_vvm_u64m1(v, tmp, mask, vl);
}

static inline vuint64m1_t vadd_mod_mask(
    vbool64_t mask, vuint64m1_t v, vuint64m1_t mod, size_t vl)
{
    vuint64m1_t tmp = __riscv_vadd_vv_u64m1(v, mod, vl);
    return __riscv_vmerge_vvm_u64m1(v, tmp, mask, vl);
}

// =========================================================
// SCALAR FORWARD NTT (FALLBACK FOR SMALL VLEN)
// =========================================================

static void ntt_scalar(
    const uint32_t p, uint32_t t, uint32_t logt,
    const uint64_t modulus, uint64_t *element,
    const uint64_t *rootOfUnityTable,
    const uint64_t *preconRootOfUnityTable)
{
    for (uint32_t m = 1; m < p; m <<= 1, t >>= 1, --logt) {
        for (uint32_t i = 0; i < m; ++i) {
            uint64_t omega = rootOfUnityTable[i + m];
            uint64_t preconOmega = preconRootOfUnityTable[i + m];

            for (uint32_t j1 = (i << logt), j2 = j1 + t; j1 < j2; ++j1) {
                uint64_t x = element[j1 + t];

                uint64_t q = ((uint128_t)x * preconOmega) >> 64;
                uint64_t tval = x * omega - q * modulus;
                if (tval >= modulus) tval -= modulus;

                uint64_t u = element[j1];

                uint64_t a = u + tval;
                if (a >= modulus) a -= modulus;
                element[j1] = a;

                uint64_t b = u - tval;
                if (b >= modulus) b += modulus;
                element[j1 + t] = b;
            }
        }
    }
}

// =========================================================
// SCALAR INVERSE NTT (FALLBACK FOR SMALL VLEN)
// =========================================================

static void intt_scalar(
    const uint32_t p, const uint64_t modulus, uint64_t *element,
    const uint64_t *rootOfUnityInverseTable,
    const uint64_t *preconRootOfUnityInverseTable,
    const uint64_t cycloOrderInv, const uint64_t preconCycloOrderInv)
{
    for (uint32_t i = 0; i < p; i += 2) {
        uint64_t omega = rootOfUnityInverseTable[(i + p) >> 1];
        uint64_t preconOmega = preconRootOfUnityInverseTable[(i + p) >> 1];
        uint64_t omegaFactor = element[i + 1];
        uint64_t loVal = element[i + 0];

        uint64_t aux2 = loVal + omegaFactor;
        if (aux2 >= modulus) aux2 -= modulus;

        uint64_t q = ((uint128_t)aux2 * preconCycloOrderInv) >> 64;
        uint64_t mult1 = aux2 * cycloOrderInv;
        uint64_t mult2 = q * modulus;
        aux2 = mult1 - mult2;
        if (aux2 >= modulus) aux2 -= modulus;

        element[i + 0] = aux2;

        aux2 = loVal - omegaFactor;
        if (aux2 >= modulus) aux2 += modulus;

        q = ((uint128_t)aux2 * preconOmega) >> 64;
        mult1 = aux2 * omega;
        mult2 = q * modulus;
        aux2 = mult1 - mult2;
        if (aux2 >= modulus) aux2 -= modulus;

        q = ((uint128_t)aux2 * preconCycloOrderInv) >> 64;
        mult1 = aux2 * cycloOrderInv;
        mult2 = q * modulus;
        aux2 = mult1 - mult2;
        if (aux2 >= modulus) aux2 -= modulus;

        element[i + 1] = aux2;
    }

    uint32_t t = 2;
    uint32_t logt = 2;
    for (uint32_t m = p >> 2; m >= 1; m >>= 1, t <<= 1, ++logt) {
        for (uint32_t i = 0; i < m; ++i) {
            uint64_t omega = rootOfUnityInverseTable[i + m];
            uint64_t preconOmega = preconRootOfUnityInverseTable[i + m];

            for (uint32_t j1 = (i << logt), j2 = j1 + t; j1 < j2; ++j1) {
                uint64_t hiVal = element[j1 + t];
                uint64_t loVal = element[j1 + 0];

                uint64_t aux2 = loVal + hiVal;
                if (aux2 >= modulus) aux2 -= modulus;
                element[j1 + 0] = aux2;

                aux2 = loVal - hiVal;
                if (aux2 >= modulus) aux2 += modulus;

                uint64_t q = ((uint128_t)aux2 * preconOmega) >> 64;
                uint64_t mult1 = aux2 * omega;
                uint64_t mult2 = q * modulus;
                aux2 = mult1 - mult2;
                if (aux2 >= modulus) aux2 -= modulus;

                element[j1 + t] = aux2;
            }
        }
    }
}

// =========================================================
// MEMORY CLEANUP
// =========================================================

void free_ntts_mem(void)
{
    if (aux_increment) {
        free(aux_increment);
        aux_increment = NULL;
    }
    if (auxiliary_array) {
        free(auxiliary_array);
        auxiliary_array = NULL;
    }
    initialized_mem = 1;
}

// =========================================================
// FORWARD NTT (VECTOR + SCALAR FALLBACK)
// =========================================================

void ntt_korn_lambiote_vector(
    const uint32_t p,
    uint32_t t,
    uint32_t logt,
    const uint64_t modulus,
    uint64_t *element,
    const uint64_t *rootOfUnityTable,
    const uint64_t *preconRootOfUnityTable)
{
    size_t gvl = __riscv_vsetvl_e64m1(p);

    // Scalar fallback for small VLEN
    if (gvl < 8 || (gvl & (gvl - 1)) != 0) {
        ntt_scalar(p, t, logt, modulus, element, rootOfUnityTable, preconRootOfUnityTable);
        return;
    }

    uint8_t element_or_aux = 0;
    uint32_t stopvalue_sft1 = p >> 1;
    size_t max_gvl = gvl;

    uint64_t *input_stage_array = element;
    uint64_t *output_stage_array = NULL;
    uint64_t *swap_pointer_aux = NULL;

    uint32_t log_of_gvl = 0;
    for (size_t tmp = gvl; tmp > 1; tmp >>= 1) log_of_gvl++;

    vuint64m1_t v_coef_mod = __riscv_vmv_v_x_u64m1(modulus, gvl);

    if (initialized_mem) {
        initialized_mem = 0;
        aux_increment = malloc(gvl * sizeof(uint64_t));
        auxiliary_array = malloc(p * sizeof(uint64_t));
        for (uint32_t i = 0; i < gvl; i++)
            aux_increment[i] = i;
    }

    output_stage_array = auxiliary_array;

    vuint64m1_t v_index_1 = __riscv_vmv_v_x_u64m1(1, gvl);
    vuint64m1_t v_index_original = __riscv_vle64_v_u64m1(aux_increment, gvl);
    vuint64m1_t v_index_stage_aux = __riscv_vmv_v_x_u64m1(1, gvl);
    vuint64m1_t v_m = __riscv_vmv_v_x_u64m1(2, gvl);

    uint32_t m = 1;

    vuint64m1_t v_S = __riscv_vmv_v_x_u64m1(rootOfUnityTable[1], gvl);
    vuint64m1_t v_precomp_aux = __riscv_vmv_v_x_u64m1(preconRootOfUnityTable[1], gvl);

    // ---- STAGE 0: First Loop (Broadcast) ----
    for (uint32_t i = 0; i < p; i += (max_gvl << 1)) {
        vuint64m1_t v_U = __riscv_vle64_v_u64m1(&input_stage_array[i >> 1], max_gvl);
        vuint64m1_t v_V = __riscv_vle64_v_u64m1(&input_stage_array[(i >> 1) + stopvalue_sft1], max_gvl);

        vuint64m1_t v_q = __riscv_vmulhu_vv_u64m1(v_V, v_precomp_aux, max_gvl);
        vuint64m1_t v_mult = __riscv_vmul_vv_u64m1(v_V, v_S, max_gvl);
        vuint64m1_t v_aux = __riscv_vmul_vv_u64m1(v_q, v_coef_mod, max_gvl);
        v_V = __riscv_vsub_vv_u64m1(v_mult, v_aux, max_gvl);

        vbool64_t mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_V, max_gvl);
        v_V = vsub_mod_mask(mask, v_V, v_coef_mod, max_gvl);

        vuint64m1_t v_result_0 = __riscv_vadd_vv_u64m1(v_U, v_V, max_gvl);
        mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_0, max_gvl);
        v_result_0 = vsub_mod_mask(mask, v_result_0, v_coef_mod, max_gvl);

        __riscv_vsse64_v_u64m1(&output_stage_array[i], 16, v_result_0, max_gvl);

        vuint64m1_t v_result_1 = __riscv_vsub_vv_u64m1(v_U, v_V, max_gvl);
        mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_1, max_gvl);
        v_result_1 = vadd_mod_mask(mask, v_result_1, v_coef_mod, max_gvl);

        __riscv_vsse64_v_u64m1(&output_stage_array[i + 1], 16, v_result_1, max_gvl);
    }

    m <<= 1;
    element_or_aux ^= 1;

    vuint64m1_t v_omega_og = __riscv_vle64_v_u64m1(&rootOfUnityTable[0], max_gvl);
    vuint64m1_t v_precomp_og = __riscv_vle64_v_u64m1(&preconRootOfUnityTable[0], max_gvl);

    swap_pointer_aux = output_stage_array;
    output_stage_array = input_stage_array;
    input_stage_array = swap_pointer_aux;

    // ---- INTERMEDIATE LOOPS: vrgather ----
    for (; m < gvl; m <<= 1, element_or_aux ^= 1) {
        vuint64m1_t v_index = __riscv_vand_vv_u64m1(v_index_original, v_index_stage_aux, gvl);
        v_index = __riscv_vadd_vv_u64m1(v_index, v_m, gvl);
        vuint64m1_t v_S = __riscv_vrgather_vv_u64m1(v_omega_og, v_index, gvl);
        vuint64m1_t v_precomp_aux = __riscv_vrgather_vv_u64m1(v_precomp_og, v_index, gvl);

        v_index_stage_aux = __riscv_vsll_vv_u64m1(v_index_stage_aux, v_index_1, gvl);
        v_index_stage_aux = __riscv_vor_vv_u64m1(v_index_stage_aux, v_index_1, gvl);
        v_m = __riscv_vsll_vv_u64m1(v_m, v_index_1, gvl);

        for (uint32_t i = 0; i < p; i += (max_gvl << 1)) {
            vuint64m1_t v_U = __riscv_vle64_v_u64m1(&input_stage_array[i >> 1], max_gvl);
            vuint64m1_t v_V = __riscv_vle64_v_u64m1(&input_stage_array[(i >> 1) + stopvalue_sft1], max_gvl);

            vuint64m1_t v_q = __riscv_vmulhu_vv_u64m1(v_V, v_precomp_aux, max_gvl);
            vuint64m1_t v_mult = __riscv_vmul_vv_u64m1(v_V, v_S, max_gvl);
            vuint64m1_t v_aux = __riscv_vmul_vv_u64m1(v_q, v_coef_mod, max_gvl);
            v_V = __riscv_vsub_vv_u64m1(v_mult, v_aux, max_gvl);

            vbool64_t mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_V, max_gvl);
            v_V = vsub_mod_mask(mask, v_V, v_coef_mod, max_gvl);

            vuint64m1_t v_result_0 = __riscv_vadd_vv_u64m1(v_U, v_V, max_gvl);
            mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_0, max_gvl);
            v_result_0 = vsub_mod_mask(mask, v_result_0, v_coef_mod, max_gvl);

            vuint64m1_t v_result_1 = __riscv_vsub_vv_u64m1(v_U, v_V, max_gvl);
            mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_1, max_gvl);
            v_result_1 = vadd_mod_mask(mask, v_result_1, v_coef_mod, max_gvl);

            __riscv_vsse64_v_u64m1(&output_stage_array[i], 16, v_result_0, max_gvl);
            __riscv_vsse64_v_u64m1(&output_stage_array[i + 1], 16, v_result_1, max_gvl);
        }

        swap_pointer_aux = output_stage_array;
        output_stage_array = input_stage_array;
        input_stage_array = swap_pointer_aux;
    }

    // ---- FINAL LOOPS: vload (with skips) ----
    for (; m < p; m <<= 1, element_or_aux ^= 1) {
        for (uint32_t i = 0; i < (m >> log_of_gvl); i++) {
            uint64_t index = i << log_of_gvl;

            vuint64m1_t v_S = __riscv_vle64_v_u64m1(&rootOfUnityTable[index + m], max_gvl);
            vuint64m1_t v_precomp_aux = __riscv_vle64_v_u64m1(&preconRootOfUnityTable[index + m], max_gvl);

            for (uint32_t j = 0; j < (stopvalue_sft1 / m); j++) {
                vuint64m1_t v_U = __riscv_vle64_v_u64m1(&input_stage_array[index + j * m], max_gvl);
                vuint64m1_t v_V = __riscv_vle64_v_u64m1(&input_stage_array[index + j * m + stopvalue_sft1], max_gvl);

                vuint64m1_t v_q = __riscv_vmulhu_vv_u64m1(v_V, v_precomp_aux, max_gvl);
                vuint64m1_t v_mult = __riscv_vmul_vv_u64m1(v_V, v_S, max_gvl);
                vuint64m1_t v_aux = __riscv_vmul_vv_u64m1(v_q, v_coef_mod, max_gvl);
                v_V = __riscv_vsub_vv_u64m1(v_mult, v_aux, max_gvl);

                vbool64_t mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_V, max_gvl);
                v_V = vsub_mod_mask(mask, v_V, v_coef_mod, max_gvl);

                vuint64m1_t v_result_0 = __riscv_vadd_vv_u64m1(v_U, v_V, max_gvl);
                mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_0, max_gvl);
                v_result_0 = vsub_mod_mask(mask, v_result_0, v_coef_mod, max_gvl);

                vuint64m1_t v_result_1 = __riscv_vsub_vv_u64m1(v_U, v_V, max_gvl);
                mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_1, max_gvl);
                v_result_1 = vadd_mod_mask(mask, v_result_1, v_coef_mod, max_gvl);

                __riscv_vsse64_v_u64m1(&output_stage_array[((index + j * m) << 1)], 16, v_result_0, max_gvl);
                __riscv_vsse64_v_u64m1(&output_stage_array[((index + j * m) << 1) + 1], 16, v_result_1, max_gvl);
            }
        }

        swap_pointer_aux = output_stage_array;
        output_stage_array = input_stage_array;
        input_stage_array = swap_pointer_aux;
    }

    // Copy back if needed
    if (element_or_aux) {
        for (uint32_t i = 0; i < p; i += max_gvl) {
            vuint64m1_t v_auxarr = __riscv_vle64_v_u64m1(&input_stage_array[i], max_gvl);
            __riscv_vse64_v_u64m1(&element[i], v_auxarr, max_gvl);
        }
    }
}

// =========================================================
// INVERSE NTT (VECTOR + SCALAR FALLBACK)
// =========================================================

void intt_pease_vector_mulh(
    const uint32_t p,
    const uint64_t modulus,
    uint64_t *element,
    const uint64_t *rootOfUnityInverseTable,
    const uint64_t *preconRootOfUnityInverseTable,
    const uint64_t cycloOrderInv,
    const uint64_t preconCycloOrderInv)
{
    size_t gvl = __riscv_vsetvl_e64m1(p);

    // Scalar fallback for small VLEN
    if (gvl < 8 || (gvl & (gvl - 1)) != 0) {
        intt_scalar(p, modulus, element, rootOfUnityInverseTable,
                    preconRootOfUnityInverseTable, cycloOrderInv, preconCycloOrderInv);
        return;
    }

    uint8_t element_or_aux = 0;
    uint32_t stopvalue_sft1 = p >> 1;
    size_t max_gvl = gvl;

    uint64_t *input_stage_array = element;
    uint64_t *output_stage_array = NULL;
    uint64_t *swap_pointer_aux = NULL;

    uint32_t log_of_gvl = 0;
    for (size_t tmp = gvl; tmp > 1; tmp >>= 1) log_of_gvl++;

    vuint64m1_t v_coef_mod = __riscv_vmv_v_x_u64m1(modulus, gvl);
    vuint64m1_t v_preconCycloOrderInv = __riscv_vmv_v_x_u64m1(preconCycloOrderInv, gvl);
    vuint64m1_t v_cycloOrderInv = __riscv_vmv_v_x_u64m1(cycloOrderInv, gvl);

    if (initialized_mem) {
        initialized_mem = 0;
        aux_increment = malloc(gvl * sizeof(uint64_t));
        auxiliary_array = malloc(p * sizeof(uint64_t));
        for (uint32_t i = 0; i < gvl; i++)
            aux_increment[i] = i;
    }

    output_stage_array = auxiliary_array;

    vuint64m1_t v_index_1 = __riscv_vmv_v_x_u64m1(1, gvl);
    vuint64m1_t v_index_original = __riscv_vle64_v_u64m1(aux_increment, gvl);
    vuint64m1_t v_index_stage_aux = __riscv_vmv_v_x_u64m1((gvl >> 1) - 1, gvl);
    vuint64m1_t v_m = __riscv_vmv_v_x_u64m1(gvl >> 1, gvl);

    uint32_t m = p >> 1;

    // ---- FIRST STAGE ----
    for (uint32_t i = 0; i < (m >> log_of_gvl); i++) {
        uint64_t index = i << log_of_gvl;

        vuint64m1_t v_S = __riscv_vle64_v_u64m1(&rootOfUnityInverseTable[index + m], max_gvl);
        vuint64m1_t v_precomp_aux = __riscv_vle64_v_u64m1(&preconRootOfUnityInverseTable[index + m], max_gvl);

        vuint64m1_t v_U = __riscv_vlse64_v_u64m1(&input_stage_array[(index << 1)], 16, max_gvl);
        vuint64m1_t v_V = __riscv_vlse64_v_u64m1(&input_stage_array[(index << 1) + 1], 16, max_gvl);

        vuint64m1_t v_result_0 = __riscv_vadd_vv_u64m1(v_U, v_V, max_gvl);
        vbool64_t mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_0, max_gvl);
        v_result_0 = vsub_mod_mask(mask, v_result_0, v_coef_mod, max_gvl);

        vuint64m1_t v_q = __riscv_vmulhu_vv_u64m1(v_result_0, v_preconCycloOrderInv, max_gvl);
        vuint64m1_t v_mult = __riscv_vmul_vv_u64m1(v_result_0, v_cycloOrderInv, max_gvl);
        vuint64m1_t v_aux = __riscv_vmul_vv_u64m1(v_q, v_coef_mod, max_gvl);
        v_result_0 = __riscv_vsub_vv_u64m1(v_mult, v_aux, max_gvl);
        mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_0, max_gvl);
        v_result_0 = vsub_mod_mask(mask, v_result_0, v_coef_mod, max_gvl);

        vuint64m1_t v_result_1 = __riscv_vsub_vv_u64m1(v_U, v_V, max_gvl);
        mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_1, max_gvl);
        v_result_1 = vadd_mod_mask(mask, v_result_1, v_coef_mod, max_gvl);

        v_q = __riscv_vmulhu_vv_u64m1(v_result_1, v_precomp_aux, max_gvl);
        v_mult = __riscv_vmul_vv_u64m1(v_result_1, v_S, max_gvl);
        v_aux = __riscv_vmul_vv_u64m1(v_q, v_coef_mod, max_gvl);
        v_result_1 = __riscv_vsub_vv_u64m1(v_mult, v_aux, max_gvl);
        mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_1, max_gvl);
        v_result_1 = vsub_mod_mask(mask, v_result_1, v_coef_mod, max_gvl);

        v_q = __riscv_vmulhu_vv_u64m1(v_result_1, v_preconCycloOrderInv, max_gvl);
        v_mult = __riscv_vmul_vv_u64m1(v_result_1, v_cycloOrderInv, max_gvl);
        v_aux = __riscv_vmul_vv_u64m1(v_q, v_coef_mod, max_gvl);
        v_result_1 = __riscv_vsub_vv_u64m1(v_mult, v_aux, max_gvl);
        mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_1, max_gvl);
        v_result_1 = vsub_mod_mask(mask, v_result_1, v_coef_mod, max_gvl);

        __riscv_vse64_v_u64m1(&output_stage_array[index], v_result_0, max_gvl);
        __riscv_vse64_v_u64m1(&output_stage_array[index + stopvalue_sft1], v_result_1, max_gvl);
    }

    swap_pointer_aux = output_stage_array;
    output_stage_array = input_stage_array;
    input_stage_array = swap_pointer_aux;
    m >>= 1;
    element_or_aux ^= 1;

    // ---- INTERMEDIATE STAGES (m >= gvl) ----
    for (; m >= gvl; m >>= 1, element_or_aux ^= 1) {
        for (uint32_t i = 0; i < (m >> log_of_gvl); i++) {
            uint64_t index = i << log_of_gvl;

            vuint64m1_t v_S = __riscv_vle64_v_u64m1(&rootOfUnityInverseTable[index + m], max_gvl);
            vuint64m1_t v_precomp_aux = __riscv_vle64_v_u64m1(&preconRootOfUnityInverseTable[index + m], max_gvl);

            for (uint32_t j = 0; j < (stopvalue_sft1 / m); j += 2) {
                vuint64m1_t v_U = __riscv_vlse64_v_u64m1(&input_stage_array[((index + j * m) << 1)], 16, max_gvl);
                vuint64m1_t v_V = __riscv_vlse64_v_u64m1(&input_stage_array[((index + j * m) << 1) + 1], 16, max_gvl);
                vuint64m1_t _v_U = __riscv_vlse64_v_u64m1(&input_stage_array[((index + (j + 1) * m) << 1)], 16, max_gvl);
                vuint64m1_t _v_V = __riscv_vlse64_v_u64m1(&input_stage_array[((index + (j + 1) * m) << 1) + 1], 16, max_gvl);

                vuint64m1_t v_result_0 = __riscv_vadd_vv_u64m1(v_U, v_V, max_gvl);
                vuint64m1_t _v_result_0 = __riscv_vadd_vv_u64m1(_v_U, _v_V, max_gvl);

                vbool64_t mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_0, max_gvl);
                v_result_0 = vsub_mod_mask(mask, v_result_0, v_coef_mod, max_gvl);
                vbool64_t _mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, _v_result_0, max_gvl);
                _v_result_0 = vsub_mod_mask(_mask, _v_result_0, v_coef_mod, max_gvl);

                vuint64m1_t v_result_1 = __riscv_vsub_vv_u64m1(v_U, v_V, max_gvl);
                mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_1, max_gvl);
                v_result_1 = vadd_mod_mask(mask, v_result_1, v_coef_mod, max_gvl);

                vuint64m1_t _v_result_1 = __riscv_vsub_vv_u64m1(_v_U, _v_V, max_gvl);
                _mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, _v_result_1, max_gvl);
                _v_result_1 = vadd_mod_mask(_mask, _v_result_1, v_coef_mod, max_gvl);

                vuint64m1_t v_q = __riscv_vmulhu_vv_u64m1(v_result_1, v_precomp_aux, max_gvl);
                vuint64m1_t _v_q = __riscv_vmulhu_vv_u64m1(_v_result_1, v_precomp_aux, max_gvl);

                vuint64m1_t v_mult = __riscv_vmul_vv_u64m1(v_result_1, v_S, max_gvl);
                vuint64m1_t v_aux = __riscv_vmul_vv_u64m1(v_q, v_coef_mod, max_gvl);
                v_result_1 = __riscv_vsub_vv_u64m1(v_mult, v_aux, max_gvl);
                mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_1, max_gvl);
                v_result_1 = vsub_mod_mask(mask, v_result_1, v_coef_mod, max_gvl);

                vuint64m1_t _v_mult = __riscv_vmul_vv_u64m1(_v_result_1, v_S, max_gvl);
                vuint64m1_t _v_aux = __riscv_vmul_vv_u64m1(_v_q, v_coef_mod, max_gvl);
                _v_result_1 = __riscv_vsub_vv_u64m1(_v_mult, _v_aux, max_gvl);
                _mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, _v_result_1, max_gvl);
                _v_result_1 = vsub_mod_mask(_mask, _v_result_1, v_coef_mod, max_gvl);

                __riscv_vse64_v_u64m1(&output_stage_array[index + j * m], v_result_0, max_gvl);
                __riscv_vse64_v_u64m1(&output_stage_array[index + j * m + stopvalue_sft1], v_result_1, max_gvl);
                __riscv_vse64_v_u64m1(&output_stage_array[index + (j + 1) * m], _v_result_0, max_gvl);
                __riscv_vse64_v_u64m1(&output_stage_array[index + (j + 1) * m + stopvalue_sft1], _v_result_1, max_gvl);
            }
        }

        swap_pointer_aux = output_stage_array;
        output_stage_array = input_stage_array;
        input_stage_array = swap_pointer_aux;
    }

    vuint64m1_t v_omega_og = __riscv_vle64_v_u64m1(&rootOfUnityInverseTable[0], max_gvl);
    vuint64m1_t v_precomp_og = __riscv_vle64_v_u64m1(&preconRootOfUnityInverseTable[0], max_gvl);

    // ---- STAGES WITH VRGATHER (m >= 2) ----
    for (; m >= 2; m >>= 1, element_or_aux ^= 1) {
        vuint64m1_t v_index = __riscv_vand_vv_u64m1(v_index_original, v_index_stage_aux, gvl);
        v_index = __riscv_vadd_vv_u64m1(v_index, v_m, gvl);
        vuint64m1_t v_S = __riscv_vrgather_vv_u64m1(v_omega_og, v_index, gvl);
        vuint64m1_t v_precomp_aux = __riscv_vrgather_vv_u64m1(v_precomp_og, v_index, gvl);

        v_index_stage_aux = __riscv_vsrl_vv_u64m1(v_index_stage_aux, v_index_1, gvl);
        v_m = __riscv_vsrl_vv_u64m1(v_m, v_index_1, gvl);

        for (uint32_t i = 0; i < p; i += (max_gvl << 2)) {
            uint32_t ii = i + (max_gvl << 1);

            vuint64m1_t v_U = __riscv_vlse64_v_u64m1(&input_stage_array[i], 16, max_gvl);
            vuint64m1_t v_V = __riscv_vlse64_v_u64m1(&input_stage_array[i + 1], 16, max_gvl);
            vuint64m1_t _v_U = __riscv_vlse64_v_u64m1(&input_stage_array[ii], 16, max_gvl);
            vuint64m1_t _v_V = __riscv_vlse64_v_u64m1(&input_stage_array[ii + 1], 16, max_gvl);

            vuint64m1_t v_result_0 = __riscv_vadd_vv_u64m1(v_U, v_V, max_gvl);
            vuint64m1_t _v_result_0 = __riscv_vadd_vv_u64m1(_v_U, _v_V, max_gvl);

            vbool64_t mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_0, max_gvl);
            v_result_0 = vsub_mod_mask(mask, v_result_0, v_coef_mod, max_gvl);
            vbool64_t _mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, _v_result_0, max_gvl);
            _v_result_0 = vsub_mod_mask(_mask, _v_result_0, v_coef_mod, max_gvl);

            vuint64m1_t v_result_1 = __riscv_vsub_vv_u64m1(v_U, v_V, max_gvl);
            mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_1, max_gvl);
            v_result_1 = vadd_mod_mask(mask, v_result_1, v_coef_mod, max_gvl);

            vuint64m1_t _v_result_1 = __riscv_vsub_vv_u64m1(_v_U, _v_V, max_gvl);
            _mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, _v_result_1, max_gvl);
            _v_result_1 = vadd_mod_mask(_mask, _v_result_1, v_coef_mod, max_gvl);

            vuint64m1_t v_q = __riscv_vmulhu_vv_u64m1(v_result_1, v_precomp_aux, max_gvl);
            vuint64m1_t _v_q = __riscv_vmulhu_vv_u64m1(_v_result_1, v_precomp_aux, max_gvl);

            vuint64m1_t v_mult = __riscv_vmul_vv_u64m1(v_result_1, v_S, max_gvl);
            vuint64m1_t v_aux = __riscv_vmul_vv_u64m1(v_q, v_coef_mod, max_gvl);
            v_result_1 = __riscv_vsub_vv_u64m1(v_mult, v_aux, max_gvl);
            mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_1, max_gvl);
            v_result_1 = vsub_mod_mask(mask, v_result_1, v_coef_mod, max_gvl);

            vuint64m1_t _v_mult = __riscv_vmul_vv_u64m1(_v_result_1, v_S, max_gvl);
            vuint64m1_t _v_aux = __riscv_vmul_vv_u64m1(_v_q, v_coef_mod, max_gvl);
            _v_result_1 = __riscv_vsub_vv_u64m1(_v_mult, _v_aux, max_gvl);
            _mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, _v_result_1, max_gvl);
            _v_result_1 = vsub_mod_mask(_mask, _v_result_1, v_coef_mod, max_gvl);

            __riscv_vse64_v_u64m1(&output_stage_array[i >> 1], v_result_0, max_gvl);
            __riscv_vse64_v_u64m1(&output_stage_array[(i >> 1) + stopvalue_sft1], v_result_1, max_gvl);
            __riscv_vse64_v_u64m1(&output_stage_array[ii >> 1], _v_result_0, max_gvl);
            __riscv_vse64_v_u64m1(&output_stage_array[(ii >> 1) + stopvalue_sft1], _v_result_1, max_gvl);
        }

        swap_pointer_aux = output_stage_array;
        output_stage_array = input_stage_array;
        input_stage_array = swap_pointer_aux;
    }

    // ---- FINAL STAGE (m == 1 if applicable) ----
    if (m != 0) {
        v_omega_og = __riscv_vmv_v_x_u64m1(rootOfUnityInverseTable[1], max_gvl);
        v_precomp_og = __riscv_vmv_v_x_u64m1(preconRootOfUnityInverseTable[1], max_gvl);

        for (uint32_t i = 0; i < p; i += (max_gvl << 2)) {
            uint32_t ii = i + (max_gvl << 1);

            vuint64m1_t v_U = __riscv_vlse64_v_u64m1(&input_stage_array[i], 16, max_gvl);
            vuint64m1_t v_V = __riscv_vlse64_v_u64m1(&input_stage_array[i + 1], 16, max_gvl);
            vuint64m1_t _v_U = __riscv_vlse64_v_u64m1(&input_stage_array[ii], 16, max_gvl);
            vuint64m1_t _v_V = __riscv_vlse64_v_u64m1(&input_stage_array[ii + 1], 16, max_gvl);

            vuint64m1_t v_result_0 = __riscv_vadd_vv_u64m1(v_U, v_V, max_gvl);
            vuint64m1_t _v_result_0 = __riscv_vadd_vv_u64m1(_v_U, _v_V, max_gvl);

            vbool64_t mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_0, max_gvl);
            v_result_0 = vsub_mod_mask(mask, v_result_0, v_coef_mod, max_gvl);
            vbool64_t _mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, _v_result_0, max_gvl);
            _v_result_0 = vsub_mod_mask(_mask, _v_result_0, v_coef_mod, max_gvl);

            vuint64m1_t v_result_1 = __riscv_vsub_vv_u64m1(v_U, v_V, max_gvl);
            mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_1, max_gvl);
            v_result_1 = vadd_mod_mask(mask, v_result_1, v_coef_mod, max_gvl);

            vuint64m1_t _v_result_1 = __riscv_vsub_vv_u64m1(_v_U, _v_V, max_gvl);
            _mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, _v_result_1, max_gvl);
            _v_result_1 = vadd_mod_mask(_mask, _v_result_1, v_coef_mod, max_gvl);

            vuint64m1_t v_q = __riscv_vmulhu_vv_u64m1(v_result_1, v_precomp_og, max_gvl);
            vuint64m1_t _v_q = __riscv_vmulhu_vv_u64m1(_v_result_1, v_precomp_og, max_gvl);

            vuint64m1_t v_mult = __riscv_vmul_vv_u64m1(v_result_1, v_omega_og, max_gvl);
            vuint64m1_t v_aux = __riscv_vmul_vv_u64m1(v_q, v_coef_mod, max_gvl);
            v_result_1 = __riscv_vsub_vv_u64m1(v_mult, v_aux, max_gvl);
            mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, v_result_1, max_gvl);
            v_result_1 = vsub_mod_mask(mask, v_result_1, v_coef_mod, max_gvl);

            vuint64m1_t _v_mult = __riscv_vmul_vv_u64m1(_v_result_1, v_omega_og, max_gvl);
            vuint64m1_t _v_aux = __riscv_vmul_vv_u64m1(_v_q, v_coef_mod, max_gvl);
            _v_result_1 = __riscv_vsub_vv_u64m1(_v_mult, _v_aux, max_gvl);
            _mask = __riscv_vmsleu_vv_u64m1_b64(v_coef_mod, _v_result_1, max_gvl);
            _v_result_1 = vsub_mod_mask(_mask, _v_result_1, v_coef_mod, max_gvl);

            __riscv_vse64_v_u64m1(&output_stage_array[i >> 1], v_result_0, max_gvl);
            __riscv_vse64_v_u64m1(&output_stage_array[(i >> 1) + stopvalue_sft1], v_result_1, max_gvl);
            __riscv_vse64_v_u64m1(&output_stage_array[ii >> 1], _v_result_0, max_gvl);
            __riscv_vse64_v_u64m1(&output_stage_array[(ii >> 1) + stopvalue_sft1], _v_result_1, max_gvl);
        }

        element_or_aux ^= 1;
    }

    // Copy back if needed
    if (element_or_aux) {
        for (uint32_t i = 0; i < p; i += max_gvl) {
            vuint64m1_t v_auxarr = __riscv_vle64_v_u64m1(&output_stage_array[i], max_gvl);
            __riscv_vse64_v_u64m1(&element[i], v_auxarr, max_gvl);
        }
    }
}