// file name = ntt-rvv.h
#ifndef NTT_RVV_H
#define NTT_RVV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Vectorized forward NTT (RVV 0.7.1-style intrinsics)
void ntt_korn_lambiote_vector(
    const uint32_t p,
    uint32_t t,
    uint32_t logt,
    const uint64_t modulus,
    uint64_t *element,
    const uint64_t *rootOfUnityTable,
    const uint64_t *preconRootOfUnityTable
);

// Vectorized inverse NTT
void intt_pease_vector_mulh(
    const uint32_t p,
    const uint64_t modulus,
    uint64_t *element,
    const uint64_t *rootOfUnityInverseTable,
    const uint64_t *preconRootOfUnityInverseTable,
    const uint64_t cycloOrderInv,
    const uint64_t preconCycloOrderInv
);

// Free internal scratch buffers
void free_ntts_mem(void);

#ifdef __cplusplus
}
#endif

#endif // NTT_RVV_H
