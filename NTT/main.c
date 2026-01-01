#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "ntt-rvv.h"

#define N   8
#define MOD 17

// forward roots (ω = 9)
static const uint64_t root[N] = {
    1, 9, 13, 15, 16, 8, 4, 2
};

// inverse roots (ω⁻¹ = 2)
static const uint64_t root_inv[N] = {
    1, 2, 4, 8, 16, 15, 13, 9
};

// Barrett precondition
static inline uint64_t precon(uint64_t c, uint64_t mod) {
    __uint128_t x = (__uint128_t)c << 64;
    return (uint64_t)(x / mod);
}

uint64_t precon_root[N];
uint64_t precon_root_inv[N];

static inline uint64_t mod_add(uint64_t a, uint64_t b) {
    a += b;
    return (a >= MOD) ? a - MOD : a;
}

static inline uint64_t mod_sub(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (a + MOD - b);
}

static inline uint64_t mod_mul(uint64_t a, uint64_t b) {
    return (a * b) % MOD;
}

/* ---------------- SCALAR REFERENCE NTT ---------------- */

void ntt_scalar(uint64_t *a) {
    for (int m = 1; m < N; m <<= 1) {
        int step = N / (m << 1);
        for (int i = 0; i < N; i += (m << 1)) {
            for (int j = 0; j < m; j++) {
                uint64_t u = a[i + j];
                uint64_t v = mod_mul(a[i + j + m], root[j * step]);
                a[i + j]     = mod_add(u, v);
                a[i + j + m] = mod_sub(u, v);
            }
        }
    }
}

void intt_scalar(uint64_t *a) {
    for (int m = N >> 1; m >= 1; m >>= 1) {
        int step = N / (m << 1);
        for (int i = 0; i < N; i += (m << 1)) {
            for (int j = 0; j < m; j++) {
                uint64_t u = a[i + j];
                uint64_t v = a[i + j + m];
                a[i + j]     = mod_add(u, v);
                a[i + j + m] = mod_mul(mod_sub(u, v), root_inv[j * step]);
            }
        }
    }

    uint64_t N_inv = 15; // 8⁻¹ mod 17
    for (int i = 0; i < N; i++)
        a[i] = mod_mul(a[i], N_inv);
}

/* ---------------- UTILS ---------------- */

static inline void dump(const char *tag, uint64_t *a) {
    printf("%-22s", tag);
    for (int i = 0; i < N; i++)
        printf("%2lu ", a[i]);
    printf("\n");
}

/* ---------------- MAIN ---------------- */

int main(void)
{
    uint64_t a[N]  = {1,2,3,4,5,6,7,8};
    uint64_t s[N], s2[N];
    uint64_t v[N], v2[N];

    memcpy(s, a, sizeof(a));
    memcpy(v, a, sizeof(a));

    for (int i = 0; i < N; i++) {
        precon_root[i]     = precon(root[i], MOD);
        precon_root_inv[i] = precon(root_inv[i], MOD);
    }

    uint64_t N_inv        = 15;
    uint64_t precon_N_inv = precon(N_inv, MOD);

    /* -------- SCALAR -------- */

    printf("\n==== SCALAR NTT ====\n");
    dump("INPUT a:", s);
    ntt_scalar(s);
    dump("NTT(a):", s);

    memcpy(s2, s, sizeof(s));
    intt_scalar(s2);
    dump("INTT(NTT(a)):", s2);

    /* -------- RVV -------- */

    printf("\n==== RVV NTT ====\n");
    dump("INPUT a:", v);

    ntt_korn_lambiote_vector(
        N, N >> 1, 3, MOD, v, root, precon_root
    );

    dump("NTT(a):", v);

    memcpy(v2, v, sizeof(v));

    intt_pease_vector_mulh(
        N, MOD, v2,
        root_inv, precon_root_inv,
        N_inv, precon_N_inv
    );

    dump("INTT(NTT(a)):", v2);

    /* -------- CHECK -------- */

    int ok = 1;
    for (int i = 0; i < N; i++)
        if (a[i] != s2[i] || a[i] != v2[i])
            ok = 0;

    printf("\nVerification: %s\n", ok ? "PASS" : "FAIL");

    free_ntts_mem();
    return 0;
}
