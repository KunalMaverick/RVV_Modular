#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "util_modular.h"

// ================================================================
//  GOLDEN SCALAR MODULAR FUNCTIONS
// ================================================================
void add_vv_mod_u64_golden(size_t n,const uint64_t *a,const uint64_t *b,uint64_t m,uint64_t *c){
    for(size_t i=0;i<n;i++){ uint64_t s=a[i]+b[i]; c[i]=(s>=m)?(s-m):s; }
}

void add_vx_mod_u64_golden(size_t n,const uint64_t *a,uint64_t b,uint64_t m,uint64_t *c){
    for(size_t i=0;i<n;i++){ uint64_t s=a[i]+b; c[i]=(s>=m)?(s-m):s; }
}

void sub_vv_mod_u64_golden(size_t n,const uint64_t *a,const uint64_t *b,uint64_t m,uint64_t *c){
    for(size_t i=0;i<n;i++){ uint64_t d=a[i]-b[i]; c[i]=(a[i]<b[i])?(d+m):d; }
}

void sub_vx_mod_u64_golden(size_t n,const uint64_t *a,uint64_t b,uint64_t m,uint64_t *c){
    for(size_t i=0;i<n;i++){ uint64_t d=a[i]-b; c[i]=(a[i]<b)?(d+m):d; }
}

void neg_v_mod_u64_golden(size_t n,const uint64_t *a,uint64_t m,uint64_t *c){
    for(size_t i=0;i<n;i++){ c[i]=(a[i]==0)?0:(m-a[i]); }
}

// ================================================================
// HELPERS
// ================================================================
double tsec(){ return (double)clock()/CLOCKS_PER_SEC; }

void fill_rand(uint64_t *a,size_t n,uint64_t m){
    for(size_t i=0;i<n;i++)
        a[i]=(((uint64_t)rand()<<32)|rand())%m;
}

void compute_mu(uint64_t m,uint64_t mu[2]){
    __uint128_t x=~(__uint128_t)0;
    __uint128_t q=x/m;
    mu[0]=(uint64_t)q;
    mu[1]=(uint64_t)(q>>64);
}

// ================================================================
// SCALAR MATMUL (GOLDEN)
// ================================================================
static void matmul_u64_scalar(const uint64_t *A,
                              const uint64_t *B,
                              uint64_t *C,
                              size_t dim)
{
    for (size_t i = 0; i < dim; i++)
        for (size_t j = 0; j < dim; j++) {
            uint64_t sum = 0;
            for (size_t k = 0; k < dim; k++)
                sum += A[i*dim + k] * B[k*dim + j];
            C[i*dim + j] = sum;
        }
}

// ================================================================
// MATRIX TRANSPOSE
// ================================================================
static void transpose_matrix_u64(const uint64_t *B,
                                 uint64_t *B_t,
                                 size_t dim)
{
    for(size_t i=0;i<dim;i++)
        for(size_t j=0;j<dim;j++)
            B_t[j*dim + i] = B[i*dim + j];
}

// ================================================================
// MATRIX COMPARISON
// ================================================================
static int compare_matrix_u64(const uint64_t *X,
                              const uint64_t *Y,
                              size_t dim)
{
    for (size_t i = 0; i < dim*dim; i++)
        if (X[i] != Y[i])
            return 0;
    return 1;
}

// ================================================================
//  TEST VECTOR–VECTOR MODULAR FUNCTIONS (add VV / sub VV)
// ================================================================
void test_vv(const char *name,
             void (*g)(size_t,const uint64_t*,const uint64_t*,uint64_t,uint64_t*),
             void (*r)(size_t,const uint64_t*,const uint64_t*,uint64_t,uint64_t*),
             size_t n,uint64_t m,int it)
{
    uint64_t *a  = allocate_tensor_1d_uint64(n);
    uint64_t *b  = allocate_tensor_1d_uint64(n);
    uint64_t *cg = allocate_tensor_1d_uint64(n);
    uint64_t *cr = allocate_tensor_1d_uint64(n);

    fill_rand(a,n,m); 
    fill_rand(b,n,m);

    // One debug run
    g(n,a,b,m,cg);
    r(n,a,b,m,cr);

    printf("\n==== DEBUG %s ====\n", name);
    printf(" idx |     a      b     | golden   rvv\n");
    printf("--------------------------------------------\n");
    for(size_t i=0;i<8;i++)
        printf("%4zu | %6lu  %6lu | %6lu  %6lu\n",
               i, a[i], b[i], cg[i], cr[i]);

    // Timings
    double t0=tsec();
    for(int i=0;i<it;i++) g(n,a,b,m,cg);
    double ts=(tsec()-t0)*1e3/it;

    t0=tsec();
    for(int i=0;i<it;i++) r(n,a,b,m,cr);
    double tv=(tsec()-t0)*1e3/it;

    int ok = 1;
    for(size_t i=0;i<n;i++)
        if(cg[i]!=cr[i]) { ok = 0; break; }

    printf("%s: %.3f ms vs %.3f ms | %.2fx | %s\n",
            name, ts, tv, ts/tv, ok ? "PASS" : "FAIL");

    free(a); free(b); free(cg); free(cr);
}

// ================================================================
//  TEST VECTOR–SCALAR MODULAR FUNCTIONS (add VX / sub VX)
// ================================================================
void test_vx(const char *name,
             void (*g)(size_t,const uint64_t*,uint64_t,uint64_t,uint64_t*),
             void (*r)(size_t,const uint64_t*,uint64_t,uint64_t,uint64_t*),
             size_t n,uint64_t m,int it)
{
    uint64_t *a  = allocate_tensor_1d_uint64(n);
    uint64_t *cg = allocate_tensor_1d_uint64(n);
    uint64_t *cr = allocate_tensor_1d_uint64(n);

    fill_rand(a,n,m);

    uint64_t b = rand() % m;

    // One debug run
    g(n,a,b,m,cg);
    r(n,a,b,m,cr);

    printf("\n==== DEBUG %s ====\n", name);
    printf("scalar b = %lu\n", b);
    printf(" idx |     a     | golden   rvv\n");
    printf("----------------------------------\n");
    for(size_t i=0;i<8;i++)
        printf("%4zu | %6lu | %6lu  %6lu\n",
               i, a[i], cg[i], cr[i]);

    // Timings
    double t0=tsec();
    for(int i=0;i<it;i++) g(n,a,b,m,cg);
    double ts=(tsec()-t0)*1e3/it;

    t0=tsec();
    for(int i=0;i<it;i++) r(n,a,b,m,cr);
    double tv=(tsec()-t0)*1e3/it;

    int ok = 1;
    for(size_t i=0;i<n;i++)
        if(cg[i]!=cr[i]) { ok = 0; break; }

    printf("%s: %.3f ms vs %.3f ms | %.2fx | %s\n",
            name, ts, tv, ts/tv, ok ? "PASS" : "FAIL");

    free(a); free(cg); free(cr);
}

// ================================================================
//  TEST UNARY VECTOR MODULAR FUNCTIONS (neg)
// ================================================================
void test_unary(const char *name,
                void (*g)(size_t,const uint64_t*,uint64_t,uint64_t*),
                void (*r)(size_t,const uint64_t*,uint64_t,uint64_t*),
                size_t n,uint64_t m,int it)
{
    uint64_t *a  = allocate_tensor_1d_uint64(n);
    uint64_t *cg = allocate_tensor_1d_uint64(n);
    uint64_t *cr = allocate_tensor_1d_uint64(n);

    fill_rand(a,n,m);

    g(n,a,m,cg);
    r(n,a,m,cr);

    printf("\n==== DEBUG %s ====\n", name);
    printf(" idx |     a     | golden   rvv\n");
    printf("----------------------------------\n");
    for(size_t i=0;i<8;i++)
        printf("%4zu | %6lu | %6lu  %6lu\n",
               i, a[i], cg[i], cr[i]);

    // timing
    double t0=tsec();
    for(int i=0;i<it;i++) g(n,a,m,cg);
    double ts=(tsec()-t0)*1e3/it;

    t0=tsec();
    for(int i=0;i<it;i++) r(n,a,m,cr);
    double tv=(tsec()-t0)*1e3/it;

    int ok = 1;
    for(size_t i=0;i<n;i++)
        if(cg[i]!=cr[i]) { ok = 0; break; }

    printf("%s: %.3f ms vs %.3f ms | %.2fx | %s\n",
            name, ts, tv, ts/tv, ok ? "PASS" : "FAIL");

    free(a); free(cg); free(cr);
}
void test_mul_vv_only(const char *name,
                      size_t n, uint64_t m, int it,
                      const uint64_t *mu)
{
    uint64_t *a  = allocate_tensor_1d_uint64(n);
    uint64_t *b  = allocate_tensor_1d_uint64(n);
    uint64_t *lo = allocate_tensor_1d_uint64(n);
    uint64_t *hi = allocate_tensor_1d_uint64(n);
    uint64_t *c  = allocate_tensor_1d_uint64(n);

    fill_rand(a, n, m);
    fill_rand(b, n, m);

    // Single debug run
    mul_vv_mod_u64(n, a, b, m, mu, lo, hi, c);

    printf("\n==== DEBUG %s ====\n", name);
    printf(" idx |     a       b     |        lo         hi      |   c (mod)\n");
    printf("--------------------------------------------------------------------------\n");

    for (size_t i = 0; i < 8; i++)
        printf("%4zu | %6lu  %6lu | %12lu  %10lu | %6lu\n",
               i, a[i], b[i], lo[i], hi[i], c[i]);

    // Timing
    double t0 = tsec();
    for (int i = 0; i < it; i++)
        mul_vv_mod_u64(n, a, b, m, mu, lo, hi, c);
    double tv = (tsec() - t0) * 1e3;

    printf("%s Time: %.3f ms\n", name, tv);

    free(a); free(b); free(lo); free(hi); free(c);
}


void test_mul_vx_only(const char *name,
                      size_t n, uint64_t m, int it,
                      const uint64_t *mu)
{
    uint64_t *a  = allocate_tensor_1d_uint64(n);
    uint64_t *lo = allocate_tensor_1d_uint64(n);
    uint64_t *hi = allocate_tensor_1d_uint64(n);
    uint64_t *c  = allocate_tensor_1d_uint64(n);

    fill_rand(a, n, m);

    uint64_t b = rand() % m;

    // Single debug run
    mul_vx_mod_u64(n, a, b, m, mu, lo, hi, c);

    printf("\n==== DEBUG %s ====\n", name);
    printf("scalar b = %lu\n", b);
    printf(" idx |      a       |        lo         hi      |   c (mod)\n");
    printf("---------------------------------------------------------------------\n");

    for (size_t i = 0; i < 8; i++)
        printf("%4zu | %6lu | %12lu  %10lu | %6lu\n",
               i, a[i], lo[i], hi[i], c[i]);

    // Timing
    double t0 = tsec();
    for (int i = 0; i < it; i++)
        mul_vx_mod_u64(n, a, b, m, mu, lo, hi, c);
    double tv = (tsec() - t0) * 1e3;

    printf("%s Time: %.3f ms\n", name, tv);

    free(a); free(lo); free(hi); free(c);
}



// ================================================================
// MATMUL BENCHMARK
// ================================================================
static void test_matmul(size_t dim)
{
    printf("\n═══════════════════════════════════════════════\n");
    printf("  MATMUL (uint64) BENCHMARK (%zu x %zu)\n", dim, dim);
    printf("═══════════════════════════════════════════════\n");

    size_t total = dim * dim;

    uint64_t *A   = allocate_tensor_1d_uint64(total);
    uint64_t *B   = allocate_tensor_1d_uint64(total);
    uint64_t *B_t = allocate_tensor_1d_uint64(total);
    uint64_t *C_g = allocate_tensor_1d_uint64(total);
    uint64_t *C_v = allocate_tensor_1d_uint64(total);
    uint64_t *C_stride = allocate_tensor_1d_uint64(total);

    fill_matrix_rand_u64(A, dim);
    fill_matrix_rand_u64(B, dim);

    transpose_matrix_u64(B, B_t, dim);

    double t0 = tsec();
    matmul_u64_scalar(A, B, C_g, dim);
    double t_scalar = (tsec() - t0) * 1e3;

    t0 = tsec();
    matmul_u64_rvv(A, B_t, C_v, dim);
    double t_rvv = (tsec() - t0) * 1e3;

    t0 = tsec();
    matmul_u64_256x256((const uint64_t (*)[256])A,
                       (const uint64_t (*)[256])B,
                       (uint64_t (*)[256])C_stride);
    double t_stride = (tsec() - t0) * 1e3;

    int ok1 = compare_matrix_u64(C_g, C_v, dim);
    int ok2 = compare_matrix_u64(C_g, C_stride, dim);

    printf("Scalar     : %.3f ms\n", t_scalar);
    printf("RVV (B^T)  : %.3f ms  | %s\n", t_rvv, ok1?"PASS":"FAIL");
    printf("RVV stride : %.3f ms  | %s\n", t_stride, ok2?"PASS":"FAIL");

    printf("Speedups → RVV(B^T): %.2fx   RVV(stride): %.2fx\n",
           t_scalar/t_rvv, t_scalar/t_stride);

    free(A); free(B); free(B_t); free(C_g); free(C_v); free(C_stride);
}

// ================================================================
// RVV MODULAR MATMUL ONLY
// ================================================================
static void test_mod_matmul_rvv_only(
    size_t dim,
    uint64_t modulus,
    const uint64_t *mu
)
{
    printf("\n═══════════════════════════════════════════════\n");
    printf("  RVV MODULAR MATMUL ONLY (%zu x %zu)\n", dim, dim);
    printf("═══════════════════════════════════════════════\n");

    size_t total = dim * dim;

    uint64_t *A      = allocate_tensor_1d_uint64(total);
    uint64_t *B      = allocate_tensor_1d_uint64(total);
    uint64_t *B_t    = allocate_tensor_1d_uint64(total);
    uint64_t *C_rvv  = allocate_tensor_1d_uint64(total);

    uint64_t *tmp_lo  = allocate_tensor_1d_uint64(dim);
    uint64_t *tmp_hi  = allocate_tensor_1d_uint64(dim);
    uint64_t *tmp_out = allocate_tensor_1d_uint64(dim);

    for (size_t i = 0; i < total; i++) {
        A[i] = (((uint64_t)rand() << 32) | rand()) % modulus;
        B[i] = (((uint64_t)rand() << 32) | rand()) % modulus;
    }

    transpose_matrix_u64(B, B_t, dim);

    double t0 = tsec();
    matmul_mod_u64_rvv(A, B_t, C_rvv, dim, modulus, mu, tmp_lo, tmp_hi, tmp_out);

    // matmul_mod_u64_rvv_inspect(
    // A,
    // B_t,
    // C_rvv,
    // dim,
    // modulus,
    // mu,
    // tmp_lo,
    // tmp_hi,
    // tmp_out,
    // /* inspect_i */ 0,
    // /* inspect_j */ 0,
    // /* max_print_k */ 16
    // );    

double t_rvv = (tsec() - t0) * 1e3;

    printf("RVV ModMatmul Time: %.3f ms\n", t_rvv);

    printf("\n---  (Row 0, first 8) ---\n");
    for (int x = 0; x < 8; x++) printf("%lu ", C_rvv[x]);
    printf("\n");

    printf("\n---  (Row 1, first 8) ---\n");
    for (int x = 0; x < 8; x++) printf("%lu ", C_rvv[dim + x]);
    printf("\n");

    printf("\n---  (Center values) ---\n");
    size_t mid = (dim/2) * dim + (dim/2);
    for (int off = -4; off <= 4; off++) printf("%lu ", C_rvv[mid + off]);
    printf("\n");

    printf("\n---  (Last row, last 8) ---\n");
    for (int x = dim - 8; x < dim; x++) printf("%lu ", C_rvv[(dim-1)*dim + x]);
    printf("\n");

    free(A); free(B); free(B_t);
    free(C_rvv);
    free(tmp_lo); free(tmp_hi); free(tmp_out);
}

// ================================================================
// MAIN
// ================================================================
int main(){
    size_t n=256;
    uint64_t m=40961;
    int it=50;

    srand(time(NULL));

    uint64_t mu[2];
    compute_mu(m,mu);

    printf("\n==== BASIC MODULAR ====\n");

    test_vv("Add VV", add_vv_mod_u64_golden, add_vv_mod_u64, n, m, it);
    test_vv("Sub VV", sub_vv_mod_u64_golden, sub_vv_mod_u64, n, m, it);

    test_vx("Add VX", add_vx_mod_u64_golden, add_vx_mod_u64, n, m, it);
    test_vx("Sub VX", sub_vx_mod_u64_golden, sub_vx_mod_u64, n, m, it);

    test_unary("Negate", neg_v_mod_u64_golden, neg_v_mod_u64, n, m, it);
    
    test_mul_vv_only("Mul VV", n, m, it, mu);
    test_mul_vx_only("Mul VX", n, m, it, mu);
    
    test_matmul(256);

    test_mod_matmul_rvv_only(256, m, mu);

    printf("\nAll tests done.\n");
    return 0;
}
