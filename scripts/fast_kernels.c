/* Sparse gather-multiply-accumulate mod PMOD for fast_mixed_ctx.py.

   out[udst[g], b, k] = sum_{e in group g} V[src[e], b, k] * C[e, k]   (mod PMOD)

   V: (N, B, K) int64 residues, C: (nnz, K) int64 residues, entries sorted by
   destination; groups start at starts[g].  PMOD < 2^31 is a compile-time
   constant (one shared object per prime), so the final reduction is a
   multiply-high and the products are 32x32 -> 64 bit multiplies.
*/
#include <stdint.h>
#include <stdlib.h>

#define LIM ((uint64_t)PMOD * (uint64_t)PMOD * 3u)   /* < 2^64 - 2^62 */

void gather_mul_acc(const int64_t *V, int64_t B, int64_t K,
                    const int64_t *src, const int64_t *starts, int64_t ngroups, int64_t nnz,
                    const int64_t *udst, const int64_t *C, int64_t *out)
{
    const int64_t BK = B * K;
    uint64_t *acc = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)BK);
    for (int64_t g = 0; g < ngroups; g++) {
        const int64_t e0 = starts[g], e1 = (g + 1 < ngroups) ? starts[g + 1] : nnz;
        for (int64_t t = 0; t < BK; t++) acc[t] = 0;
        for (int64_t e = e0; e < e1; e++) {
            const int64_t *v = V + src[e] * BK;
            const int64_t *c = C + e * K;
            for (int64_t b = 0; b < B; b++) {
                const int64_t *vb = v + b * K;
                uint64_t *ab = acc + b * K;
                for (int64_t k = 0; k < K; k++) {
                    uint64_t x = ab[k] + (uint64_t)(uint32_t)vb[k] * (uint64_t)(uint32_t)c[k];
                    ab[k] = x >= LIM ? x - LIM : x;
                }
            }
        }
        int64_t *o = out + udst[g] * BK;
        for (int64_t t = 0; t < BK; t++) o[t] = (int64_t)(acc[t] % (uint64_t)PMOD);
    }
    free(acc);
}

/* out[a, t] = sum_b M[a, b, k(t)] * X[b, t]  (mod PMOD), X: (m, B, K), M: (m, m, K) */
void small_matmul(const int64_t *M, const int64_t *X, int64_t m, int64_t B, int64_t K, int64_t *out)
{
    const int64_t BK = B * K;
    uint64_t *acc = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)BK);
    for (int64_t a = 0; a < m; a++) {
        for (int64_t t = 0; t < BK; t++) acc[t] = 0;
        for (int64_t b = 0; b < m; b++) {
            const int64_t *mk = M + (a * m + b) * K;
            const int64_t *x = X + b * BK;
            for (int64_t bb = 0; bb < B; bb++)
                for (int64_t k = 0; k < K; k++) {
                    uint64_t y = acc[bb * K + k] + (uint64_t)(uint32_t)x[bb * K + k] * (uint64_t)(uint32_t)mk[k];
                    acc[bb * K + k] = y >= LIM ? y - LIM : y;
                }
        }
        for (int64_t t = 0; t < BK; t++) out[a * BK + t] = (int64_t)(acc[t] % (uint64_t)PMOD);
    }
    free(acc);
}

/* out[p, c, k] = sum_a W[p, a, k] * G[a, c]  (mod PMOD), G point independent */
void mix_cols(const int64_t *W, const int64_t *G, int64_t Np, int64_t d, int64_t nc, int64_t K, int64_t *out)
{
    uint64_t *acc = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(nc * K));
    for (int64_t p = 0; p < Np; p++) {
        for (int64_t t = 0; t < nc * K; t++) acc[t] = 0;
        for (int64_t a = 0; a < d; a++) {
            const int64_t *w = W + (p * d + a) * K;
            for (int64_t c = 0; c < nc; c++) {
                const uint64_t g = (uint64_t)G[a * nc + c];
                uint64_t *ac = acc + c * K;
                for (int64_t k = 0; k < K; k++) {
                    uint64_t y = ac[k] + (uint64_t)(uint32_t)w[k] * g;
                    ac[k] = y >= LIM ? y - LIM : y;
                }
            }
        }
        for (int64_t t = 0; t < nc * K; t++) out[p * nc * K + t] = (int64_t)(acc[t] % (uint64_t)PMOD);
    }
    free(acc);
}

/* out[i] = a[i]^e mod PMOD */
void modpow(const int64_t *a, int64_t n, uint64_t e, int64_t *out)
{
    for (int64_t i = 0; i < n; i++) {
        uint64_t r = 1, b = (uint64_t)a[i] % PMOD, x = e;
        while (x) {
            if (x & 1) r = r * b % PMOD;
            b = b * b % PMOD;
            x >>= 1;
        }
        out[i] = (int64_t)r;
    }
}

/* gather_mul_acc with the coefficients stored as uint32 */
void gather_mul_acc_u32(const int64_t *V, int64_t B, int64_t K,
                        const int64_t *src, const int64_t *starts, int64_t ngroups, int64_t nnz,
                        const int64_t *udst, const uint32_t *C, int64_t *out)
{
    const int64_t BK = B * K;
    uint64_t *acc = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)BK);
    for (int64_t g = 0; g < ngroups; g++) {
        const int64_t e0 = starts[g], e1 = (g + 1 < ngroups) ? starts[g + 1] : nnz;
        for (int64_t t = 0; t < BK; t++) acc[t] = 0;
        for (int64_t e = e0; e < e1; e++) {
            const int64_t *v = V + src[e] * BK;
            const uint32_t *c = C + e * K;
            for (int64_t b = 0; b < B; b++) {
                const int64_t *vb = v + b * K;
                uint64_t *ab = acc + b * K;
                for (int64_t k = 0; k < K; k++) {
                    uint64_t x = ab[k] + (uint64_t)(uint32_t)vb[k] * (uint64_t)c[k];
                    ab[k] = x >= LIM ? x - LIM : x;
                }
            }
        }
        int64_t *o = out + udst[g] * BK;
        for (int64_t t = 0; t < BK; t++) o[t] = (int64_t)(acc[t] % (uint64_t)PMOD);
    }
    free(acc);
}
