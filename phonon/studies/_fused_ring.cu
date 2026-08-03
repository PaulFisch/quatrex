// Fused three-phonon ring contraction, complex128 (double2), sm_90.
//
// Contract (per task c, tau t; all complex double):
//   Tm = PL_c @ Ga_{c,t}          PL: (B*B, B) constant per task
//   Um = Gb_{c,t} @ PR_c          PR: (B, B*B) constant per task
//   S  = Tm.reshape(B, B*B) @ Um.reshape(B*B, B)      -> (B, B)
//
// Fusion identity used here: with the middle flat index (k2*B + p),
//   S[i,j]       = sum_p sum_k2 Tcol_p[i,k2] * Urow_p[k2,j]
//   Tcol_p[i,k2] = sum_k1 PL[i*B + k2, k1] * Ga[k1, p]
//   Urow_p[k2,j] = sum_d  Gb[k2, d]        * PR[d, p*B + j]
// so the B*B-long middle contraction is streamed one p-panel at a
// time and Tm/Um never touch global memory: Tcol/Urow live in shared
// memory (B*B each), S accumulates in registers.
//
// Layouts (C-contiguous, complex128 interleaved = double2):
//   PL (C, B*B, B), PR (C, B, B*B), Ga/Gb/S (C, W, B, B)
// Grid: C*W blocks; block bt handles the flat (c, t) = (bt/W, bt%W)
// pair (Ga/Gb/S offsets are just bt*B*B). Threads: (B/OPT)*B; thread
// tid owns the OPT outputs S[i0 + m*(B/OPT), j] with i0 = tid/B,
// j = tid%B.  OPT==1 additionally caches the thread's PL row (B
// double2) in registers; OPT>1 rows would spill, so PL is re-read
// through L1 per panel instead.
//
// Build: nvcc -arch=sm_90 -O3 -std=c++17 --use_fast_math -cubin \
//            -Xptxas=-v -o _fused_ring.cubin _fused_ring.cu
// (fast_math does not change FP64 mul/fma codegen; parity is checked
// against cuBLAS in _fused_ring_bench.py anyway.)

__device__ __forceinline__ void cfma(double2 &acc, const double2 a,
                                     const double2 b) {
    // acc += a * b, complex: 4 real FMAs.
    acc.x = fma(a.x, b.x, acc.x);
    acc.x = fma(-a.y, b.y, acc.x);
    acc.y = fma(a.x, b.y, acc.y);
    acc.y = fma(a.y, b.x, acc.y);
}

template <int B, int OPT, bool REGPL>
__device__ __forceinline__ void ring_impl(
        const double2 *__restrict__ PL, const double2 *__restrict__ PR,
        const double2 *__restrict__ Ga, const double2 *__restrict__ Gb,
        double2 *__restrict__ S, const int C, const int W) {
    constexpr int B2 = B * B;
    constexpr int NT = (B / OPT) * B;        // threads per block

    const long long bt = blockIdx.x;         // flat (c, t)
    const int c = (int)(bt / W);
    const double2 *pl = PL + (long long)c * (B2 * B);
    const double2 *pr = PR + (long long)c * (B * B2);
    const double2 *ga = Ga + bt * B2;
    const double2 *gb = Gb + bt * B2;
    double2 *s = S + bt * B2;

    extern __shared__ double2 sm[];
    double2 *Gas = sm;                       // [k1*B + p]
    double2 *Gbs = Gas + B2;                 // [k2*B + d]
    double2 *Tc = Gbs + B2;                  // [i*(B+1) + k2]
    double2 *Ur = Tc + B * (B + 1);          // [k2*(B+1) + j]

    const int tid = threadIdx.x;
    const int j = tid % B;
    const int i0 = tid / B;

    for (int e = tid; e < B2; e += NT) {
        Gas[e] = ga[e];
        Gbs[e] = gb[e];
    }

    double2 plrow[REGPL ? B : 1];
    if constexpr (REGPL) {
#pragma unroll
        for (int k1 = 0; k1 < B; ++k1) plrow[k1] = pl[tid * B + k1];
    }
    __syncthreads();

    double2 acc[OPT];
#pragma unroll
    for (int m = 0; m < OPT; ++m) acc[m] = make_double2(0.0, 0.0);

    for (int p = 0; p < B; ++p) {
        double2 tv[OPT], uv[OPT];
#pragma unroll
        for (int m = 0; m < OPT; ++m) {
            const int r = tid + m * NT;      // flat element of Tcol / Urow
            const int k2 = r / B, jj = r % B;
            tv[m] = make_double2(0.0, 0.0);
            uv[m] = make_double2(0.0, 0.0);
#pragma unroll
            for (int k1 = 0; k1 < B; ++k1) {
                double2 a;
                if constexpr (REGPL) a = plrow[k1];
                else a = pl[r * B + k1];
                cfma(tv[m], a, Gas[k1 * B + p]);
            }
#pragma unroll
            for (int d = 0; d < B; ++d)
                cfma(uv[m], Gbs[k2 * B + d], pr[d * B2 + p * B + jj]);
        }
        __syncthreads();                     // iter p-1 step-3 readers done
#pragma unroll
        for (int m = 0; m < OPT; ++m) {
            const int r = tid + m * NT;
            Tc[(r / B) * (B + 1) + (r % B)] = tv[m];
            Ur[(r / B) * (B + 1) + (r % B)] = uv[m];
        }
        __syncthreads();
#pragma unroll
        for (int k2 = 0; k2 < B; ++k2) {
            const double2 u = Ur[k2 * (B + 1) + j];
#pragma unroll
            for (int m = 0; m < OPT; ++m)
                cfma(acc[m], Tc[(i0 + m * (B / OPT)) * (B + 1) + k2], u);
        }
    }
#pragma unroll
    for (int m = 0; m < OPT; ++m) s[tid + m * NT] = acc[m];
}

#define RING_WRAP(NAME, B, OPT, REGPL, MINBLK)                            \
    extern "C" __global__ void                                            \
    __launch_bounds__(((B) / (OPT)) * (B), MINBLK)                        \
    NAME(const double2 *__restrict__ PL, const double2 *__restrict__ PR,  \
         const double2 *__restrict__ Ga, const double2 *__restrict__ Gb,  \
         double2 *__restrict__ S, const int C, const int W) {             \
        ring_impl<B, OPT, REGPL>(PL, PR, Ga, Gb, S, C, W);                \
    }

// name suffix: oN = N outputs/thread; x2 = >=2 blocks/SM forced
// (caps regs, may spill); nr = PL read through L1 instead of regs.
RING_WRAP(fused_ring_b18_o1, 18, 1, true, 1)     // 324 thr, PL in regs
RING_WRAP(fused_ring_b18_o1x2, 18, 1, true, 2)   // + reg cap for 2 blk/SM
RING_WRAP(fused_ring_b18_o1nr, 18, 1, false, 2)  // PL via L1, 2 blk/SM
RING_WRAP(fused_ring_b18_o2, 18, 2, false, 1)    // 162 thr, 2 outputs
RING_WRAP(fused_ring_b18_o3, 18, 3, false, 1)    // 108 thr, 3 outputs
RING_WRAP(fused_ring_b36_o2, 36, 2, false, 1)    // 648 thr, 2 outputs
RING_WRAP(fused_ring_b36_o4, 36, 4, false, 1)    // 324 thr, 4 outputs

// ---- v2: register-tiled stage 3 (kernel round 2, after ncu P1) ----
//
// P1 attribution of v1: smem pipe 79% / FMA 28% / 162 regs -> 17%
// occupancy. Stage 3 paid (1+OPT)/OPT smem reads per cfma. Here each
// thread owns a TI x TJ output tile: per (panel, k2) it reads TI Tc
// + TJ Ur entries for TI*TJ cfmas -> (TI+TJ)/(TI*TJ) reads/cfma
// (2x2: 1.0, 2x3: 0.83 vs v1-o1's 2.0). PL is read through L1 (no
// register cache: v1's REGPL was the register-pressure culprit);
// stages 1/2 split B*B elements evenly across the (B/TI)*(B/TJ)
// threads. Same smem layout/footprint as v1.
template <int B, int TI, int TJ>
__device__ __forceinline__ void ring_rt_impl(
        const double2 *__restrict__ PL, const double2 *__restrict__ PR,
        const double2 *__restrict__ Ga, const double2 *__restrict__ Gb,
        double2 *__restrict__ S, const int C, const int W) {
    constexpr int B2 = B * B;
    constexpr int NI = B / TI, NJ = B / TJ;
    constexpr int NT = NI * NJ;              // threads per block
    constexpr int E = B2 / NT;               // stage-1/2 elems per thread

    const long long bt = blockIdx.x;
    const int c = (int)(bt / W);
    const double2 *pl = PL + (long long)c * (B2 * B);
    const double2 *pr = PR + (long long)c * (B * B2);
    const double2 *ga = Ga + bt * B2;
    const double2 *gb = Gb + bt * B2;
    double2 *s = S + bt * B2;

    extern __shared__ double2 sm[];
    double2 *Gas = sm;                       // [k1*B + p]
    double2 *Gbs = Gas + B2;                 // [k2*B + d]
    double2 *Tc = Gbs + B2;                  // [i*(B+1) + k2]
    double2 *Ur = Tc + B * (B + 1);          // [k2*(B+1) + j]

    const int tid = threadIdx.x;
    const int ti = (tid / NJ) * TI;          // tile row origin in i
    const int tj = (tid % NJ) * TJ;          // tile col origin in j

    for (int e = tid; e < B2; e += NT) {
        Gas[e] = ga[e];
        Gbs[e] = gb[e];
    }
    __syncthreads();

    double2 acc[TI][TJ];
#pragma unroll
    for (int a = 0; a < TI; ++a)
#pragma unroll
        for (int bb = 0; bb < TJ; ++bb) acc[a][bb] = make_double2(0.0, 0.0);

    for (int p = 0; p < B; ++p) {
        double2 tv[E], uv[E];
#pragma unroll
        for (int m = 0; m < E; ++m) {
            const int r = tid + m * NT;      // flat element of Tcol / Urow
            const int k2 = r / B, jj = r % B;
            tv[m] = make_double2(0.0, 0.0);
            uv[m] = make_double2(0.0, 0.0);
#pragma unroll
            for (int k1 = 0; k1 < B; ++k1)
                cfma(tv[m], __ldg(&pl[r * B + k1]), Gas[k1 * B + p]);
#pragma unroll
            for (int d = 0; d < B; ++d)
                cfma(uv[m], Gbs[k2 * B + d],
                     __ldg(&pr[d * B2 + p * B + jj]));
        }
        __syncthreads();                     // panel p-1 stage-3 readers done
#pragma unroll
        for (int m = 0; m < E; ++m) {
            const int r = tid + m * NT;
            Tc[(r / B) * (B + 1) + (r % B)] = tv[m];
            Ur[(r / B) * (B + 1) + (r % B)] = uv[m];
        }
        __syncthreads();
#pragma unroll
        for (int k2 = 0; k2 < B; ++k2) {
            double2 tr[TI], ur[TJ];
#pragma unroll
            for (int a = 0; a < TI; ++a)
                tr[a] = Tc[(ti + a) * (B + 1) + k2];
#pragma unroll
            for (int bb = 0; bb < TJ; ++bb)
                ur[bb] = Ur[k2 * (B + 1) + (tj + bb)];
#pragma unroll
            for (int a = 0; a < TI; ++a)
#pragma unroll
                for (int bb = 0; bb < TJ; ++bb)
                    cfma(acc[a][bb], tr[a], ur[bb]);
        }
    }
#pragma unroll
    for (int a = 0; a < TI; ++a)
#pragma unroll
        for (int bb = 0; bb < TJ; ++bb)
            s[(ti + a) * B + (tj + bb)] = acc[a][bb];
}

#define RING_RT_WRAP(NAME, B, TI, TJ, MINBLK)                             \
    extern "C" __global__ void                                            \
    __launch_bounds__(((B) / (TI)) * ((B) / (TJ)), MINBLK)                \
    NAME(const double2 *__restrict__ PL, const double2 *__restrict__ PR,  \
         const double2 *__restrict__ Ga, const double2 *__restrict__ Gb,  \
         double2 *__restrict__ S, const int C, const int W) {             \
        ring_rt_impl<B, TI, TJ>(PL, PR, Ga, Gb, S, C, W);                 \
    }

RING_RT_WRAP(fused_ring_b18_rt22, 18, 2, 2, 1)    // 81 thr, 2x2 tiles
RING_RT_WRAP(fused_ring_b18_rt22x4, 18, 2, 2, 4)  // + 4 blk/SM reg cap
RING_RT_WRAP(fused_ring_b18_rt23, 18, 2, 3, 1)    // 54 thr, 2x3 tiles
RING_RT_WRAP(fused_ring_b18_rt33, 18, 3, 3, 1)    // 36 thr, 3x3 tiles
RING_RT_WRAP(fused_ring_b36_rt22, 36, 2, 2, 1)    // 324 thr, 2x2 tiles
RING_RT_WRAP(fused_ring_b36_rt33, 36, 3, 3, 1)    // 144 thr, 3x3 tiles

// ---- v3: fixed-k2 element mapping + stage-3 tiles ----
//
// rt lesson: tiling stage 3 alone loses what stages 1/2 pay when PL
// leaves registers. Here each thread's E stage-1/2 elements all share
// ONE k2 row (r = k2*B + jj), so per panel the Gb row read serves E
// elements (stage-2 smem traffic / E) with zero extra register cost;
// PL stays in L1; stage 3 keeps the 2x3 register tile of rt23.
// NT = 54: k2 = tid/3 covers [0,18) with 3 threads x E=6 jj-columns.
template <int B>
__device__ __forceinline__ void ring_v3_impl(
        const double2 *__restrict__ PL, const double2 *__restrict__ PR,
        const double2 *__restrict__ Ga, const double2 *__restrict__ Gb,
        double2 *__restrict__ S, const int C, const int W) {
    constexpr int B2 = B * B;
    constexpr int PERK = 3;                  // threads per k2 row
    constexpr int NT = B * PERK;             // 54 at B=18
    constexpr int E = B / PERK;              // 6 jj columns per thread
    constexpr int TI = 2, TJ = 3;            // stage-3 tile (NT = 9*6*... )

    const long long bt = blockIdx.x;
    const int c = (int)(bt / W);
    const double2 *pl = PL + (long long)c * (B2 * B);
    const double2 *pr = PR + (long long)c * (B * B2);
    const double2 *ga = Ga + bt * B2;
    const double2 *gb = Gb + bt * B2;
    double2 *s = S + bt * B2;

    extern __shared__ double2 sm[];
    double2 *Gas = sm;
    double2 *Gbs = Gas + B2;
    double2 *Tc = Gbs + B2;                  // [i*(B+1) + k2]
    double2 *Ur = Tc + B * (B + 1);          // [k2*(B+1) + j]

    const int tid = threadIdx.x;
    const int k2r = tid / PERK;              // this thread's k2 row
    const int jj0 = (tid % PERK) * E;        // first jj column
    const int ti = (tid / (B / TJ)) * TI;    // stage-3 tile origins
    const int tj = (tid % (B / TJ)) * TJ;

    for (int e = tid; e < B2; e += NT) {
        Gas[e] = ga[e];
        Gbs[e] = gb[e];
    }
    __syncthreads();

    double2 acc[TI][TJ];
#pragma unroll
    for (int a = 0; a < TI; ++a)
#pragma unroll
        for (int bb = 0; bb < TJ; ++bb) acc[a][bb] = make_double2(0.0, 0.0);

    for (int p = 0; p < B; ++p) {
        double2 tv[E], uv[E];
#pragma unroll
        for (int m = 0; m < E; ++m) {
            tv[m] = make_double2(0.0, 0.0);
            uv[m] = make_double2(0.0, 0.0);
        }
        // stage 1: tv[m] for elements r = k2r*B + (jj0+m)
#pragma unroll
        for (int k1 = 0; k1 < B; ++k1) {
            const double2 g = Gas[k1 * B + p];       // broadcast
#pragma unroll
            for (int m = 0; m < E; ++m)
                cfma(tv[m], __ldg(&pl[(k2r * B + jj0 + m) * B + k1]), g);
        }
        // stage 2: one Gb smem read serves all E columns
#pragma unroll
        for (int d = 0; d < B; ++d) {
            const double2 g = Gbs[k2r * B + d];
#pragma unroll
            for (int m = 0; m < E; ++m)
                cfma(uv[m], g, __ldg(&pr[d * B2 + p * B + jj0 + m]));
        }
        __syncthreads();
#pragma unroll
        for (int m = 0; m < E; ++m) {
            Tc[k2r * (B + 1) + (jj0 + m)] = tv[m];   // row i = k2r? no:
            // Tcol element (i,k2) uses flat r = i*B + k2 in v1's layout;
            // here r = k2r*B + jj -> i = k2r, k2 = jj0+m.
            Ur[k2r * (B + 1) + (jj0 + m)] = uv[m];
        }
        __syncthreads();
#pragma unroll
        for (int k2 = 0; k2 < B; ++k2) {
            double2 tr[TI], ur[TJ];
#pragma unroll
            for (int a = 0; a < TI; ++a)
                tr[a] = Tc[(ti + a) * (B + 1) + k2];
#pragma unroll
            for (int bb = 0; bb < TJ; ++bb)
                ur[bb] = Ur[k2 * (B + 1) + (tj + bb)];
#pragma unroll
            for (int a = 0; a < TI; ++a)
#pragma unroll
                for (int bb = 0; bb < TJ; ++bb)
                    cfma(acc[a][bb], tr[a], ur[bb]);
        }
    }
#pragma unroll
    for (int a = 0; a < TI; ++a)
#pragma unroll
        for (int bb = 0; bb < TJ; ++bb)
            s[(ti + a) * B + (tj + bb)] = acc[a][bb];
}

#define RING_V3_WRAP(NAME, B, MINBLK)                                     \
    extern "C" __global__ void                                            \
    __launch_bounds__((B) * 3, MINBLK)                                    \
    NAME(const double2 *__restrict__ PL, const double2 *__restrict__ PR,  \
         const double2 *__restrict__ Ga, const double2 *__restrict__ Gb,  \
         double2 *__restrict__ S, const int C, const int W) {             \
        ring_v3_impl<B>(PL, PR, Ga, Gb, S, C, W);                         \
    }

RING_V3_WRAP(fused_ring_b18_v3, 18, 1)
RING_V3_WRAP(fused_ring_b18_v3x3, 18, 3)
