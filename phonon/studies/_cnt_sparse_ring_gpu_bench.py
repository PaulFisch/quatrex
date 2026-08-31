"""GH200 reference benchmark for the exact CNT atom-triplet sparse ring.

The two FC3 vertex actions are custom sparse CUDA kernels; the final dense
``T @ U`` contraction is unchanged.  Both paths use the real CNT vertex,
complex128 Green-function batches, and the production batch size of 161.
This is a private performance/correctness study, not a production kernel.
"""

import argparse
import json
import time
from pathlib import Path

import cupy as cp
import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
VERTEX = ROOT / "phonon/studies/out/anderson_test/cnt33_L4_inputs/fc3_blocks.hdf5"
NW = 161
D = 36
NAT = D // 3


CUDA = r'''
#include <cuComplex.h>

extern "C" __global__
void sparse_left(const int n, const int nw, const int d, const int nat,
                 const int nrows, const int* rows, const int* ptr,
                 const int* contracted, const double* values,
                 const cuDoubleComplex* g, cuDoubleComplex* out) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= n) return;
    int b = index % d; index /= d;
    int e_cart = index % 3; index /= 3;
    int a_cart = index % 3; index /= 3;
    int row_position = index % nrows;
    int w = index / nrows;
    int row = rows[row_position];
    int a_atom = row / nat;
    int e_atom = row - a_atom * nat;
    int a = 3 * a_atom + a_cart;
    int e = 3 * e_atom + e_cart;
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int p = ptr[row_position]; p < ptr[row_position + 1]; ++p) {
        int c0 = 3 * contracted[p];
        for (int c_cart = 0; c_cart < 3; ++c_cart) {
            double value = values[27 * p + 9 * a_cart + 3 * c_cart + e_cart];
            cuDoubleComplex term = g[(w * d + c0 + c_cart) * d + b];
            sum = cuCadd(sum, make_cuDoubleComplex(value * cuCreal(term),
                                                   value * cuCimag(term)));
        }
    }
    out[((w * d + a) * d + e) * d + b] = sum;
}

extern "C" __global__
void sparse_right(const int n, const int nw, const int d, const int nat,
                  const int nrows, const int* rows, const int* ptr,
                  const int* contracted, const double* values,
                  const cuDoubleComplex* g, cuDoubleComplex* out) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= n) return;
    int e = index % d; index /= d;
    int b_cart = index % 3; index /= 3;
    int j_cart = index % 3; index /= 3;
    int row_position = index % nrows;
    int w = index / nrows;
    int row = rows[row_position];
    int j_atom = row / nat;
    int b_atom = row - j_atom * nat;
    int j = 3 * j_atom + j_cart;
    int b = 3 * b_atom + b_cart;
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int p = ptr[row_position]; p < ptr[row_position + 1]; ++p) {
        int d0 = 3 * contracted[p];
        for (int d_cart = 0; d_cart < 3; ++d_cart) {
            double value = values[27 * p + 9 * j_cart + 3 * d_cart + b_cart];
            cuDoubleComplex term = g[(w * d + e) * d + d0 + d_cart];
            sum = cuCadd(sum, make_cuDoubleComplex(value * cuCreal(term),
                                                   value * cuCimag(term)));
        }
    }
    out[((w * d + e) * d + b) * d + j] = sum;
}
'''


left_kernel = cp.RawKernel(CUDA, "sparse_left")
right_kernel = cp.RawKernel(CUDA, "sparse_right")


def load_blocks():
    blocks = {}
    with h5py.File(VERTEX, "r") as handle:
        for name in handle["fc3_blocks"]:
            dataset = handle["fc3_blocks"][name]
            key = tuple(int(dataset.attrs[x]) for x in ("I", "J", "K"))
            blocks[key] = np.asarray(dataset).real
    return blocks


def pack(block):
    cart = block.reshape(NAT, 3, NAT, 3, NAT, 3).transpose(0, 2, 4, 1, 3, 5)
    norms = np.linalg.norm(cart.reshape(NAT, NAT, NAT, 27), axis=-1)
    active = norms > 1e-12 * norms.max()
    rows = []
    ptr = [0]
    contracted = []
    values = []
    for outer in range(NAT):
        for other in range(NAT):
            middle = np.flatnonzero(active[outer, :, other])
            if not len(middle):
                continue
            rows.append(outer * NAT + other)
            for contracted_atom in middle:
                contracted.append(contracted_atom)
                values.append(cart[outer, contracted_atom, other])
            ptr.append(len(contracted))
    return tuple(cp.asarray(x) for x in (
        np.asarray(rows, dtype=np.int32),
        np.asarray(ptr, dtype=np.int32),
        np.asarray(contracted, dtype=np.int32),
        np.asarray(values, dtype=np.float64),
    )), int(active.sum())


def event_ms(function, repeats=8):
    for _ in range(2):
        function()
    cp.cuda.Stream.null.synchronize()
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    for _ in range(repeats):
        function()
    stop.record()
    stop.synchronize()
    return float(cp.cuda.get_elapsed_time(start, stop) / repeats)


def benchmark(name, left, right, ga, gb):
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    pl = cp.asarray(left.transpose(0, 2, 1).reshape(D * D, D))
    pr = cp.asarray(right.transpose(1, 2, 0).reshape(D, D * D))
    left_pack, left_active = pack(left)
    right_pack, right_active = pack(right)
    l_rows, l_ptr, l_contract, l_values = left_pack
    r_rows, r_ptr, r_contract, r_values = right_pack

    dense_t = cp.empty((NW, D * D, D), dtype=cp.complex128)
    dense_u = cp.empty((NW, D, D * D), dtype=cp.complex128)
    dense_s = cp.empty((NW, D, D), dtype=cp.complex128)
    sparse_t = cp.empty_like(dense_t)
    sparse_u = cp.empty_like(dense_u)
    sparse_s = cp.empty_like(dense_s)

    left_n = NW * len(l_rows) * 3 * 3 * D
    right_n = NW * len(r_rows) * 3 * 3 * D
    threads = 256

    def dense_actions():
        cp.matmul(pl, ga, out=dense_t)
        cp.matmul(gb, pr, out=dense_u)

    def dense_all():
        dense_actions()
        cp.matmul(
            dense_t.reshape(NW, D, D * D),
            dense_u.reshape(NW, D * D, D),
            out=dense_s,
        )

    def sparse_actions():
        sparse_t.fill(0)
        sparse_u.fill(0)
        left_kernel(
            ((left_n + threads - 1) // threads,), (threads,),
            (left_n, NW, D, NAT, len(l_rows), l_rows, l_ptr, l_contract,
             l_values, ga, sparse_t),
        )
        right_kernel(
            ((right_n + threads - 1) // threads,), (threads,),
            (right_n, NW, D, NAT, len(r_rows), r_rows, r_ptr, r_contract,
             r_values, gb, sparse_u),
        )

    def sparse_all():
        sparse_actions()
        cp.matmul(
            sparse_t.reshape(NW, D, D * D),
            sparse_u.reshape(NW, D * D, D),
            out=sparse_s,
        )

    dense_all()
    sparse_all()
    cp.cuda.Stream.null.synchronize()
    error = float(cp.linalg.norm(dense_s - sparse_s).get()
                  / cp.linalg.norm(dense_s).get())
    dense_actions_ms = event_ms(dense_actions)
    sparse_actions_ms = event_ms(sparse_actions)
    dense_total_ms = event_ms(dense_all)
    sparse_total_ms = event_ms(sparse_all)
    return {
        "case": name,
        "left_active_triplets": left_active,
        "right_active_triplets": right_active,
        "left_active_output_rows": int(len(l_rows)),
        "right_active_output_rows": int(len(r_rows)),
        "relative_error": error,
        "dense_vertex_actions_ms": dense_actions_ms,
        "sparse_vertex_actions_ms": sparse_actions_ms,
        "dense_total_ring_ms": dense_total_ms,
        "sparse_total_ring_ms": sparse_total_ms,
        "measured_total_ratio": sparse_total_ms / dense_total_ms,
        "measured_speedup": dense_total_ms / sparse_total_ms,
    }


def run():
    blocks = load_blocks()
    onsite = blocks[(2, 2, 2)]
    cross30 = blocks[(2, 2, 3)]
    cross52 = blocks[(2, 3, 2)]
    rng = np.random.default_rng(3108)
    ga = cp.asarray(
        rng.normal(size=(NW, D, D)) + 1j * rng.normal(size=(NW, D, D)))
    gb = cp.asarray(
        rng.normal(size=(NW, D, D)) + 1j * rng.normal(size=(NW, D, D)))
    started = time.time()
    representatives = {
        "onsite": onsite,
        "cross30": cross30,
        "cross52": cross52,
    }
    cases = [
        benchmark(f"{left}-{right}", left_value, right_value, ga, gb)
        for left, left_value in representatives.items()
        for right, right_value in representatives.items()
    ]
    # Exact category counts for the slow-rank 8x2 grouped mask, produced by
    # _cnt_reblock_acceleration.atom_sparse_quad_categories.
    task_weights = {
        "onsite-onsite": 48,
        "onsite-cross30": 184,
        "onsite-cross52": 92,
        "cross30-onsite": 184,
        "cross30-cross30": 708,
        "cross30-cross52": 354,
        "cross52-onsite": 92,
        "cross52-cross30": 354,
        "cross52-cross52": 177,
    }
    task_weighted_ratio = sum(
        task_weights[row["case"]] * row["measured_total_ratio"]
        for row in cases
    ) / sum(task_weights.values())
    result = {
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cupy": cp.__version__,
        "frequency_batch": NW,
        "dtype": "complex128",
        "cases": cases,
        "task_weights": task_weights,
        "task_weighted_total_ratio": task_weighted_ratio,
        "task_weighted_speedup": 1.0 / task_weighted_ratio,
        "wall_seconds": time.time() - started,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = run()
    rendered = json.dumps(result, indent=2)
    print(rendered, flush=True)
    if args.json is not None:
        args.json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
