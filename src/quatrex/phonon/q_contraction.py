# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Coupled-q contractions for the three-phonon self-energy."""

import numpy as np
from qttools import NDArray
from qttools.comm import comm as ranks

from quatrex.phonon.bubble import phi_perms, ring_contract_pre, ring_pool


class QContractionMixin:
    """Dense and factored coupled-q contraction methods."""

    @staticmethod
    def _qfold_is_translation_invariant(qv, xp) -> bool:
        """Exact test: does every q-folded vertex block satisfy
        ``Phi[(I+1, K1+1, K2+1)] == Phi[(I, K1, K2)]``?

        Only shifts whose image is also present are compared, and at least
        one such pair must exist -- otherwise there is nothing to share and
        we must not claim invariance. Any mismatch (including a missing
        partner where a present one is expected) returns False, so the
        caller falls back to the absolute key.
        """
        compared = 0
        for phi in qv.values():
            if not isinstance(phi, dict):
                return False
            for (I, K1, K2), blk in phi.items():
                shifted = phi.get((I + 1, K1 + 1, K2 + 1))
                if shifted is None:
                    continue
                a = xp.asarray(blk)
                b = xp.asarray(shifted)
                if a.shape != b.shape or not bool(xp.all(a == b)):
                    return False
                compared += 1
        return compared > 0

    def _rotate_internal_q(
        self, owned, qdm, qv, q_lo, q_hi, nq, nk, n_tau, dtype,
        legs, _fold_l, _fold_g, stlv, stgv, start, xp,
        q_split=False, **kw,
    ):
        """Drive :meth:`_contract_dense_q` over the internal-momentum pairs.

        With the transverse axis replicated this is one call over the whole
        axis -- the pre-rotation behaviour, bit for bit.

        With it sectioned (``q_split``) the bubble cannot be evaluated from a
        rank's own slice alone: it is a CONVOLUTION over q, so every term at
        an external momentum needs a SECOND internal momentum, generally on
        another rank. The fix is systolic rather than a gather, because a
        gather would restore the replication the sectioning exists to remove:

        * leg A stays this rank's own slice throughout;
        * leg B starts as a copy of it and is passed once around ``comm.q``,
          so after step ``s`` rank ``r`` holds the slice owned by
          ``(r + s) mod P_q``;
        * each step contracts the pairs it now holds and ACCUMULATES.

        Across all ranks and steps every ordered slice pair occurs exactly
        once, so the q sum is complete
        (``test_internal_q_slice_pairs_tile_the_convolution``). The existing
        ``all_reduce`` over ``comm.q`` still finishes the job: each rank now
        contributes partial sums to every external momentum instead of
        complete sums to a disjoint few, which that collective handles
        identically.

        See ``phonon/docs/bubble_positivity.md`` Sec. 7.
        """
        bounds = [r * nq // ranks.q.size for r in range(ranks.q.size + 1)]
        contract = (
            self._contract_micro_dense_q
            if self._micro_pair_index is not None else self._contract_dense_q
        )
        if not q_split:
            # Whole axis local: one pass, offsets zero, plain assignment.
            contract(
                owned, qdm, qv, q_lo, q_hi, nq, nk, n_tau, dtype,
                *legs, _fold_l, _fold_g, stlv, stgv, start, xp, **kw)
            return

        rank, size = ranks.q.rank, ranks.q.size
        a_lo, a_hi = bounds[rank], bounds[rank + 1]
        buf = tuple({k: xp.array(v, copy=True) for k, v in d.items()}
                    for d in legs)
        for step in range(size):
            src = (rank + step) % size
            b_lo, b_hi = bounds[src], bounds[src + 1]
            contract(
                owned, qdm, qv, 0, nq, nq, nk, n_tau, dtype,
                *legs, _fold_l, _fold_g, stlv, stgv, start, xp,
                a_slice=(a_lo, a_hi), b_slice=(b_lo, b_hi),
                a_off=a_lo, b_off=b_lo, legs_b=buf,
                accumulate=(step > 0), **kw)
            if step == size - 1:
                break
            buf = self._rotate_q_buffers(buf, bounds, rank, size, step, xp)

    def _micro_links_for_outputs(
        self, outputs: list[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        """Primitive Green links consumed by grouped output pairs."""
        links: set[tuple[int, int]] = set()
        for pair in outputs:
            for quads in self._micro_pair_index.get(pair, {}).values():
                for quad in quads:
                    links.add((quad.k1, quad.k1p))
                    links.add((quad.k2, quad.k2p))
        return links

    def _contract_micro_dense_gamma(
        self, owned, *, n_tau, dtype, gl, gg, glr, ggr,
        stlv, stgv, start, xp,
    ) -> None:
        """Dense Gamma bubble on primitive FC3 and grouped Dyson blocks."""
        if getattr(self, "_micro_gamma_pre", None) is None:
            pre = {}
            for group_pair, primitive_pairs in self._micro_pair_index.items():
                tasks = []
                for (i, j), quads in primitive_pairs.items():
                    _bi, si = self._micro_layout.locate(i)
                    _bj, sj = self._micro_layout.locate(j)
                    for quad in quads:
                        tasks.append((
                            si.start, sj.start, quad.k1, quad.k1p,
                            quad.k2, quad.k2p,
                        ) + phi_perms(
                            xp.asarray(quad.phi_left),
                            xp.asarray(quad.phi_right), xp,
                        ))
                pre[group_pair] = tasks
            self._micro_gamma_pre = pre

        def _run(lo, hi):
            result = {}
            for I, J in owned:
                out_l = xp.zeros(
                    (hi - lo, int(self.block_sizes[I]),
                     int(self.block_sizes[J])), dtype=dtype)
                out_g = xp.zeros_like(out_l)
                for (oi, oj, k1, k1p, k2, k2p, *p) in (
                    self._micro_gamma_pre.get((I, J), ())
                ):
                    PL, PR, nI, bK2, nJ = p
                    gla, glb = gl[(k1, k1p)][lo:hi], gl[(k2, k2p)][lo:hi]
                    gga, ggb = gg[(k1, k1p)][lo:hi], gg[(k2, k2p)][lo:hi]
                    ggra, ggrb = ggr[(k1, k1p)][lo:hi], ggr[(k2, k2p)][lo:hi]
                    glra, glrb = glr[(k1, k1p)][lo:hi], glr[(k2, k2p)][lo:hi]

                    def ring(a, b):
                        return ring_contract_pre(
                            PL, PR, nI, bK2, nJ, a, b, xp)

                    sl = ring(gla, glb) + ring(gla, ggrb) + ring(ggra, glb)
                    sg = ring(gga, ggb) + ring(gga, glrb) + ring(glra, ggb)
                    d = self._micro_layout.micro_dof
                    out_l[:, oi:oi + d, oj:oj + d] += sl
                    out_g[:, oi:oi + d, oj:oj + d] += sg
                result[(I, J)] = (out_l, out_g)
            return result

        pool, n_threads = ring_pool()
        nt = min(n_threads, max(1, n_tau // self._tau_min_chunk))
        if pool is not None and xp is np and nt > 1:
            bounds = [(i * n_tau // nt, (i + 1) * n_tau // nt)
                      for i in range(nt)]
            chunks = list(pool.map(lambda bound: _run(*bound), bounds))
            result = {
                pair: tuple(xp.concatenate([c[pair][side] for c in chunks],
                                           axis=0)
                            for side in (0, 1))
                for pair in owned
            }
        else:
            result = _run(0, n_tau)
        for (I, J), (out_l, out_g) in result.items():
            stlv.blocks[I - start, J - start] = out_l
            stgv.blocks[I - start, J - start] = out_g

    def _contract_micro_dense_q(
        self, owned, qdm, qv, q_lo, q_hi, nq, nk, n_tau, dtype,
        gl_q, gg_q, glr_q, ggr_q, _fold_l, _fold_g,
        stlv, stgv, start, xp,
        fast_now=False, verify_now=False, stxv=None, release=None,
        a_slice=None, b_slice=None, a_off=0, b_off=0, legs_b=None,
        accumulate=False,
    ) -> None:
        """Dense coupled-q reference contraction on primitive microblocks."""
        if fast_now or verify_now:
            raise NotImplementedError(
                "primitive microblocks use the plain six-ring coupled-q path")
        del stxv, release
        a_lo, a_hi = (0, nq) if a_slice is None else a_slice
        b_lo, b_hi = (0, nq) if b_slice is None else b_slice
        gl_a, gg_a, glr_a, ggr_a = gl_q, gg_q, glr_q, ggr_q
        gl_b, gg_b, glr_b, ggr_b = (
            (gl_q, gg_q, glr_q, ggr_q) if legs_b is None else legs_b)
        cache_key = (q_lo, q_hi, nq, a_lo, a_hi, b_lo, b_hi, tuple(owned))
        transient_tile = a_slice is not None or b_slice is not None
        cache = getattr(self, "_micro_qtasks_cache", {})
        qtasks = None if transient_tile else cache.get(cache_key)
        if qtasks is None:
            qsum = np.empty((nq, nq), dtype=np.int64)
            qprime = np.arange(nq, dtype=np.int64)
            for iq_ext in range(nq):
                qsum[qprime, np.asarray(qdm[iq_ext], dtype=np.int64)] = iq_ext
            perm_left: dict[tuple, NDArray] = {}
            perm_right: dict[tuple, NDArray] = {}
            qtasks = {}
            bulk_vertex = self._vfactors is not None
            for iqp in range(a_lo, a_hi):
                for iq2 in range(b_lo, b_hi):
                    iq_ext = int(qsum[iqp, iq2])
                    if not q_lo <= iq_ext < q_hi:
                        continue
                    phi_l = qv.get((iqp, iq2))
                    phi_r = qv.get((iq2, iqp))
                    if phi_l is None or phi_r is None:
                        continue
                    for group_pair in owned:
                        for (i, j), quads in self._micro_pair_index.get(
                            group_pair, {}
                        ).items():
                            _bi, si = self._micro_layout.locate(i)
                            _bj, sj = self._micro_layout.locate(j)
                            for quad in quads:
                                pl = phi_l.get((i, quad.k1, quad.k2))
                                pr = phi_r.get((j, quad.k2p, quad.k1p))
                                if pl is None or pr is None:
                                    continue
                                left_key = (
                                    (iqp, iq2, quad.k1 - i, quad.k2 - i,
                                     "left")
                                    if bulk_vertex else
                                    (iqp, iq2, i, quad.k1, quad.k2)
                                )
                                right_key = (
                                    (iq2, iqp, quad.k2p - j, quad.k1p - j,
                                     "right")
                                    if bulk_vertex else
                                    (iq2, iqp, j, quad.k2p, quad.k1p)
                                )
                                PL = perm_left.get(left_key)
                                if PL is None:
                                    nI, bK1, bK2 = pl.shape
                                    PL = xp.ascontiguousarray(
                                        xp.conj(xp.asarray(pl)).transpose(
                                            0, 2, 1)
                                    ).reshape(nI * bK2, bK1)
                                    perm_left[left_key] = PL
                                else:
                                    nI, bK1, bK2 = pl.shape
                                PR = perm_right.get(right_key)
                                if PR is None:
                                    nJ, bK2p, bK1p = pr.shape
                                    PR = xp.ascontiguousarray(
                                        xp.asarray(pr).transpose(1, 2, 0)
                                    ).reshape(bK2p, bK1p * nJ)
                                    perm_right[right_key] = PR
                                else:
                                    nJ = pr.shape[0]
                                p = (PL, PR, nI, bK2, nJ)
                                qtasks.setdefault(group_pair, []).append((
                                    iq_ext, iqp, iq2, si.start, sj.start,
                                    quad.k1, quad.k1p, quad.k2, quad.k2p,
                                ) + p)
            if not transient_tile:
                cache[cache_key] = qtasks
                self._micro_qtasks_cache = cache

        if not getattr(self, "_ring_stats_printed", False) and ranks.rank == 0:
            self._ring_stats_printed = True
            ntasks = sum(len(v) for v in qtasks.values())
            print(
                "PhPh SSE ring (microblock dense coupled-q): "
                f"group_pairs={len(qtasks)} qtasks={ntasks} nq={nq} "
                f"q_local={q_hi - q_lo} n_tau={n_tau}", flush=True)

        def _run(lo, hi):
            result = {}
            d = self._micro_layout.micro_dof
            for I, J in owned:
                out_l = xp.zeros(
                    (hi - lo, nq, int(self.block_sizes[I]),
                     int(self.block_sizes[J])), dtype=dtype)
                out_g = xp.zeros_like(out_l)
                for (iqe, iqp, iq2, oi, oj, k1, k1p, k2, k2p,
                     *p) in qtasks.get((I, J), ()):
                    gla = gl_a[(k1, k1p)][lo:hi, iqp - a_off]
                    glb = gl_b[(k2, k2p)][lo:hi, iq2 - b_off]
                    gga = gg_a[(k1, k1p)][lo:hi, iqp - a_off]
                    ggb = gg_b[(k2, k2p)][lo:hi, iq2 - b_off]
                    ggra = ggr_a[(k1, k1p)][lo:hi, iqp - a_off]
                    ggrb = ggr_b[(k2, k2p)][lo:hi, iq2 - b_off]
                    glra = glr_a[(k1, k1p)][lo:hi, iqp - a_off]
                    glrb = glr_b[(k2, k2p)][lo:hi, iq2 - b_off]
                    sl = _fold_l(p, gla, glb, ggra, ggrb)
                    sg = _fold_g(p, gga, ggb, glra, glrb)
                    out_l[:, iqe, oi:oi + d, oj:oj + d] += sl
                    out_g[:, iqe, oi:oi + d, oj:oj + d] += sg
                result[(I, J)] = (out_l, out_g)
            return result

        def _run_batched(lo, hi):
            """GPU/NumPy task-batched form of the same primitive ring.

            The original microblock reference loop launched six sets of three
            tiny matrix multiplications for every q/quad task.  At q9 L5 that
            is more than five million Python iterations.  Flattening the task
            axis into strided batched GEMMs changes only the final reduction
            order and is the same optimisation used by the grouped dense
            kernel below.
            """
            w = hi - lo

            def _stack(family, keys):
                if set(family) != set(keys):
                    return None
                return xp.stack([
                    xp.ascontiguousarray(family[key]) for key in keys
                ])

            a_keys = sorted(gl_a)
            b_keys = sorted(gl_b)
            a_index = {key: i for i, key in enumerate(a_keys)}
            b_index = {key: i for i, key in enumerate(b_keys)}
            GL = _stack(gl_a, a_keys)
            GG = _stack(gg_a, a_keys)
            GGR = _stack(ggr_a, a_keys)
            GLR = _stack(glr_a, a_keys)
            GLb = _stack(gl_b, b_keys)
            GGb = _stack(gg_b, b_keys)
            GGRb = _stack(ggr_b, b_keys)
            GLRb = _stack(glr_b, b_keys)
            if any(value is None for value in
                   (GL, GG, GGR, GLR, GLb, GGb, GGRb, GLRb)):
                return _run(lo, hi)

            def _scatter_micro(out, iqe, oi, oj, values, bs_i, bs_j):
                d = values.shape[-1]
                rr = xp.arange(d, dtype=xp.int64)[None, :, None]
                cc = xp.arange(d, dtype=xp.int64)[None, None, :]
                target = (
                    iqe[:, None, None] * (bs_i * bs_j)
                    + (oi[:, None, None] + rr) * bs_j
                    + (oj[:, None, None] + cc)
                ).reshape(-1)
                vals = values.transpose(0, 2, 3, 1).reshape(-1, values.shape[1])
                out_flat = out.reshape(out.shape[0], -1).T
                if xp is np:
                    np.add.at(out_flat, target, vals)
                else:
                    import cupyx
                    cupyx.scatter_add(out_flat.real, target, vals.real)
                    cupyx.scatter_add(out_flat.imag, target, vals.imag)

            result = {}
            for I, J in owned:
                tasks = qtasks.get((I, J), ())
                bs_i = int(self.block_sizes[I])
                bs_j = int(self.block_sizes[J])
                out_l = xp.zeros((w, nq, bs_i, bs_j), dtype=dtype)
                out_g = xp.zeros_like(out_l)
                if not tasks:
                    result[(I, J)] = (out_l, out_g)
                    continue

                groups: dict[tuple, list] = {}
                for task in tasks:
                    PL, PR, nI, bK2, nJ = task[9:14]
                    groups.setdefault(
                        (PL.shape, PR.shape, nI, bK2, nJ), []
                    ).append(task)
                for (shape_l, shape_r, nI, bK2, nJ), ts in groups.items():
                    n_tasks = len(ts)
                    bK1 = shape_l[1]
                    bK1p = shape_r[1] // nJ
                    per_task_tau = 16 * (
                        2 * bK1 * bK1p + shape_l[0] * bK1p
                        + bK2 * shape_r[1] + 2 * nI * nJ
                    )
                    preferred_tasks = min(n_tasks, 256)
                    tau_width = int(max(1, min(
                        w, self._tau_chunk_bytes
                        // max(per_task_tau * preferred_tasks, 1)
                    )))
                    task_width = max(1, min(
                        n_tasks, self._tau_chunk_bytes
                        // max(per_task_tau * tau_width, 1)
                    ))
                    iqe_all = xp.asarray([t[0] for t in ts], dtype=xp.int64)
                    qa_all = xp.asarray([t[1] for t in ts], dtype=xp.int64) - a_off
                    qb_all = xp.asarray([t[2] for t in ts], dtype=xp.int64) - b_off
                    oi_all = xp.asarray([t[3] for t in ts], dtype=xp.int64)
                    oj_all = xp.asarray([t[4] for t in ts], dtype=xp.int64)
                    ai_all = xp.asarray([
                        a_index[(t[5], t[6])] for t in ts
                    ], dtype=xp.int64)
                    bi_all = xp.asarray([
                        b_index[(t[7], t[8])] for t in ts
                    ], dtype=xp.int64)

                    for c0 in range(0, n_tasks, task_width):
                        c1 = min(c0 + task_width, n_tasks)
                        PLc = xp.stack([t[9] for t in ts[c0:c1]])[:, None]
                        PRc = xp.stack([t[10] for t in ts[c0:c1]])[:, None]
                        ai, bi = ai_all[c0:c1], bi_all[c0:c1]
                        qa, qb = qa_all[c0:c1], qb_all[c0:c1]

                        for w0 in range(0, w, tau_width):
                            w1 = min(w0 + tau_width, w)
                            tau = xp.arange(lo + w0, lo + w1)[None, :]

                            def _legs(A, B):
                                return (
                                    A[ai[:, None], tau, qa[:, None]],
                                    B[bi[:, None], tau, qb[:, None]],
                                )

                            def _ring(Ga, Gb):
                                left = PLc @ Ga
                                right = Gb @ PRc
                                return (
                                    left.reshape(c1 - c0, w1 - w0,
                                                 nI, bK2 * bK1p)
                                    @ right.reshape(c1 - c0, w1 - w0,
                                                    bK2 * bK1p, nJ)
                                )

                            sl = None
                            for A, B in ((GL, GLb), (GL, GGRb), (GGR, GLb)):
                                Ga, Gb = _legs(A, B)
                                term = _ring(Ga, Gb)
                                sl = term if sl is None else sl + term
                            sg = None
                            for A, B in ((GG, GGb), (GG, GLRb), (GLR, GGb)):
                                Ga, Gb = _legs(A, B)
                                term = _ring(Ga, Gb)
                                sg = term if sg is None else sg + term
                            _scatter_micro(
                                out_l[w0:w1], iqe_all[c0:c1],
                                oi_all[c0:c1], oj_all[c0:c1],
                                sl.astype(dtype, copy=False), bs_i, bs_j,
                            )
                            _scatter_micro(
                                out_g[w0:w1], iqe_all[c0:c1],
                                oi_all[c0:c1], oj_all[c0:c1],
                                sg.astype(dtype, copy=False), bs_i, bs_j,
                            )
                result[(I, J)] = (out_l, out_g)
            return result

        result = (
            _run_batched(0, n_tau) if self._dense_q_batched
            else _run(0, n_tau)
        )
        for (I, J), (out_l, out_g) in result.items():
            shape = ((n_tau,) + tuple(nk) +
                     (int(self.block_sizes[I]), int(self.block_sizes[J])))
            if accumulate:
                stlv.blocks[I - start, J - start] = (
                    stlv.blocks[I - start, J - start] + out_l.reshape(shape))
                stgv.blocks[I - start, J - start] = (
                    stgv.blocks[I - start, J - start] + out_g.reshape(shape))
            else:
                stlv.blocks[I - start, J - start] = out_l.reshape(shape)
                stgv.blocks[I - start, J - start] = out_g.reshape(shape)

    @staticmethod
    def _negate_q_across_comm(x, nq, lo, hi, xp):
        """``q -> -q`` on an axis that ``comm.q`` has sectioned.

        The negation is not local: it maps a rank's own section onto the
        section of whichever rank owns the negated momenta, so it needs the
        whole axis. Rebuild it by summing zero-padded contributions -- the
        sections are disjoint, so the sum IS the gather, and it needs no
        padding arithmetic for an uneven split (``all_gather_v`` would pad to
        the widest section and the trim would have to be undone by hand).

        The full-axis temporary is transient and one leg wide; the buffers
        themselves stay sectioned, which is where the memory is.
        """
        full = xp.zeros(x.shape[:1] + (nq,) + x.shape[2:], dtype=x.dtype)
        full[:, lo:hi] = x
        recv = xp.empty_like(full)
        ranks.q.all_reduce(xp.ascontiguousarray(full), recv, op="sum")
        neg = (-xp.arange(nq)) % nq
        return xp.take(recv, neg, axis=1)[:, lo:hi]

    @staticmethod
    def _rotate_q_buffers(buf, bounds, rank, size, step, xp):
        """One hop of the leg-B ring: send to ``rank-1``, receive ``rank+1``.

        The next slice a rank needs is the one its successor currently holds,
        so the ring turns one way only. Buffers are sized from the SOURCE
        slice, which generally differs from the local one when ``nq`` does not
        divide ``P_q`` -- assuming they match is the obvious way to get a
        truncated message here.
        """
        dst = (rank - 1) % size
        src = (rank + 1) % size
        nxt = (rank + step + 1) % size
        width = bounds[nxt + 1] - bounds[nxt]
        out = []
        for d in buf:
            new = {}
            for key in sorted(d):
                val = d[key]
                shape = val.shape[:1] + (width,) + val.shape[2:]
                recv = xp.empty(shape, dtype=val.dtype)
                ranks.q.send_recv(xp.ascontiguousarray(val), dst, recv, src)
                new[key] = recv
            out.append(new)
        return tuple(out)

    def _contract_dense_q(
        self, owned, qdm, qv, q_lo, q_hi, nq, nk, n_tau, dtype,
        gl_q, gg_q, glr_q, ggr_q, _fold_l, _fold_g,
        stlv, stgv, start, xp,
        fast_now=False, verify_now=False, stxv=None, release=None,
        a_slice=None, b_slice=None, a_off=0, b_off=0, legs_b=None,
        accumulate=False,
    ):
        """DENSE coupled-q vertex-pair contraction (the reference path).

        Consumes the dense q-folded vertex dict ``qv`` and the q-flattened
        tau-domain Green's function band dicts, writes the per-(I, J)
        Sigma^{<,>} tau blocks into the stack views. See
        ``_contract_factored_q`` for the tensor-decomposed equivalent.

        ``fast_now`` (sse_greater_from_lesser): 4-ring bosonic tau fold --
        Sigma^> is contracted from the (gg, gg) ring only and its two
        absorption cross terms are the mixed Sigma^< rings of pair (J, I)
        at negated external q, folded downstream (JI/orbital transpose in
        the stack state, tau reversal + q negation in stage 5); the mixed
        rings are accumulated per (I, J, q_ext) and written into ``stxv``.
        ``verify_now``: legacy 6-ring result shipped, the reconstruction
        identity checked in place (single-rank runs only).
        """
        a_lo, a_hi = (0, nq) if a_slice is None else a_slice
        b_lo, b_hi = (0, nq) if b_slice is None else b_slice
        gl_a, gg_a, glr_a, ggr_a = gl_q, gg_q, glr_q, ggr_q
        gl_b, gg_b, glr_b, ggr_b = (
            (gl_q, gg_q, glr_q, ggr_q) if legs_b is None else legs_b)
        cache_key = (q_lo, q_hi, nq, a_lo, a_hi, b_lo, b_hi)
        if getattr(self, "_qtasks_cache_key", None) == cache_key:
            qtasks = self._qtasks_cache
        else:
            bulk_vertex = self._vfactors is not None
            if not bulk_vertex and self._perm_share == "auto":
                bulk_vertex = self._qfold_is_translation_invariant(qv, xp)
                if ranks.rank == 0:
                    print(f"PhPh SSE perm cache: q-folded vertex "
                          f"{'IS' if bulk_vertex else 'is NOT'} translation "
                          f"invariant -> "
                          f"{'offset' if bulk_vertex else 'absolute'} key",
                          flush=True)
            perm_cache: dict[tuple, tuple] = {}
            qtasks = {}
            for iqp in range(a_lo, a_hi):
                for iq2 in range(b_lo, b_hi):
                    iq_ext = (iqp + iq2) % nq
                    if not q_lo <= iq_ext < q_hi:
                        continue
                    phiL = qv.get((iqp, iq2))   # legs (q', q_ext-q')
                    phiR = qv.get((iq2, iqp))   # legs (q_ext-q', q')
                    if phiL is None or phiR is None:
                        continue
                    for (I, J) in owned:
                        for K1, K2, K1p, K2p, _pl, _pr in self._phi_pair_index[(I, J)]:
                            pl = phiL.get((I, K1, K2))
                            pr = phiR.get((J, K2p, K1p))
                            if pl is None or pr is None:
                                continue
                            pkey = (
                                (iqp, iq2, K1 - I, K2 - I, K2p - J, K1p - J)
                                if bulk_vertex
                                else (iqp, iq2, I, K1, K2, J, K2p, K1p)
                            )
                            pre = perm_cache.get(pkey)
                            if pre is None:
                                pre = phi_perms(
                                    xp.conj(xp.asarray(pl)), xp.asarray(pr), xp
                                )
                                perm_cache[pkey] = pre
                            qtasks.setdefault((I, J), []).append(
                                (iq_ext, iqp, iq2, K1, K1p, K2, K2p) + pre)
            self._qtasks_cache_key = cache_key
            self._qtasks_cache = qtasks

        if not getattr(self, "_ring_stats_printed", False) and ranks.rank == 0:
            self._ring_stats_printed = True
            n_tasks = sum(len(t) for t in qtasks.values())
            n_rings = 4 if fast_now else 6
            flops = 0.0
            for tasks in qtasks.values():
                for t in tasks:
                    PL, PR, nI, bK2, nJ = t[7], t[8], t[9], t[10], t[11]
                    bK1 = PL.shape[1]
                    bK2p = PR.shape[0]
                    bK1p = PR.shape[1] // nJ
                    flops += n_rings * 8 * n_tau * (
                        nI * bK2 * bK1 * bK1p
                        + bK2 * bK2p * bK1p * nJ
                        + nI * bK2 * bK1p * nJ
                    )
            self._ring_model_gflop = flops / 1e9
            print(
                f"PhPh SSE ring (coupled-q): pairs={len(qtasks)} "
                f"qtasks={n_tasks} nq={nq} q_local={q_hi - q_lo} "
                f"n_tau={n_tau} model={flops / 1e9:.1f} GFLOP/pass",
                flush=True,
            )

        def _contract_tau_q(lo, hi):
            res = {}
            need_x = fast_now or verify_now
            for (I, J), tasks in qtasks.items():
                bs_I = int(self.block_sizes[I])
                bs_J = int(self.block_sizes[J])
                out_l = xp.zeros((hi - lo, nq, bs_I, bs_J), dtype=dtype)
                out_g = xp.zeros((hi - lo, nq, bs_I, bs_J), dtype=dtype)
                out_x = (xp.zeros((hi - lo, nq, bs_I, bs_J), dtype=dtype)
                         if need_x else None)
                out_t56 = (xp.zeros((hi - lo, nq, bs_I, bs_J), dtype=dtype)
                           if verify_now else None)
                for (iq_ext, iqp, iq2, K1, K1p, K2, K2p, *pre) in tasks:
                    gla = gl_a[(K1, K1p)][lo:hi, iqp - a_off]
                    glb = gl_b[(K2, K2p)][lo:hi, iq2 - b_off]
                    ggra = ggr_a[(K1, K1p)][lo:hi, iqp - a_off]
                    ggrb = ggr_b[(K2, K2p)][lo:hi, iq2 - b_off]
                    tx = t56 = None
                    if need_x:
                        PL, PR, nI, bK2, nJ = pre
                        tx = (ring_contract_pre(
                                  PL, PR, nI, bK2, nJ, gla, ggrb, xp)
                              + ring_contract_pre(
                                  PL, PR, nI, bK2, nJ, ggra, glb, xp))
                        sl = ring_contract_pre(
                            PL, PR, nI, bK2, nJ, gla, glb, xp) + tx
                    else:
                        sl = _fold_l(pre, gla, glb, ggra, ggrb)
                    gga = gg_a[(K1, K1p)][lo:hi, iqp - a_off]
                    ggb = gg_b[(K2, K2p)][lo:hi, iq2 - b_off]
                    if fast_now:
                        PL, PR, nI, bK2, nJ = pre
                        sg = ring_contract_pre(
                            PL, PR, nI, bK2, nJ, gga, ggb, xp)
                    elif verify_now:
                        PL, PR, nI, bK2, nJ = pre
                        t56 = (ring_contract_pre(
                                   PL, PR, nI, bK2, nJ, gga,
                                   glr_b[(K2, K2p)][lo:hi, iq2 - b_off], xp)
                               + ring_contract_pre(
                                   PL, PR, nI, bK2, nJ,
                                   glr_a[(K1, K1p)][lo:hi, iqp - a_off], ggb, xp))
                        sg = ring_contract_pre(
                            PL, PR, nI, bK2, nJ, gga, ggb, xp) + t56
                    else:
                        sg = _fold_g(
                            pre,
                            gga,
                            ggb,
                            glr_a[(K1, K1p)][lo:hi, iqp - a_off],
                            glr_b[(K2, K2p)][lo:hi, iq2 - b_off],
                        )
                    out_l[:, iq_ext] += sl
                    out_g[:, iq_ext] += sg
                    if tx is not None:
                        out_x[:, iq_ext] += tx
                    if t56 is not None:
                        out_t56[:, iq_ext] += t56
                res[(I, J)] = (out_l, out_g, out_x, out_t56)
            return res

        def _write_q(I, J, out_l, out_g, out_x=None):
            """Write one slice pair's contribution into the Sigma stack.

            ACCUMULATES when ``accumulate`` is set, which is what the
            q-distributed rotation needs: a rank visits ``P_q`` slice pairs
            and each contributes to the same external momenta, so a plain
            assignment would keep only the last step. Assignment stays the
            default so the replicated path -- one pass over the whole axis --
            is untouched.
            """
            bs_I = int(self.block_sizes[I])
            bs_J = int(self.block_sizes[J])
            blk_shape = (n_tau,) + tuple(nk) + (bs_I, bs_J)
            i, j = I - start, J - start
            if accumulate:
                stlv.blocks[i, j] = stlv.blocks[i, j] + out_l.reshape(blk_shape)
                stgv.blocks[i, j] = stgv.blocks[i, j] + out_g.reshape(blk_shape)
                if out_x is not None:
                    stxv.blocks[i, j] = (stxv.blocks[i, j]
                                         + out_x.reshape(blk_shape))
                return
            stlv.blocks[i, j] = out_l.reshape(blk_shape)
            stgv.blocks[i, j] = out_g.reshape(blk_shape)
            if out_x is not None:
                stxv.blocks[i, j] = out_x.reshape(blk_shape)

        def _scatter_add_q(out, idx, vals):
            outT = out.transpose(1, 0, 2, 3)          # view
            if xp is np:
                np.add.at(outT, idx, vals)
            else:
                import cupyx
                cupyx.scatter_add(outT.real, idx, vals.real)
                cupyx.scatter_add(outT.imag, idx, vals.imag)

        def _contract_tau_q_batched(lo, hi):
            """Same math as _contract_tau_q with the (q', quad) task axis
            flattened into strided-batched GEMMs (per (I, J), grouped by
            ring shape). Reduction order differs from the task loop only
            within the scatter-add -> gate at ~1e-12, not bit."""
            w = hi - lo
            def _stack(d):
                keys = sorted(d.keys())
                shapes = {d[k].shape[-2:] for k in keys}
                if len(shapes) > 1:
                    return None, None
                return xp.stack([xp.ascontiguousarray(d[k]) for k in keys]), \
                    {k: i for i, k in enumerate(keys)}
            GL, gl_i = _stack(gl_a)
            GG, gg_i = _stack(gg_a)
            GGR, ggr_i = _stack(ggr_a)
            GLR, _glr_i = (GG, gg_i) if fast_now else _stack(glr_a)
            if legs_b is None:
                GLb, GGb, GGRb, GLRb, glb_i = GL, GG, GGR, GLR, gl_i
            else:
                GLb, glb_i = _stack(gl_b)
                GGb, _ = _stack(gg_b)
                GGRb, _ = _stack(ggr_b)
                GLRb = GGb if fast_now else _stack(glr_b)[0]
            if any(x is None for x in
                   (GL, GG, GGR, GLR, GLb, GGb, GGRb, GLRb)):
                return _contract_tau_q(lo, hi)
            if release is not None:
                release()

            need_x = fast_now or verify_now
            res = {}
            for (I, J), tasks in qtasks.items():
                bs_I = int(self.block_sizes[I])
                bs_J = int(self.block_sizes[J])
                out_l = xp.zeros((w, nq, bs_I, bs_J), dtype=dtype)
                out_g = xp.zeros((w, nq, bs_I, bs_J), dtype=dtype)
                out_x = (xp.zeros((w, nq, bs_I, bs_J), dtype=dtype)
                         if need_x else None)
                out_t56 = (xp.zeros((w, nq, bs_I, bs_J), dtype=dtype)
                           if verify_now else None)
                # group tasks by ring shape (uniform blocks -> one group)
                groups: dict[tuple, list] = {}
                for t in tasks:
                    PL, PR, nI, bK2, nJ = t[7], t[8], t[9], t[10], t[11]
                    groups.setdefault(
                        (PL.shape, PR.shape, nI, bK2, nJ), []).append(t)
                for (shpL, shpR, nI, bK2, nJ), ts in groups.items():
                    Tn = len(ts)
                    bK1 = shpL[1]
                    bK1p = shpR[1] // nJ
                    per_tt = 16 * (
                        2 * bK1 * bK1p + shpL[0] * bK1p
                        + bK2 * shpR[1] + 2 * nI * nJ)
                    c_pref = min(Tn, 256)
                    wt = int(max(16, min(
                        w, self._tau_chunk_bytes // max(per_tt * c_pref, 1))))
                    C = max(16, min(Tn, self._tau_chunk_bytes
                                    // max(per_tt * wt, 1)))
                    iqe = xp.asarray([t[0] for t in ts], dtype=xp.int64)
                    a_id = xp.asarray([gl_i[(t[3], t[4])] for t in ts],
                                      dtype=xp.int64)
                    b_id = xp.asarray([glb_i[(t[5], t[6])] for t in ts],
                                      dtype=xp.int64)
                    qa = xp.asarray([t[1] for t in ts], dtype=xp.int64) - a_off
                    qb = xp.asarray([t[2] for t in ts], dtype=xp.int64) - b_off
                    for c0 in range(0, Tn, C):
                        c1 = min(c0 + C, Tn)
                        PLc = xp.stack([t[7] for t in ts[c0:c1]])
                        PRc = xp.stack([t[8] for t in ts[c0:c1]])
                        PLc = PLc[:, None]            # (C,1,nIbK2,bK1)
                        PRc = PRc[:, None]            # (C,1,bK2p,bK1pnJ)
                        ai, bi = a_id[c0:c1], b_id[c0:c1]
                        qi, q2 = qa[c0:c1], qb[c0:c1]

                        for w0 in range(0, w, wt):
                            w1 = min(w0 + wt, w)
                            _tau_ix = xp.arange(lo + w0, lo + w1)[None, :]

                            def _legs(A, B):
                                Ga = A[ai[:, None], _tau_ix, qi[:, None]]
                                Gb = B[bi[:, None], _tau_ix, q2[:, None]]
                                return Ga, Gb

                            def _ring(Ga, Gb):
                                Tm = PLc @ Ga         # (C,wt,nIbK2,bK1p)
                                Um = Gb @ PRc         # (C,wt,bK2,bK1pnJ)
                                return (
                                    Tm.reshape(c1 - c0, w1 - w0,
                                               nI, bK2 * bK1p)
                                    @ Um.reshape(c1 - c0, w1 - w0,
                                                 bK2 * bK1p, nJ))

                            tx = t56 = None
                            if need_x:
                                Ga, Gb = _legs(GL, GGRb)
                                tx = _ring(Ga, Gb)
                                Ga, Gb = _legs(GGR, GLb)
                                tx = tx + _ring(Ga, Gb)
                                Ga, Gb = _legs(GL, GLb)
                                sl = _ring(Ga, Gb) + tx
                            else:
                                sl = None
                                for A, B in ((GL, GLb), (GL, GGRb), (GGR, GLb)):
                                    Ga, Gb = _legs(A, B)
                                    s = _ring(Ga, Gb)
                                    sl = s if sl is None else sl + s
                            if fast_now:
                                Ga, Gb = _legs(GG, GGb)
                                sg = _ring(Ga, Gb)
                            elif verify_now:
                                Ga, Gb = _legs(GG, GLRb)
                                t56 = _ring(Ga, Gb)
                                Ga, Gb = _legs(GLR, GGb)
                                t56 = t56 + _ring(Ga, Gb)
                                Ga, Gb = _legs(GG, GGb)
                                sg = _ring(Ga, Gb) + t56
                            else:
                                sg = None
                                for A, B in ((GG, GGb), (GG, GLRb), (GLR, GGb)):
                                    Ga, Gb = _legs(A, B)
                                    s = _ring(Ga, Gb)
                                    sg = s if sg is None else sg + s
                            _scatter_add_q(
                                out_l[w0:w1], iqe[c0:c1],
                                sl.astype(dtype, copy=False))
                            _scatter_add_q(
                                out_g[w0:w1], iqe[c0:c1],
                                sg.astype(dtype, copy=False))
                            if tx is not None:
                                _scatter_add_q(
                                    out_x[w0:w1], iqe[c0:c1],
                                    tx.astype(dtype, copy=False))
                            if t56 is not None:
                                _scatter_add_q(
                                    out_t56[w0:w1], iqe[c0:c1],
                                    t56.astype(dtype, copy=False))
                res[(I, J)] = (out_l, out_g, out_x, out_t56)
            return res

        pool, n_threads = ring_pool()
        nt = min(n_threads, max(1, n_tau // self._tau_min_chunk))  # see nq==1
        if self._dense_q_batched:
            res = _contract_tau_q_batched(0, n_tau)
        elif pool is not None and xp is np and nt > 1:
            bnds = [(i * n_tau // nt, (i + 1) * n_tau // nt)
                    for i in range(nt)]
            chunks = list(pool.map(lambda b: _contract_tau_q(*b), bnds))
            res = {
                ij: tuple(
                    xp.concatenate([c[ij][t] for c in chunks], axis=0)
                    if chunks[0][ij][t] is not None else None
                    for t in range(4)
                )
                for ij in qtasks
            }
        else:
            res = _contract_tau_q(0, n_tau)
        for (I, J), (out_l, out_g, out_x, _t56) in res.items():
            _write_q(I, J, out_l, out_g, out_x if fast_now else None)

        if verify_now:
            rev = (-xp.arange(n_tau)) % n_tau
            qneg = xp.arange(nq).reshape(nk)
            for ax, k in enumerate(nk):
                qneg = xp.take(qneg, (-xp.arange(k)) % k, axis=ax)
            qneg = qneg.reshape(-1)
            worst = worst_rel = 0.0
            for (I, J) in res:
                rec = res[(J, I)][2][rev][:, qneg].swapaxes(-1, -2)
                d = float(xp.max(xp.abs(res[(I, J)][3] - rec)))
                scale = float(xp.max(xp.abs(res[(I, J)][3]))) or 1.0
                worst = max(worst, d)
                worst_rel = max(worst_rel, d / scale)
            self._fold_verify_done += 1
            if ranks.rank == 0:
                print(
                    "PhPh SSE fold-verify (coupled-q) "
                    f"[{self._fold_verify_done}/{self._fold_verify}]"
                    f": max|d|={worst:.3e} rel={worst_rel:.3e} "
                    f"({'OK' if worst_rel < 1e-10 else 'MISMATCH'})",
                    flush=True,
                )

    def _contract_factored_q(
        self, owned, q_lo, q_hi, nq, nk, n_tau, dtype,
        gl_q, gg_q, glr_q, ggr_q, stlv, stgv, start, xp,
    ):
        """Coupled-q contraction with the TENSOR-DECOMPOSED vertex.

        Exact factored equivalent of ``_contract_dense_q`` (same fold terms,
        same left-vertex conjugation -- applied to the row factors inside the
        Grams). The quad sum collapses onto two summed Grams and the
        q'-convolution runs as an FFT; see ``quatrex.phonon.bubble_factored``.
        """
        from quatrex.phonon.bubble_factored import contract_tau_q_factored

        vf = self._vfactors
        if self._micro_pair_index is not None:
            quads_by_pair = {
                pair: [
                    (q.k1, q.k2, q.k1p, q.k2p) for q in quads
                ]
                for group_pair in owned
                for pair, quads in self._micro_pair_index.get(
                    group_pair, {}).items()
            }
            factor_block_sizes = np.full(
                self._micro_layout.n_microblocks,
                self._micro_layout.micro_dof, dtype=int)
        else:
            quads_by_pair = {
                (I, J): [
                    (K1, K2, K1p, K2p)
                    for (K1, K2, K1p, K2p, _pl, _pr)
                    in self._phi_pair_index[(I, J)]
                ]
                for (I, J) in owned
                if (I, J) in self._phi_pair_index
            }
            factor_block_sizes = self.block_sizes
        off_pos = vf.offset_index()
        Dt = xp.asarray(vf.D * vf.lambdas[None, :])
        UB = xp.asarray(vf.UB)
        UC = UB if vf.UB is vf.UC else xp.asarray(vf.UC)
        g_dicts = {"l": gl_q, "g": gg_q, "lr": glr_q, "gr": ggr_q}
        shared = UB is UC or bool(xp.array_equal(UB, UC))

        def _run(lo, hi):
            return contract_tau_q_factored(
                quads_by_pair, factor_block_sizes, tuple(nk), q_lo, q_hi, nq,
                g_dicts, Dt, UB, UC, off_pos, lo, hi, xp, shared, dtype,
            )

        def _write_grouped(result):
            if self._micro_pair_index is None:
                for (I, J), (out_l, out_g) in result.items():
                    bs_I = int(self.block_sizes[I])
                    bs_J = int(self.block_sizes[J])
                    shape = (n_tau,) + tuple(nk) + (bs_I, bs_J)
                    stlv.blocks[I - start, J - start] = out_l.reshape(shape)
                    stgv.blocks[I - start, J - start] = out_g.reshape(shape)
                return

            grouped = {}
            d = self._micro_layout.micro_dof
            for I, J in owned:
                grouped[(I, J)] = (
                    xp.zeros((n_tau, nq, int(self.block_sizes[I]),
                              int(self.block_sizes[J])), dtype=dtype),
                    xp.zeros((n_tau, nq, int(self.block_sizes[I]),
                              int(self.block_sizes[J])), dtype=dtype),
                )
            for (i, j), (out_l, out_g) in result.items():
                I, si = self._micro_layout.locate(i)
                J, sj = self._micro_layout.locate(j)
                grouped[(I, J)][0][..., si, sj] += out_l
                grouped[(I, J)][1][..., si, sj] += out_g
            for (I, J), (out_l, out_g) in grouped.items():
                shape = ((n_tau,) + tuple(nk) +
                         (int(self.block_sizes[I]), int(self.block_sizes[J])))
                stlv.blocks[I - start, J - start] = out_l.reshape(shape)
                stgv.blocks[I - start, J - start] = out_g.reshape(shape)

        pool, n_threads = ring_pool()
        bytes_per_table_tau = (
            nq * Dt.shape[1] ** 2 * xp.dtype(dtype).itemsize
        )
        table_multiplier = 1
        for quads in quads_by_pair.values():
            a_links = {(q[0], q[2]) for q in quads}
            b_links = {(q[1], q[3]) for q in quads}
            n_links = (len(a_links | b_links) if shared
                       else len(a_links) + len(b_links))
            table_multiplier = max(table_multiplier, 4 * n_links + 16)
        bytes_per_tau = bytes_per_table_tau * table_multiplier
        cap = max(1, self._tau_chunk_bytes // max(bytes_per_tau, 1))
        n_chunks = max(1, -(n_tau // -int(cap)))                 # ceil
        if pool is not None and xp is np:
            n_chunks = max(
                n_chunks, min(n_threads, max(1, n_tau // self._tau_min_chunk))
            )

        bnds = [(i * n_tau // n_chunks, (i + 1) * n_tau // n_chunks)
                for i in range(n_chunks)]
        bnds = [(lo, hi) for lo, hi in bnds if hi > lo]

        if len(bnds) == 1:
            _write_grouped(_run(*bnds[0]))
            return

        if pool is not None and xp is np and n_threads > 1:
            chunks = list(pool.map(lambda b: _run(*b), bnds))
        else:
            chunks = [_run(lo, hi) for lo, hi in bnds]
        joined = {
            pair: (
                xp.concatenate([c[pair][0] for c in chunks], axis=0),
                xp.concatenate([c[pair][1] for c in chunks], axis=0),
            )
            for pair in quads_by_pair
        }
        _write_grouped(joined)


