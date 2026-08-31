# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Experimental analytic-pole runtime for the phonon solver."""

import numpy as np
from qttools import xp
from qttools.comm import comm
from qttools.profiling import Profiler
from qttools.toeplitz.toeplitz import get_periodic_superblocks
from qttools.utils.gpu_utils import get_host

profiler = Profiler()


def _q_block(arr, idx):
    """Select one transverse momentum from a contact block."""
    if arr is None:
        return None
    return arr[(slice(None),) + tuple(idx)]


class PoleRuntimeMixin:
    """Opt-in analytic-pole operations for the phonon solver."""

    def _band_edges_for(self, index_slice=()):
        """Lead band edges (THz) for one q, or ``None`` when the gate is off.

        Homogenised from the DYNAMICAL matrix by the same
        ``get_periodic_superblocks`` the OBC applies to the system matrix, so
        the edges are the branch points of the contact the operator carries.
        Taking them from the system matrix instead would mean removing the
        ``z^2 I`` the phonon Dyson operator adds, at a frequency that would
        then have to be chosen.

        Frequency independent and cached per q: ``D`` does not move during the
        SCBA, so this runs once per q per run.
        """
        from quatrex.phonon.experimental.pole.pole_sector import lead_band_edges

        if getattr(self._pole_cfg, "band_edges", "none") != "lead":
            return None
        key = tuple(int(i) for i in index_slice)
        cache = getattr(self, "_band_edge_cache", None)
        if cache is None:
            cache = self._band_edge_cache = {}
        if key in cache:
            return cache[key]

        blocks = self._pole_blocks(self.dynamical_matrix,
                                   index_slice=index_slice)
        if (1, 0) not in blocks:
            # A single-block device has no periodic layer to homogenise.
            cache[key] = None
            return None
        d_10, d_00, d_01 = get_periodic_superblocks(
            a_ii=blocks[(0, 0)], a_ji=blocks[(1, 0)], a_ij=blocks[(0, 1)],
            block_sections=self.block_sections,
        )

        def _one(b):
            a = xp.asarray(b)
            return a.reshape(a.shape[-2:])      # drop any leading singletons

        edges = lead_band_edges(_one(d_00), _one(d_01), _one(d_10))
        if comm.rank == 0 and not cache:
            print(f"  pole sector: {edges.size} lead band edges "
                  f"[{edges.min():.3f}, {edges.max():.3f}] THz feed edge_factor="
                  f"{self._pole_cfg.edge_factor:g}", flush=True)
        cache[key] = edges
        return edges

    def _pole_blocks(self, matrix, index_slice=()):
        """Dense block-tridiagonal view of a DSDBSparse for one stack element.

        ``index_slice`` addresses the TRANSVERSE q axes, which sit at the END
        of the stack shape. The dynamical matrix carries a leading singleton
        where the Keldysh buffers carry frequency, so a bare ``(i, j)`` lands on
        the wrong axes: on a ``(1, 9, 9)`` stack it consumed the singleton and
        one q index, leaving ``(9, 6, 6)`` where a block was wanted, and every
        ``i > 0`` raised "Index 1 is out of bounds for axis 0 with size 1".
        Padding on the LEFT puts the q indices where they belong whatever the
        buffer's leading rank is.
        """
        if index_slice:
            rank = len(getattr(matrix, "global_stack_shape", ()) or ())
            pad = max(0, rank - len(index_slice))
            index_slice = (0,) * pad + tuple(index_slice)
        view = matrix.stack[index_slice] if index_slice else matrix.stack[...]
        n = matrix.num_local_blocks
        out = {}
        for i in range(n):
            for j in range(max(0, i - 1), min(n, i + 2)):
                out[(i, j)] = view.blocks[i, j]
        return out

    @profiler.profile(label="PhononSolver: Positivity", level="default", comm=comm)
    def _check_positivity(self, out: tuple) -> None:
        """Positivity gate on the RECONSTRUCTED TOTAL ``G^{<,>}``.

        ``bubble_positivity.md`` Thm 1-2: the solver stores ``-i G^{<,>} >= 0``,
        and the bubble is a congruence of it. The gate is on the total and
        never on a sector -- ``G_PP`` and ``G_BB`` are separately PSD, but
        ``G_PP + G_PB + G_BP`` is not, so a per-sector check would report
        violations that are not there.

        Off by default (``pole_sector.psd_check``): it costs a batched
        eigen-decomposition per block window. This closes
        ``bubble_positivity.md``'s open item "a production positivity gate
        behind a flag".
        """
        cfg = getattr(self.config.phonon, "pole_sector", None)
        if cfg is None or not getattr(cfg, "psd_check", False):
            return
        from quatrex.phonon.experimental.pole.pole_audit import psd_residual

        n_freq = int(self.local_frequencies.shape[0])
        targets = [("g_lesser", out[0], -1.0), ("g_greater", out[1], -1.0)]
        if self._psd_sigma is not None:
            sl, sg = self._psd_sigma
            targets = [("sigma_lesser", sl, -1.0),
                       ("sigma_greater", sg, -1.0)] + targets
        skip = xp.abs(xp.asarray(self.local_frequencies).real) < 1e-6
        for name, buf, sign in targets:
            rep = psd_residual(
                buf.data.reshape(n_freq, -1), buf.rows, buf.cols,
                self.block_sizes, sign=sign, skip=skip,
            )
            self.psd_report[name] = rep
            if comm.rank == 0:
                flag = "VIOLATION" if rep["worst"] < -self._psd_tol else "ok"
                print(f"  positivity {name:15s} worst={rep['worst']:+.3e} "
                      f"at w[{rep['omega_index']}]  {flag}", flush=True)

    def _pole_frequency_context(self, local_freqs) -> dict:
        """Global grid, local offset and reducer for the pole continuation.

        The continuation sums over the WHOLE frequency axis, which
        ``comm.stack`` splits. Both of its terms are linear in ``Delta`` with
        grid-only coefficients and neither reindexes frequency, so each rank
        can contract its own columns and a single sum-reduction completes the
        result -- no transposition of the distributed buffer, and no
        distributed root finding: every rank ends up with the same operator
        and solves the same poles.

        Serial runs get the identity reducer, so the path is bit-identical to
        the undistributed one rather than merely equivalent.
        """
        if comm.stack.size <= 1:
            return {}

        n_ranks = comm.stack.size
        sizes = np.empty(n_ranks, dtype=np.int64)
        comm.stack.all_gather(np.array([local_freqs.size], dtype=np.int64),
                              sizes)
        sizes = np.asarray(get_host(sizes), dtype=np.int64).ravel()
        offset = int(sizes[:comm.stack.rank].sum())

        n_max = int(sizes.max())
        send = np.zeros(n_max, dtype=float)
        send[:local_freqs.size] = np.asarray(
            get_host(local_freqs), dtype=float).ravel()
        recv = np.empty(n_max * n_ranks, dtype=float)
        comm.stack.all_gather(send, recv)
        recv = np.asarray(get_host(recv), dtype=float).reshape(n_ranks, n_max)
        global_freqs = np.concatenate(
            [recv[r, :int(sizes[r])] for r in range(n_ranks)])

        def _reduce(arr):
            send = xp.ascontiguousarray(arr)
            recv = xp.empty_like(send)
            comm.stack.all_reduce(send, recv, op="sum")
            return recv

        return {"global_freqs": global_freqs, "freq_offset": offset,
                "reduce": _reduce}

    @profiler.profile(label="PhononSolver: Pole sector", level="default", comm=comm)
    def _update_pole_sector(self, sse_lesser, sse_greater,
                            g_retarded=None, g_lesser=None) -> None:
        """Refresh the pole set from the current (mixed) self-energy.

        Runs in the ``"stack"`` distribution state, right after the selected
        solve and before the system matrix is released. Rebuilds
        ``Delta = Sigma^> - Sigma^<`` from the buffers the Dyson operator was
        just built from -- never from a cache, which could drift out of step
        with ``Sigma^R`` across the mixer -- continues it to complex frequency,
        and corrects the previous iterate's poles.

        Staging note: the frequency axis is split across ``comm.stack``, while
        the continuation sums over ALL frequencies. The distributed form
        contracts in the ``"nnz"`` state, where every rank owns the whole axis,
        and is deferred; until it lands this refuses a split stack rather than
        silently continuing an incomplete self-energy.
        """
        if not self._pole_enabled:
            return

        from quatrex.phonon.experimental.pole.pole_probe import delta_from_sigma
        from quatrex.phonon.experimental.pole.pole_sector import PoleSector

        if comm.block.size > 1:
            raise NotImplementedError(
                "pole_sector: block-distributed devices are not supported yet "
                "(the pole solve needs the whole operator on one rank)."
            )

        freqs = np.asarray(get_host(self.local_frequencies), dtype=float)
        if self._pole is None:
            self._pole = PoleSector(
                self._pole_cfg, freqs,
                **self._pole_frequency_context(freqs),
            )

        delta = delta_from_sigma(sse_lesser.data, sse_greater.data)
        if float(xp.abs(delta).max()) == 0.0:
            self.pole_state = None
            return

        nq = 1
        if delta.ndim > 2:
            nq = int(np.prod(delta.shape[1:-1]))
        if nq > 1 and getattr(self._pole_cfg, "extraction_only", False):
            self._census_over_q(delta, sse_lesser)
            self.pole_state = None
            return
        if nq > 1:
            self._update_pole_sector_q(delta, sse_lesser, sse_greater,
                                       g_retarded, g_lesser)
            return

        self._pole.set_operator_context(
            band_edges=self._band_edges_for(),
            delta=delta,
            d_blocks=self._pole_blocks(self.dynamical_matrix),
            obc_left=(self.obc_blocks.retarded[0]
                      if self.obc_blocks.retarded[0] is not None else None),
            obc_right=(self.obc_blocks.retarded[-1]
                       if self.obc_blocks.retarded[-1] is not None else None),
            block_sizes=self.block_sizes,
            rows=sse_lesser.rows,
            cols=sse_lesser.cols,
        )
        with profiler.profile_range("PhononSolver: Pole solve", "default",
                                    comm):
            self.pole_state = self._pole.refresh()
        with profiler.profile_range("PhononSolver: Pole legs", "default",
                                    comm):
            if getattr(self._pole_cfg, "leg", "congruence") == "congruence":
                self._pole_layout = getattr(self._pole, "_layout", None)
                state = self.pole_state
                if self._pole_layout is not None and state is not None \
                        and state.clusters:
                    leg_l, leg_g = self._build_pole_legs(
                        sse_lesser, sse_greater, g_retarded, [state],
                        [self._pole], np.array([0], dtype=int))
                    if leg_l is not None:
                        state.g_pp_lesser = leg_l[0].reshape(
                            sse_lesser.data.shape)
                        state.g_pp_greater = leg_g[0].reshape(
                            sse_greater.data.shape)
                    self._report_pole_registration(
                        [state], g_lesser, state.g_pp_lesser,
                        np.array([0], dtype=int))
            else:
                self._build_pole_keldysh(sse_lesser, sse_greater, g_retarded,
                                         g_lesser)
        if comm.rank == 0 and self.pole_state is not None:
            print(self.pole_state.report(), flush=True)

    def _q_indices(self, nq: int, shape):
        """Which q to solve, honouring ``q_stride`` and ``q_max``."""
        sel = list(range(0, nq, max(1, int(getattr(self._pole_cfg,
                                                   "q_stride", 1)))))
        cap = int(getattr(self._pole_cfg, "q_max", 0) or 0)
        if cap:
            sel = sel[:cap]
        return [(iq, tuple(int(i) for i in np.unravel_index(iq, shape)))
                for iq in sel]

    def _update_pole_sector_q(self, delta, sse_lesser, sse_greater,
                              g_retarded, g_lesser) -> None:
        """Coupled-q, Stage 1: one pole problem per q, one leg per q.

        ``leg="congruence"`` adds no analytic sector -- its whole action is to
        modify the leg the ring convolves, and the ring performs the q fold
        itself, downstream. So the sector is per-q and UNCOUPLED here, and the
        vertex fold the guard in ``set_operator_context`` speaks of is needed
        only where sectors are restored analytically (there
        ``Sigma_q = sum_q' B[G_q', G_{q-q'}]`` pairs pole sets across q). Those
        routes stay refused, by that guard, which still sees only a slice.

        One ``PoleSector`` PER q. The tracker, the promoted set and the epoch
        counter are per-q: a mode at one q has no relation to one at another,
        so a shared tracker would match them across q and either fuse two
        unrelated modes or churn membership every iteration -- exactly what the
        hysteresis exists to prevent.
        """
        from quatrex.phonon.experimental.pole.pole_probe import BlockLayout
        from quatrex.phonon.experimental.pole.pole_sector import PoleSector, refresh_many

        if getattr(self._pole_cfg, "leg", "congruence") != "congruence":
            raise NotImplementedError(
                f"pole_sector: leg={self._pole_cfg.leg!r} restores analytic "
                "sectors beside the ring, and those pair pole sets from "
                "DIFFERENT q (Sigma_q sums over q' and q-q'). That fold is not "
                "built. Use leg='congruence' on a q-resolved device, or "
                "extraction_only=True for a census."
            )

        freqs = np.asarray(get_host(self.local_frequencies), dtype=float)
        shape = tuple(int(k) for k in delta.shape[1:-1])
        nq = int(np.prod(shape))
        d_flat = delta.reshape(delta.shape[0], nq, delta.shape[-1])
        todo = self._q_indices(nq, shape)

        if self._pole_q is None:
            self._pole_q = {}
        acc_l = xp.zeros_like(sse_lesser.data)
        acc_g = xp.zeros_like(sse_greater.data)
        states, promoted = [], 0

        layout = BlockLayout(sse_lesser.rows, sse_lesser.cols,
                             self.block_sizes, band=1)
        sectors = []
        for iq, idx in todo:
            sec = self._pole_q.get(iq)
            if sec is None:
                sec = PoleSector(self._pole_cfg, freqs,
                                 **self._pole_frequency_context(freqs))
                self._pole_q[iq] = sec
            sec.set_operator_context(
                band_edges=self._band_edges_for(idx),
                delta=d_flat[:, iq, :],
                d_blocks=self._pole_blocks(self.dynamical_matrix,
                                           index_slice=idx),
                obc_left=_q_block(self.obc_blocks.retarded[0], idx),
                obc_right=_q_block(self.obc_blocks.retarded[-1], idx),
                block_sizes=self.block_sizes,
                rows=sse_lesser.rows,
                cols=sse_lesser.cols,
                layout=layout,
            )
            sectors.append(sec)

        chunk = int(getattr(self._pole_cfg, "q_batch", 0) or 0) or len(sectors)
        solved = []
        with profiler.profile_range("PhononSolver: Pole solve", "default",
                                    comm):
            for lo in range(0, len(sectors), chunk):
                solved.extend(refresh_many(sectors[lo:lo + chunk]))

        for (iq, idx), st in zip(todo, solved):
            states.append((idx, st))
            if st is not None and st.clusters:
                promoted += st.n_poles

        self._pole_layout = layout
        with profiler.profile_range("PhononSolver: Pole legs", "default", comm):
            leg_l, leg_g = self._build_pole_legs(
                sse_lesser, sse_greater, g_retarded, solved, sectors,
                np.array([iq for iq, _ in todo], dtype=int))
        if leg_l is not None:
            for k, (iq, idx) in enumerate(todo):
                st = solved[k]
                if st is None or not st.clusters:
                    continue
                sel = (slice(None),) + idx
                st.g_pp_lesser = leg_l[k].reshape(acc_l[sel].shape)
                st.g_pp_greater = leg_g[k].reshape(acc_g[sel].shape)
                acc_l[sel] = st.g_pp_lesser
                acc_g[sel] = st.g_pp_greater
        self._report_pole_registration(solved, g_lesser, acc_l,
                                       np.array([iq for iq, _ in todo],
                                                dtype=int))

        total = states[0][1] if states and states[0][1] is not None else None
        if total is None or not any(
                st is not None and st.clusters for _, st in states):
            self.pole_state = None
        else:
            total = next(st for _, st in states if st is not None
                         and st.clusters)
            total.g_pp_lesser, total.g_pp_greater = acc_l, acc_g
            self.pole_state = total
        self.pole_q_states = states
        if comm.rank == 0:
            n_live = sum(1 for _, st in states
                         if st is not None and st.clusters)
            skipped = nq - len(todo)
            tail = f", {skipped} q skipped (q_stride/q_max)" if skipped else ""
            seeded = sum(st.n_seeded for _, st in states if st is not None)
            matched = sum(st.n_matched for _, st in states if st is not None)
            why: dict[str, int] = {}
            for _, st in states:
                if st is None:
                    continue
                for _z, reason in st.rejected:
                    key = reason.split("=")[0].split("(")[0].strip()
                    why[key] = why.get(key, 0) + 1
            hist = "  ".join(f"{k} x{v}" for k, v in
                             sorted(why.items(), key=lambda kv: -kv[1]))
            eno = [v for _, st in states if st is not None
                   for v in st.eps_z_refused if np.isfinite(v)]
            eok = [v for _, st in states if st is not None
                   for v in st.eps_z_accepted if np.isfinite(v)]
            def _pct(v):
                if not v:
                    return "n/a"
                q = np.percentile(np.asarray(v, dtype=float), (50, 90, 100))
                return "  ".join(f"{x:.3g}" for x in q)
            print(f"  pole sector: {len(todo)}/{nq} q solved, {n_live} with "
                  f"poles, {promoted} pole(s) total{tail}", flush=True)
            print(f"    seeded {seeded}, matched-as-promoted {matched}, "
                  f"accepted {promoted}, refused {seeded - promoted}"
                  + (f"  [{hist}]" if hist else ""), flush=True)
            print(f"    eps_z (med/p90/max) accepted {_pct(eok)} | "
                  f"refused {_pct(eno)}   tol {self._pole_cfg.locate_tol:g}"
                  f" in / {self._pole_cfg.locate_tol_out:g} out", flush=True)

    def _census_over_q(self, delta, sse_lesser) -> None:
        """Extraction-only census on a q-resolved device, one q at a time.

        Each q gets its own ``M_q``, so each gets its own solve. Nothing is
        allocated and no vertex is touched, which is why this can run where the
        allocating path cannot.
        """
        shape = tuple(int(k) for k in delta.shape[1:-1])
        nq = int(np.prod(shape))
        d_flat = delta.reshape(delta.shape[0], nq, delta.shape[-1])
        for iq in range(nq):
            idx = tuple(int(i) for i in np.unravel_index(iq, shape))
            try:
                self._pole.set_operator_context(
                    band_edges=self._band_edges_for(idx),
                    delta=d_flat[:, iq, :],
                    d_blocks=self._pole_blocks(self.dynamical_matrix,
                                               index_slice=idx),
                    obc_left=_q_block(self.obc_blocks.retarded[0], idx),
                    obc_right=_q_block(self.obc_blocks.retarded[-1], idx),
                    block_sizes=self.block_sizes,
                    rows=sse_lesser.rows,
                    cols=sse_lesser.cols,
                )
                if comm.rank == 0:
                    print(f"  q {idx}:", flush=True)
                self._pole.refresh()
            except (AttributeError, NameError, ImportError):
                raise
            except Exception as exc:                       # noqa: BLE001
                if comm.rank == 0:
                    print(f"  q {idx}: census failed ({type(exc).__name__}: "
                          f"{exc})", flush=True)

    @staticmethod
    def _q_stack(buf, iq, tail: int):
        """``(Q, n_freq, *tail)`` view of a stacked buffer for the chosen q.

        Every array the sector touches carries the transverse axis in the
        middle, ``(n_freq,) + nk + tail``. This is the batched form of the
        per-q slice the leg builder used to take one q at a time.
        """
        if buf is None:
            return None
        a = xp.asarray(buf)
        n_freq = int(a.shape[0])
        flat = a.reshape((n_freq, -1) + a.shape[a.ndim - tail:])
        return xp.moveaxis(flat[:, iq], 0, 1)

    def _build_pole_legs(self, sse_lesser, sse_greater, g_retarded,
                         states, sectors, iq) -> tuple:
        """The congruence leg for EVERY q and cluster in one pass.

        The per-cluster routines in :mod:`~quatrex.phonon.experimental.pole.pole_congruence` say
        what the leg is, one cluster at a time, and are what this is verified
        against. Driving production through them cost a Python loop of
        ``n_q * n_clusters`` iterations over routines that themselves looped
        over pole columns and pole pairs -- 6.85 million calls and 33 s per
        SCBA iteration on Si, against a bubble of 7.4 s. See
        :mod:`~quatrex.phonon.experimental.pole.pole_legs`.

        Only the ``congruence`` route comes through here. ``congruence_analytic``
        and the superseded ``keldysh`` route flatten each cluster into its own
        partial fractions, which is per-cluster by construction and is not a
        production setting; they keep the reference path.
        """
        from quatrex.phonon.experimental.pole.pole_legs import (
            ClusterBatch, ClusterViews, CoefficientViews,
            congruence_legs, source_fit,
        )

        n_dof = int(np.sum(self.block_sizes))
        per_q = []
        for st, sec in zip(states, sectors):
            legs = sec.bubble_clusters() if st is not None and st.clusters else []
            if st is not None:
                st.legs = legs
            per_q.append(legs)
        if not any(per_q):
            return None, None

        chunk = int(getattr(self._pole_cfg, "q_batch", 0) or 0)
        if not chunk:
            n_p = max((int(c.z.shape[0]) for legs in per_q for c in legs),
                      default=1)
            n_m = max((len(legs) for legs in per_q), default=1)
            per_q_bytes = 16 * 34 * int(self.local_frequencies.shape[0]) * \
                max(n_dof, 1) * n_p * n_m
            chunk = max(1, self._POLE_LEG_BUDGET // max(per_q_bytes, 1))
        if chunk < len(states):
            if comm.rank == 0:
                print(f"  pole sector: leg build split into "
                      f"{-(-len(states) // chunk)} chunks of {chunk} q "
                      f"(memory budget)", flush=True)
            out_l, out_g = [], []
            for lo in range(0, len(states), chunk):
                hi = min(lo + chunk, len(states))
                part = self._build_pole_legs_chunk(
                    sse_lesser, sse_greater, g_retarded, states[lo:hi],
                    per_q[lo:hi], iq[lo:hi], n_dof)
                if part[0] is None:
                    w = int(self.local_frequencies.shape[0])
                    nnz = int(self._pole_layout.nnz)
                    part = (xp.zeros((hi - lo, w, nnz), dtype=xp.complex128),
                            xp.zeros((hi - lo, w, nnz), dtype=xp.complex128))
                out_l.append(part[0])
                out_g.append(part[1])
            return xp.concatenate(out_l), xp.concatenate(out_g)
        return self._build_pole_legs_chunk(sse_lesser, sse_greater, g_retarded,
                                           states, per_q, iq, n_dof)

    # Peak working set the leg build may allocate before it cuts the q axis.
    _POLE_LEG_BUDGET = 4 << 30

    def _build_pole_legs_chunk(self, sse_lesser, sse_greater, g_retarded,
                               states, per_q, iq, n_dof) -> tuple:
        """One memory-sized slice of :meth:`_build_pole_legs`."""
        from quatrex.phonon.experimental.pole.pole_legs import (
            ClusterBatch, ClusterViews, CoefficientViews,
            congruence_legs, source_fit,
        )

        if not any(per_q):
            return None, None
        batch = ClusterBatch.from_clusters(per_q, n_dof)
        omega = xp.asarray(self.local_frequencies, dtype=float)
        widths = xp.asarray(self.local_frequency_weights, dtype=float)
        gr = self._q_stack(g_retarded.data, iq, 1)
        last_block = len(self.block_sizes) - 1

        out = {}
        for tag, buf, lo, hi in (
            ("l", sse_lesser, self.obc_blocks.lesser[0], self.obc_blocks.lesser[-1]),
            ("g", sse_greater, self.obc_blocks.greater[0], self.obc_blocks.greater[-1]),
        ):
            corners = ((self._q_stack(lo, iq, 2), 0),
                       (self._q_stack(hi, iq, 2), last_block))
            out[tag] = congruence_legs(
                batch, self._pole_layout, self._q_stack(buf.data, iq, 1), gr,
                corners, omega, widths)

        fit_l = source_fit(batch, out["l"][1], omega)
        fit_g = source_fit(batch, out["g"][1], omega)
        fit = xp.maximum(fit_l, fit_g)
        fit_host = np.asarray(get_host(fit))

        sizes = [[int(cl.z.shape[0]) for cl in legs] for legs in per_q]
        for k, (st, legs) in enumerate(zip(states, per_q)):
            if st is None or not legs:
                continue
            st.source_lesser = ClusterViews(out["l"][1][k], sizes[k])
            st.source_greater = ClusterViews(out["g"][1][k], sizes[k])
            st.c_lesser = CoefficientViews([c[k] for c in out["l"][2]],
                                           sizes[k])
            st.c_greater = CoefficientViews([c[k] for c in out["g"][2]],
                                            sizes[k])
            st.source_fit = list(fit_host[k, :len(legs)])

        if comm.rank == 0:
            over = np.argwhere(fit_host > self._pole_cfg.source_fit_tol)
            for k, m in over:
                if k < len(per_q) and m < len(per_q[k]):
                    print(f"  pole sector: cluster {per_q[k][m].label} source "
                          f"varies by {fit_host[k, m]:.2e} across its window "
                          f"(tol {self._pole_cfg.source_fit_tol:.2e}); the "
                          "analytic source model is not justified there",
                          flush=True)
        return out["l"][0].sum(axis=1), out["g"][0].sum(axis=1)

    def _report_pole_registration(self, states, g_lesser, leg, iq) -> None:
        """Where the promoted poles sit INSIDE their cells, over the whole set.

        This is the control parameter of the bubble's registration error, and
        nothing measured it before. An exactly cell-averaged leg still places
        all of a line's weight at the cell CENTRE, so the combination frequency
        Re(z_a + z_b) is displaced by up to a full cell and the ring splits the
        peak between two bins. Measured
        (test_cell_averaged_legs_do_not_fix_the_bubble_registration),
        ring/exact at the combination frequency:

            offset 0.00   1.004      offset 0.25   1.79
            offset 0.50   0.536

        and it gets WORSE with h/gamma, not better. The congruence route's
        accuracy is therefore an accident of registration unless this is small.

        Reported once for the whole q axis rather than once per q: it is a
        property of the promoted SET.
        """
        if comm.rank != 0 or leg is None:
            return
        from quatrex.phonon.experimental.pole.pole_audit import pole_pair_weight

        w_host = np.asarray(get_host(self.local_frequencies), dtype=float)
        hw = np.asarray(get_host(self.local_frequency_weights), dtype=float)
        z = np.concatenate(
            [np.asarray(get_host(cl.z)).ravel()
             for st in states if st is not None
             for cl in (st.legs or [])] or [np.zeros(0, dtype=complex)])
        z = z[np.real(z) >= 0.0]
        if z.size == 0:
            return
        k = np.argmin(np.abs(w_host[None, :] - np.real(z)[:, None]), axis=1)
        pole_cells = np.zeros(w_host.size, dtype=bool)
        pole_cells[k] = True
        good = hw[k] > 0.0
        off_worst = float(np.max(np.abs(np.real(z)[good] - w_host[k[good]])
                                 / hw[k[good]])) if good.any() else 0.0

        if g_lesser is not None:
            gl = (self._q_stack(g_lesser.data, iq, 1)
                  - xp.asarray(leg).reshape(iq.size, w_host.size, -1))
            norms = np.linalg.norm(np.asarray(get_host(gl)), axis=2).max(axis=0)
            pw = pole_pair_weight(norms, pole_cells, freqs=w_host,
                                  skip=np.abs(w_host) < 1e-6)
        else:
            pw = {"mean": float("nan"), "worst": float("nan"),
                  "omega": float("nan")}
        print(f"  pole registration: worst sub-cell offset {off_worst:.3f} "
              f"cells (0 = on a grid point, 0.5 = on a cell boundary); "
              f"pole-cell PAIRS carry {100 * pw['mean']:.3g}% of the ring's "
              f"weight, up to {100 * pw['worst']:.3g}% at "
              f"w={pw['omega']:.2f}", flush=True)

    def _build_pole_keldysh(self, sse_lesser, sse_greater,
                            g_retarded=None, g_lesser=None, q_idx=(),
                            sector=None) -> None:
        """Project the Keldysh source and reduce ``G_PP`` onto the pattern.

        Done here, in the ``"stack"`` state, because this is where the contact
        blocks live: ``Sigma_tot`` is the scattering self-energy that entered
        the Dyson solve PLUS both contacts, and it is the contact part that
        drives the device. Everything stays on the stored sparsity pattern --
        the dense intermediate would be ``(n_omega, n_dof, n_dof)``.
        """
        from quatrex.phonon.experimental.pole.pole_bridge import (
            add_contact_source, pole_keldysh_pf_sparse, project_source_sparse,
            source_at_poles, source_variation,
        )
        from quatrex.phonon.experimental.pole.pole_congruence import (
            apply_sparse, background_coefficients, coefficients_at_poles,
            fit_residual, partial_fraction_legs, pf_leg_sample,
            remainder_resolution, residue_sum, sector_cell_average,
            sector_grid_sample,
        )

        state = self.pole_state
        if state is None or not state.clusters:
            return
        freqs = xp.asarray(self.local_frequencies, dtype=float)
        rows, cols = sse_lesser.rows, sse_lesser.cols

        def _q(a):
            """This q's slice of a stacked array; identity when nq == 1.

            Every array the sector touches carries the transverse axis in the
            middle, ``(n_freq,) + nk + (nnz,)``, and the pole problem is
            independent per q -- so slicing here is the whole of coupled-q for
            this route. The contact blocks carry it too, and dropping them is
            not a small error: they are what drives the device.
            """
            if not q_idx or a is None:
                return a
            return a[(slice(None),) + tuple(q_idx)]

        sl = _q(sse_lesser.data).reshape(freqs.shape[0], -1)
        sg = _q(sse_greater.data).reshape(freqs.shape[0], -1)
        last = int(np.sum(self.block_sizes[:-1]))

        state.legs = (sector or self._pole).bubble_clusters()
        acc_l = acc_g = None
        _leg = getattr(self._pole_cfg, "leg", "congruence")
        analytic = _leg == "congruence_analytic"
        congruence = _leg.startswith("congruence")
        if congruence:
            if g_retarded is None:
                raise ValueError(
                    "pole_sector.leg='congruence' needs G^R: the retarded "
                    "background B^R_k = G^R_k - U D(w_k) V^dagger is what the "
                    "regular leg is made of.")
            n_dof = int(np.sum(self.block_sizes))
            gr = _q(g_retarded.data).reshape(freqs.shape[0], -1)
            corners_l = ((_q(self.obc_blocks.lesser[0]), 0),
                         (_q(self.obc_blocks.lesser[-1]), last))
            corners_g = ((_q(self.obc_blocks.greater[0]), 0),
                         (_q(self.obc_blocks.greater[-1]), last))
            cell_widths = xp.asarray(self.local_frequency_weights, dtype=float)
        for cl in state.legs:
            s_l = project_source_sparse(sl, rows, cols, cl.v)
            s_g = project_source_sparse(sg, rows, cols, cl.v)
            for corner_l, corner_g, off in (
                (_q(self.obc_blocks.lesser[0]),
                 _q(self.obc_blocks.greater[0]), 0),
                (_q(self.obc_blocks.lesser[-1]),
                 _q(self.obc_blocks.greater[-1]), last),
            ):
                if corner_l is not None:
                    s_l = add_contact_source(s_l, corner_l, cl.v, off)
                if corner_g is not None:
                    s_g = add_contact_source(s_g, corner_g, cl.v, off)
            eps_fit = max(source_variation(s_l, freqs, cl),
                          source_variation(s_g, freqs, cl))
            state.source_fit.append(eps_fit)
            if eps_fit > self._pole_cfg.source_fit_tol and comm.rank == 0:
                print(f"  pole sector: cluster {cl.label} source varies by "
                      f"{eps_fit:.2e} across its window (tol "
                      f"{self._pole_cfg.source_fit_tol:.2e}); the analytic "
                      "source model is not justified there", flush=True)
            state.source_lesser.append(s_l)
            state.source_greater.append(s_g)
            if congruence:
                for src, corn, c_out, g_out in (
                    (sl, corners_l, state.c_lesser, "l"),
                    (sg, corners_g, state.c_greater, "g"),
                ):
                    sv = apply_sparse(src, rows, cols, cl.v, n_dof,
                                      corners=corn)
                    co = background_coefficients(
                        cl, freqs, sv,
                        apply_sparse(gr, rows, cols, sv, n_dof))
                    c_out.append(co)
                    if analytic:
                        frozen = coefficients_at_poles(cl, freqs, co)
                        zeta, p_row, q_col = partial_fraction_legs(cl, frozen)
                        (state.pf_lesser if g_out == "l"
                         else state.pf_greater).append((zeta, p_row, q_col))
                        state.residue_sum.append(residue_sum(p_row, q_col))
                        state.mixed_fit.append((
                            fit_residual(co[1], freqs, cl),
                            remainder_resolution(co[1], frozen[1], freqs, cl),
                        ))
                        smp = pf_leg_sample(zeta, p_row, q_col, freqs,
                                            rows, cols)
                    else:
                        smp = (sector_grid_sample(cl, freqs, co, rows, cols)
                               - sector_cell_average(cl, freqs, co, rows,
                                                     cols, cell_widths))
                    if g_out == "l":
                        acc_l = smp if acc_l is None else acc_l + smp
                    else:
                        acc_g = smp if acc_g is None else acc_g + smp
                continue
            g_l = pole_keldysh_pf_sparse(
                freqs, cl, source_at_poles(s_l, freqs, cl), rows, cols)
            g_g = pole_keldysh_pf_sparse(
                freqs, cl, source_at_poles(s_g, freqs, cl), rows, cols)
            acc_l = g_l if acc_l is None else acc_l + g_l
            acc_g = g_g if acc_g is None else acc_g + g_g
        if acc_l is None or acc_g is None:
            state.g_pp_lesser = state.g_pp_greater = None
            return
        state.g_pp_lesser = acc_l.reshape(_q(sse_lesser.data).shape)
        state.g_pp_greater = acc_g.reshape(_q(sse_greater.data).shape)
        if congruence and comm.rank == 0:
            from quatrex.phonon.experimental.pole.pole_audit import pole_pair_weight

            w_host = np.asarray(get_host(freqs), dtype=float)
            hw = np.asarray(get_host(self.local_frequency_weights), dtype=float)
            off_worst = 0.0
            pole_cells = np.zeros(w_host.size, dtype=bool)
            for cl in state.legs:
                for z in np.asarray(get_host(cl.z)):
                    x = float(np.real(z))
                    if x < 0.0:
                        continue
                    k = int(np.argmin(np.abs(w_host - x)))
                    pole_cells[k] = True
                    hk = float(hw[k]) if k < hw.size else 0.0
                    if hk > 0.0:
                        off_worst = max(off_worst, abs(x - w_host[k]) / hk)
            if g_lesser is not None:
                gl = (_q(g_lesser.data).reshape(w_host.size, -1)
                      - acc_l.reshape(w_host.size, -1))
                pw = pole_pair_weight(
                    np.linalg.norm(np.asarray(get_host(gl)), axis=1),
                    pole_cells, freqs=w_host, skip=np.abs(w_host) < 1e-6)
            else:
                pw = {"mean": float("nan"), "worst": float("nan"),
                      "omega": float("nan")}
            print(f"  pole registration: worst sub-cell offset "
                  f"{off_worst:.3f} cells (0 = on a grid point, 0.5 = on a "
                  f"cell boundary); pole-cell PAIRS carry "
                  f"{100 * pw['mean']:.3g}% of the ring's weight, up to "
                  f"{100 * pw['worst']:.3g}% at w={pw['omega']:.2f}",
                  flush=True)
        if analytic and comm.rank == 0:
            tail = max(state.residue_sum) if state.residue_sum else 0.0
            fits = [a for a, _ in state.mixed_fit] or [0.0]
            regs = [b for _, b in state.mixed_fit] or [0.0]
            flag = ("" if max(regs) <= self._pole_cfg.source_fit_tol
                    else f"  eps_reg ABOVE source_fit_tol "
                         f"({self._pole_cfg.source_fit_tol:.2e})")
            print(f"  pole analytic leg: eps_tail={tail:.3e}  "
                  f"eps_fit={max(fits):.3e}  eps_reg={max(regs):.3e}{flag}",
                  flush=True)

    def _report_subcell(self, out) -> None:
        """Is the reconstruction physical BETWEEN grid points?

        The sectors act on ``G~_h(w) = P(w) + R_k``, not on ``G``. That equals
        ``G`` at the cell centres and nowhere else, and ``R_k = G - P`` is a
        DIFFERENCE of PSD objects. Offline, a crude pole model gives
        ``lambda_min = -1.000`` five percent of a cell off centre while the
        true ``G`` stays at ``+2.3e-02``; an EXACT residue keeps it healthy.
        So this measures whether the production pole model is good enough, and
        it is measured rather than assumed.

        Report only: the threshold at which a pole should be refused is not
        yet established, and gating on a guess would hide the answer.
        """
        from quatrex.phonon.experimental.pole.pole_audit import psd_residual, subcell_positivity
        from quatrex.phonon.experimental.pole.pole_bridge import (
            pole_keldysh_pf_sparse, source_at_poles,
        )
        from quatrex.phonon.experimental.pole.pole_keldysh import pole_retarded

        cfg = getattr(self.config.phonon, "pole_sector", None)
        if cfg is None or not getattr(cfg, "psd_check", False):
            return
        state = self.pole_state
        if state is None or not state.legs:
            return
        if out[0].data.ndim > 2:
            if comm.rank == 0:
                print("  ring leg positivity: skipped (q-resolved; the gate is "
                      "per-q and not written)", flush=True)
            return

        if getattr(cfg, "leg", "congruence") == "congruence":
            rows, cols = out[0].rows, out[0].cols
            n_freq = int(self.local_frequencies.shape[0])
            w_host = np.asarray(get_host(self.local_frequencies), dtype=float)
            skip = xp.asarray(np.abs(w_host) < 1e-6)
            near = np.zeros(n_freq, dtype=bool)
            for cl in state.legs:
                for z in np.asarray(get_host(cl.z)):
                    x = float(np.real(z))
                    if x < 0.0:
                        continue
                    k = int(np.argmin(np.abs(w_host - x)))
                    near[max(0, k - 2):min(n_freq, k + 3)] = True
            skip_far = xp.asarray((np.abs(w_host) < 1e-6) | ~near)
            rep_all = {}
            for name, got, pp in (("lesser", out[0], state.g_pp_lesser),
                                  ("greater", out[1], state.g_pp_greater)):
                raw = got.data.reshape(n_freq, -1)
                for tag, leg in ((name, raw - pp.reshape(n_freq, -1)),
                                 (f"{name}_control", raw)):
                    rep_all[tag] = psd_residual(
                        leg, rows, cols, self.block_sizes, sign=-1.0, skip=skip)
                    rep_all[f"{tag}_poles"] = psd_residual(
                        leg, rows, cols, self.block_sizes, sign=-1.0,
                        skip=skip_far)
            self.psd_report["ring_leg"] = rep_all
            if comm.rank == 0:
                for name in ("lesser", "greater"):
                    a, b = rep_all[name], rep_all[f"{name}_control"]
                    scale = max(abs(a["worst"]), abs(b["worst"]), 1e-300)
                    same = ("  [== control: gate is blind]"
                            if abs(a["worst"] - b["worst"]) <= 1e-3 * scale
                            else "")
                    pa = rep_all[f"{name}_poles"]
                    pb = rep_all[f"{name}_control_poles"]
                    print(f"  ring leg positivity {name:8s} "
                          f"worst={a['worst']:+.3e} at w[{a['omega_index']}]"
                          f"   pole-off control={b['worst']:+.3e}{same}"
                          f"   | in pole cells {pa['worst']:+.3e} vs "
                          f"{pb['worst']:+.3e}", flush=True)
            return
        rows, cols = out[0].rows, out[0].cols
        freqs = xp.asarray(self.local_frequencies, dtype=float)
        g_l = out[0].data.reshape(freqs.shape[0], -1)
        w = np.asarray(get_host(freqs), dtype=float)
        centres = np.array([int(np.argmin(np.abs(w - float(np.real(z)))))
                            for cl in state.legs for z in np.asarray(
                                get_host(cl.z)) if float(np.real(z)) >= 0.0])
        if centres.size == 0:
            return

        def _pole_at(omega):
            acc = None
            for cl, s_l in zip(state.legs, state.source_lesser):
                v = pole_keldysh_pf_sparse(
                    omega, cl, source_at_poles(s_l, freqs, cl), rows, cols)
                acc = v if acc is None else acc + v
            return acc

        rep = subcell_positivity(
            g_l, state.g_pp_lesser.reshape(freqs.shape[0], -1), _pole_at,
            freqs, rows, cols, self.block_sizes, centres=centres, window=1)
        self.psd_report["subcell"] = rep

        cong = None
        try:
            from quatrex.phonon.experimental.pole.pole_audit import subcell_congruence

            def _pole_ret_at(omega):
                acc = None
                for cl in state.legs:
                    v = pole_retarded(omega, cl)
                    acc = v if acc is None else acc + v
                return acc[:, rows, cols]

            cong = subcell_congruence(
                out[2].data.reshape(freqs.shape[0], -1),
                self._psd_sigma_lesser, _pole_ret_at, freqs, rows, cols,
                centres=centres)
            self.psd_report["subcell_congruence"] = cong
        except NotImplementedError as exc:
            if comm.rank == 0:
                print(f"  subcell congruence: skipped ({exc})", flush=True)

        if comm.rank == 0:
            tail = ("" if cong is None
                    else f"   congruence worst={cong['worst']:+.3e}")
            print(f"  subcell positivity: worst={rep['worst']:+.3e} at "
                  f"w[{rep['worst_centre']}], at-centres="
                  f"{rep['at_centres']:+.3e}{tail}", flush=True)

