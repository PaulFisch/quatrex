# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Communication and buffer support for three-phonon contractions."""

import numpy as np
from qttools import NDArray, sparse, xp
from qttools.comm import comm as ranks
from qttools.datastructures import DSDBSparse

_SIGMA_TAU_SLOTS = frozenset({2, 3})


class ContractionSupportMixin:
    """Layout, communication, and buffer helpers."""

    def _fold_reversed_legs(self, legs: tuple, g: DSDBSparse) -> None:
        """Apply the exact bosonic ji-transpose ``X_ij -> X_ji`` to the given
        stack-state buffers in place, at ANY ``comm.block`` size. Used on the
        reversed (absorption) G legs, and in the Sigma^>-reconstruction mode
        also on the cross-term accumulator (same pattern by construction).

        On the local nnz axis the partner ``(j,i)`` of an entry ``(i,j)`` is
        stored locally unless ``(i,j)`` straddles a block-rank boundary, in which
        case ``(j,i)`` lives on the immediate neighbour block rank (BT band =>
        ``|I-J| <= 1``). The pattern-only plan (a local gather permutation plus a
        neighbour request/reply schedule) is built once; per call only the
        boundary entries are exchanged.
        """
        local_perm, recv_dest, send_src = self._build_fold_plan(g)
        for leg in legs:
            data = leg.data
            folded = data[..., local_perm]
            for nbr in sorted(set(recv_dest) | set(send_src)):
                sbuf = xp.ascontiguousarray(data[..., send_src[nbr]]) \
                    if nbr in send_src else xp.empty((0,), dtype=data.dtype)
                rshape = data.shape[:-1] + (recv_dest[nbr].size,)
                rbuf = xp.empty(rshape, dtype=data.dtype) if nbr in recv_dest \
                    else xp.empty((0,), dtype=data.dtype)
                ranks.block.send_recv(sbuf, nbr, rbuf, nbr)
                if nbr in recv_dest:
                    folded[..., recv_dest[nbr]] = rbuf
            data[:] = folded

    def _build_fold_plan(self, g: DSDBSparse):
        """Pattern-only schedule for :meth:`_fold_reversed_legs` (cached).

        Returns ``(local_perm, recv_dest, send_src)`` where ``local_perm`` gathers
        the locally-resolvable ``(j,i)`` partners (boundary slots point at 0 and
        are overwritten by the exchange), ``recv_dest[nbr]`` are the local nnz
        positions filled from neighbour ``nbr``, and ``send_src[nbr]`` are the
        local nnz positions sent to ``nbr`` (in the order it requests them).
        """
        if self._rev_perm is not False:
            return self._rev_perm
        rows = getattr(g, "rows", None)
        cols = getattr(g, "cols", None)
        if rows is None or cols is None:
            raise TypeError(
                f"The 3-phonon bosonic fold needs the (row, col) nnz pattern, "
                f"which {type(g).__name__} does not expose. Use DSDBCOO "
                "(config.compute.dsdbsparse_type)."
            )
        off = int(getattr(g, "global_block_offset", 0))
        r = (np.asarray(rows.get() if hasattr(rows, "get") else rows) + off)
        c = (np.asarray(cols.get() if hasattr(cols, "get") else cols) + off)
        lut = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(r, c))}
        local_perm = np.zeros(r.size, dtype=np.int64)
        want = {}  # nbr -> list of (dest_local_idx, requested (j,i) key)
        for k, (i, j) in enumerate(zip(r, c)):
            src = lut.get((int(j), int(i)))
            if src is not None:
                local_perm[k] = src
            else:
                nbr = self._block_owner(int(j), g)
                want.setdefault(nbr, []).append((k, (int(j), int(i))))
        recv_dest, send_src = {}, {}
        if want or ranks.block.size > 1:
            bcomm = ranks.block._mpi_comm
            rank, size = ranks.block.rank, ranks.block.size
            for nbr in (rank - 1, rank + 1):
                if not (0 <= nbr < size):
                    continue
                my_keys = [key for _, key in want.get(nbr, [])]
                their_keys = bcomm.sendrecv(my_keys, dest=nbr, sendtag=5,
                                            source=nbr, recvtag=5)
                if nbr in want:
                    recv_dest[nbr] = np.array([d for d, _ in want[nbr]],
                                              dtype=np.int64)
                # reply with MY entry (a,b) the neighbour requested as its (j,i)
                if their_keys:
                    src_idx = []
                    for (a, b) in their_keys:
                        s = lut.get((int(a), int(b)))
                        if s is None:
                            raise RuntimeError(
                                "SSE bosonic fold: neighbour requested a "
                                f"transpose partner ({a},{b}) not stored locally "
                                "-- the ji-band reaches beyond the immediate "
                                "block neighbour; reduce comm.block.size.")
                        src_idx.append(s)
                    send_src[nbr] = np.array(src_idx, dtype=np.int64)
        # every boundary slot must be covered by exactly one neighbour recv
        covered = sum(d.size for d in recv_dest.values())
        missing = [k for k, (i, j) in enumerate(zip(r, c))
                   if lut.get((int(j), int(i))) is None]
        if len(missing) != covered:
            raise RuntimeError(
                "SSE bosonic fold: %d boundary partners unresolved by the "
                "neighbour exchange (covered %d)." % (len(missing), covered))
        local_perm = xp.asarray(local_perm)
        recv_dest = {nbr: xp.asarray(v) for nbr, v in recv_dest.items()}
        send_src = {nbr: xp.asarray(v) for nbr, v in send_src.items()}
        self._rev_perm = (local_perm, recv_dest, send_src)
        return self._rev_perm

    def _herm_masks(self, g: DSDBSparse):
        """Boolean masks over the (full, single-rank) nnz axis selecting the
        strictly-lower and strictly-upper block-triangle entries -- the
        mirrored placeholders of the Hermitian pair-halving. Cached."""
        cached = getattr(self, "_herm_mask_cache", None)
        if cached is None:
            rows, cols = g.spy()
            r = np.asarray(rows.get() if hasattr(rows, "get") else rows)
            c = np.asarray(cols.get() if hasattr(cols, "get") else cols)
            rb = np.searchsorted(self.block_offsets, r, side="right") - 1
            cb = np.searchsorted(self.block_offsets, c, side="right") - 1
            cached = self._herm_mask_cache = (
                xp.asarray(rb > cb), xp.asarray(rb < cb))
        return cached

    def _block_owner(self, orbital_row: int, g: DSDBSparse) -> int:
        """``comm.block`` rank owning the block that contains global ORBITAL
        row ``orbital_row`` (``block_section_offsets`` is in block units, so map
        it through the orbital block-offsets first)."""
        bso = np.asarray(g.block_section_offsets)
        orbital_bounds = self.block_offsets[bso]
        return int(np.searchsorted(orbital_bounds, orbital_row, side="right") - 1)

    @staticmethod
    def _fft_pad(data: NDArray, n_fft: int) -> NDArray:
        """Zero-pad along the energy axis (0) to n_fft and FFT."""
        pad_shape = (n_fft - data.shape[0],) + tuple(data.shape[1:])
        padded = xp.concatenate(
            [data, xp.zeros(pad_shape, dtype=data.dtype)], axis=0
        )
        return xp.fft.fft(padded, axis=0)

    def _links_for_range(self, a: int, b: int) -> set[tuple[int, int]]:
        """Distinct inner G band links needed by outputs owned
         by the block range [a, b)"""
        cache = getattr(self, "_links_cache", None)
        if cache is None:
            cache = self._links_cache = {}
        if (a, b) in cache:
            return cache[(a, b)]
        links: set[tuple[int, int]] = set()
        if self._micro_pair_index is not None:
            for (I, J), pairs in self._micro_pair_index.items():
                if not (a <= min(I, J) < b):
                    continue
                for quads in pairs.values():
                    for quad in quads:
                        links.add(self._micro_layout.grouped_pair(
                            quad.k1, quad.k1p))
                        links.add(self._micro_layout.grouped_pair(
                            quad.k2, quad.k2p))
        else:
            for (I, J), quads in self._phi_pair_index.items():
                if not (a <= min(I, J) < b):
                    continue
                for K1, K2, K1p, K2p, _pl, _pr in quads:
                    links.add((K1, K1p))
                    links.add((K2, K2p))
        cache[(a, b)] = links
        return links

    def _owned_outputs(self, start: int, end: int) -> list[tuple[int, int]]:
        """Output pairs (I,J) this comm.block rank owns (by min(I,J)),
        cached per block range."""
        cache = getattr(self, "_owned_cache", None)
        if cache is None:
            cache = self._owned_cache = {}
        if (start, end) not in cache:
            cache[(start, end)] = [
                ij for ij in (self._micro_pair_index
                              if self._micro_pair_index is not None
                              else self._phi_pair_index)
                if start <= min(ij) < end
            ]
        return cache[(start, end)]

    def _exchange_band_halo(
        self, gtlv, gtgv, ref: DSDBSparse, start: int, end: int,
    ) -> tuple[dict[tuple[int, int], NDArray], dict[tuple[int, int], NDArray]]:
        """Fetch the off-window band links (for both lesser and greater)
        from the immediate comm.block neighbours via a width-N_h halo.

        No MPI tags: both peers enumerate the identical sorted link set
        per direction, so messages between a fixed rank pair match in
        posting order, and the exchange runs through the backend-agnostic
        ``comm.block`` group isend/irecv.
        """
        rank, size = ranks.block.rank, ranks.block.size
        offs = ref.block_section_offsets
        lead = tuple(int(n) for n in ref.data.shape[:-1])

        def is_local(link):
            return start <= min(link) < end

        def shape(link):
            K, Kp = link
            return lead + (int(self.block_sizes[K]),
                           int(self.block_sizes[Kp]))

        neighbours = [
            (nbr, int(offs[nbr]), int(offs[nbr + 1]))
            for nbr in (rank + 1, rank - 1)
            if 0 <= nbr < size
        ]

        needed_nonlocal = {l for l in self._links_for_range(start, end)
                           if not is_local(l)}
        received = set()
        for _neigh, lo, hi in neighbours:
            received |= {l for l in self._links_for_range(start, end)
                         if lo <= min(l) < hi}
        missing = needed_nonlocal - received
        if missing:
            raise RuntimeError(
                "Phonon-phonon halo reaches beyond the immediate comm.block "
                f"neighbour (links {sorted(missing)}). Give each comm.block "
                "rank at least N_h blocks (reduce comm.block.size)."
            )

        halo_l: dict[tuple[int, int], NDArray] = {}
        halo_g: dict[tuple[int, int], NDArray] = {}
        reqs = []
        sendbufs = []

        backend = ranks.block._config["send_recv"]
        ranks.block.group_start(backend)
        for neigh, lo, hi in neighbours:
            # send: links the neighbour's outputs need that I own
            for link in sorted(l for l in self._links_for_range(lo, hi)
                               if is_local(l)):
                K, Kp = link
                bl = xp.ascontiguousarray(gtlv.blocks[K - start, Kp - start])
                bg = xp.ascontiguousarray(gtgv.blocks[K - start, Kp - start])
                sendbufs.extend((bl, bg))
                reqs.append(ranks.block.isend(bl, dest=neigh))
                reqs.append(ranks.block.isend(bg, dest=neigh))
            # recv: links my outputs need that the neighbour owns
            for link in sorted(l for l in self._links_for_range(start, end)
                               if lo <= min(l) < hi):
                bl = xp.empty(shape(link), dtype=complex)
                bg = xp.empty(shape(link), dtype=complex)
                halo_l[link], halo_g[link] = bl, bg
                reqs.append(ranks.block.irecv(bl, source=neigh))
                reqs.append(ranks.block.irecv(bg, source=neigh))
        ranks.block.group_end(backend, reqs)
        return halo_l, halo_g

    def _ensure_tau_buffers(
        self, g_lesser: DSDBSparse, n_fft: int, with_dg: bool = False
    ) -> tuple[DSDBSparse, ...]:
        """Lazily allocate the n_fft tau-buffers sharing G's pattern.

        Six buffers for the quadratic bubble (decay + sigma accumulators
        + reversed legs); ``with_dg`` extends the SAME cached set by four
        more (the direction's decay + reversed legs) for the linearized
        bubble -- allocated only when first requested, so runs that never
        call :meth:`compute_linearized` pay nothing.
        """
        cls = type(g_lesser)
        key = (cls, n_fft)
        n_needed = 10 if with_dg else 6
        if self._tau_cache is not None and self._tau_cache[0] == key:
            bufs = self._tau_cache[1]
            if len(bufs) >= n_needed:
                return bufs[:n_needed]

        rows, cols = g_lesser.spy()
        N = int(np.sum(self.block_sizes))
        pattern = sparse.coo_matrix(
            (
                xp.ones(len(rows), dtype=xp.complex128),
                (xp.asarray(rows), xp.asarray(cols)),
            ),
            shape=(N, N),
        ).tocsr()
        nk = tuple(int(k) for k in g_lesser.global_stack_shape[1:])
        existing = (
            self._tau_cache[1]
            if self._tau_cache is not None and self._tau_cache[0] == key
            else ()
        )
        q_split = getattr(g_lesser, "q_section_offsets", None) is not None
        bufs = existing + tuple(
            cls.from_sparray(
                pattern, self.block_sizes, global_stack_shape=(n_fft,) + nk,
                q_distributed=q_split and slot not in _SIGMA_TAU_SLOTS,
            )
            for slot in range(len(existing), n_needed)
        )
        gt_rows, gt_cols = bufs[0].spy()
        if not (
            bool(xp.array_equal(xp.asarray(rows), xp.asarray(gt_rows)))
            and bool(xp.array_equal(xp.asarray(cols), xp.asarray(gt_cols)))
        ):
            raise RuntimeError(
                "tau-buffer sparsity does not match G; cannot FFT raw data."
            )
        self._tau_cache = (key, bufs)
        return bufs



