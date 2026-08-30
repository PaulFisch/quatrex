"""Production A/B gate for the exact analytic JVP (mixing_method=newton).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from quatrex.core.config import parse_config, setup_context  # noqa: E402

CFG = os.environ["QX_CONFIG"]
SNAP = os.environ["QX_SNAPSHOT"]
cfg = parse_config(CFG)
if os.environ.get("QX_GBAND"):
    cfg.phonon.sse_g_band = int(os.environ["QX_GBAND"])
if os.environ.get("QX_JVP_FORM"):
    cfg.scba.experimental_mixer.newton_jvp_form = os.environ["QX_JVP_FORM"]
if os.environ.get("QX_RECON_TOL"):
    cfg.scba.experimental_mixer.newton_recon_check_tol = float(
        os.environ["QX_RECON_TOL"])
cfg.scba.mixing_method = "newton"
cfg.scba.max_iterations = 12
cfg.scba.min_iterations = 12
setup_context(cfg)

from quatrex.core.scba import SCBA  # noqa: E402
from qttools import xp  # noqa: E402
from qttools.comm import comm  # noqa: E402

scba = SCBA(cfg)

def _rank_file(base: str) -> str:
    """Match the engine's deterministic distributed-snapshot convention."""
    if comm.size == 1:
        return base
    stem = base[:-4] if base.endswith(".npz") else base
    return f"{stem}.rank{comm.rank}.npz"


snap_path = _rank_file(SNAP)
snap = np.load(snap_path)
scba.data.sigma_lesser.data[:] = xp.asarray(snap["sigma_lesser"])
scba.data.sigma_greater.data[:] = xp.asarray(snap["sigma_greater"])
scba.data.sigma_retarded_hermitian.data[:] = xp.asarray(
    snap["sigma_retarded"])
print(f"AB: loaded snapshot {snap_path}", flush=True)


class ABMixer:
    """Capture mixer: stage 0 computes the analytic JVP and emits the
    first FD probe; stages 1..4 collect F at x0 +/- eps_i * v; the last
    stage reports and exits."""

    probing = False

    def __init__(self, scba, eps_rels=(1e-5, 1e-6)):
        self.scba = scba
        self.eps_rels = eps_rels
        self.stage = 0
        self.fd = {}

    def _norm(self, v):
        return float(np.linalg.norm(v))

    def step(self, x, gx):
        s = self.stage
        if s == 0:
            self.x0 = x.copy()
            jvp = self.scba._get_phonon_jvp()
            t0 = time.time()
            self.recon = jvp.prepare()
            t_prep = time.time() - t0

            rng = np.random.default_rng(123)
            size = jvp._n_local * jvp._nnz

            def rnd():
                return (rng.standard_normal((jvp._n_local, jvp._nnz))
                        + 1j * rng.standard_normal((jvp._n_local, jvp._nnz)))

            vl = jvp._to_flat(jvp._skew_project(
                jvp._to_dense(rnd(), bt_only=True)))
            vg = jvp._to_flat(jvp._skew_project(
                jvp._to_dense(rnd(), bt_only=True)))
            vr = jvp._to_flat(jvp._to_dense(rnd(), bt_only=True))
            v = np.concatenate([vl.ravel(), vg.ravel(), vr.ravel()])
            v *= self._norm(x) / self._norm(v)
            self.v = v

            t0 = time.time()
            self.Jv_an = jvp.apply(v)
            self.t_jvp = time.time() - t0
            # Both evaluation routes on the same direction when the explicit
            # mixed-leg implementation supports this kernel.  Coupled q and
            # factored/symmetry paths deliberately use the production-kernel
            # polarisation identity only; asking for the Gamma-only bilinear
            # route there would test a different map.
            if jvp._bilinear_supported:
                alt = ("polarization" if jvp.jvp_form == "bilinear"
                       else "bilinear")
                Jv_alt = jvp.apply(v, form=alt)
                self.rel_forms = (self._norm(self.Jv_an - Jv_alt)
                                  / max(self._norm(self.Jv_an), 1e-300))
                forms = f"|{jvp.jvp_form}-{alt}|/|Jv|={self.rel_forms:.3e}"
            else:
                self.rel_forms = None
                forms = "bilinear cross unavailable for this production map"
            print(f"AB: recon={self.recon:.2e} t_prepare={t_prep:.1f}s "
                  f"t_jvp={self.t_jvp:.1f}s form={jvp.jvp_form} "
                  f"{forms}",
                  flush=True)
            self.stage = 1
            self.probing = True
            eps = self.eps_rels[0]
            return x + eps * v
        # FD probe collection: stages 1..2*len(eps): plus/minus per eps.
        i_eps = (s - 1) // 2
        eps = self.eps_rels[i_eps]
        if (s - 1) % 2 == 0:
            self.fd[(i_eps, "+")] = gx.copy()
            self.stage += 1
            self.probing = True
            return self.x0 - eps * self.v
        self.fd[(i_eps, "-")] = gx.copy()
        self.stage += 1
        if i_eps + 1 < len(self.eps_rels):
            self.probing = True
            return self.x0 + self.eps_rels[i_eps + 1] * self.v
        # Done: report.
        results = {"recon": self.recon, "t_jvp_s": self.t_jvp,
                   "rel_forms": self.rel_forms}
        an_n = self._norm(self.Jv_an)
        for k, e in enumerate(self.eps_rels):
            jfd = (self.fd[(k, "+")] - self.fd[(k, "-")]) / (2 * e)
            rel = self._norm(self.Jv_an - jfd) / max(self._norm(jfd), 1e-300)
            results[f"rel_eps{e:g}"] = rel
            print(f"AB: eps_rel={e:g}  |Jv_an - Jv_fd|/|Jv_fd| = {rel:.3e} "
                  f"(|Jv_an|={an_n:.3e})", flush=True)
        out = os.environ.get("QX_AB_OUT")
        if out:
            if comm.size > 1:
                stem = out[:-5] if out.endswith(".json") else out
                out = f"{stem}.rank{comm.rank}.json"
            Path(out).write_text(json.dumps(results, indent=1))
        ok = min(results[f"rel_eps{e:g}"] for e in self.eps_rels) < 3e-5
        print(f"AB: {'PASS' if ok else 'FAIL'}", flush=True)
        sys.stdout.flush()
        os._exit(0 if ok else 1)


scba._anderson_mixer = ABMixer(scba)
scba.run()
