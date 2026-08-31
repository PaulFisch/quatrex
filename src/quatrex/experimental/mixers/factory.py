# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Factory for experimental SCBA root finders."""

from quatrex.experimental.mixers.broyden import BroydenMixer
from quatrex.experimental.mixers.jfnk import JFNKMixer
from quatrex.experimental.mixers.newton import NewtonKrylovMixer
from quatrex.experimental.mixers.rpm import RPMMixer
from quatrex.experimental.mixers.rre import RREMixer


def build_mixer(method, config, beta, depth, jvp_factory):
    """Construct an experimental mixer."""
    if method == "broyden":
        return BroydenMixer(
            depth=depth,
            beta=beta,
            ridge=config.broyden_ridge,
            warmup=config.broyden_warmup_iters,
            trust=config.broyden_trust,
        )
    if method == "rpm":
        return RPMMixer(
            max_subspace=config.rpm_max_subspace,
            beta=beta,
            ridge=config.broyden_ridge,
            warmup=config.broyden_warmup_iters,
            trust=config.broyden_trust,
        )
    if method == "rre":
        return RREMixer(
            cycle=config.rre_cycle,
            beta=beta,
            ridge=config.rre_ridge,
        )
    if method == "jfnk":
        return JFNKMixer(
            warmup=config.jfnk_warmup_iters,
            beta=beta,
            max_krylov=config.jfnk_max_krylov,
            inner_tol=config.jfnk_inner_tol,
            forcing=config.jfnk_forcing,
            max_newton=config.jfnk_max_newton,
            eps=config.jfnk_eps,
            trust=config.jfnk_trust,
            trust_max=config.jfnk_trust_max,
            newton_damp=config.jfnk_newton_damp,
            ptc=config.jfnk_ptc,
        )
    if method == "newton":
        return NewtonKrylovMixer(
            jvp_factory=jvp_factory,
            warmup=config.newton_warmup_iters,
            switch_tol=config.newton_switch_tol,
            beta=beta,
            max_krylov=config.newton_max_krylov,
            inner_tol=config.newton_inner_tol,
            forcing=config.newton_forcing,
            max_newton=config.newton_max_newton,
            trust=config.newton_trust,
            trust_max=config.newton_trust_max,
            newton_damp=config.newton_damp,
            backtrack=config.newton_backtrack,
            precond=config.newton_precond,
            precond_rank=config.newton_precond_rank,
        )
    raise ValueError(f"Unknown experimental mixing method {method!r}.")

