# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Transient analytic-pole inputs for one three-phonon bubble."""

from dataclasses import dataclass


@dataclass
class PoleBubbleState:
    """Inputs injected into one bubble evaluation."""

    channel: tuple | None = None
    self_energy: tuple | None = None
    mixed: tuple | None = None
    covariance: tuple | None = None

    def clear(self) -> None:
        """Discard inputs after the bubble consumes them."""
        self.channel = None
        self.self_energy = None
        self.mixed = None
        self.covariance = None
