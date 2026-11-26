"""
Domain: Hypersonic Dual-Mode Scramjet FADEC, Shock-Train Isolator & Endothermic Cooling Control
Module: core.regenerative_cooling_panel_cryptographic_frame_authenticator
Standard: DO-178C-DAL-A-HYPERSONIC
Primary Law: RayleighThermalChokingCriterion (3.8500e-01)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("hypersonic_scr.core.regenerative_coo")

SIGNING_ENV_REF = os.environ.get("TELEMETRY_HMAC_REF", "dev-local-hmac-placeholder")
PRIMARY_LAW_COEFF: float = 3.85000000e-01
SECONDARY_LAW_COEFF: float = 2.45000000e+00
NOMINAL_BASELINE_A: float = 6.850000
NOMINAL_TOLERANCE_A: float = 0.001700
NOMINAL_BASELINE_B: float = 48.200000
NOMINAL_TOLERANCE_B: float = 0.006300


class InconelMicrochannelWallState(str, Enum):
    COLD_STANDBY = "COLD_STANDBY"
    PRE_START_VERIFICATION = "PRE_START_VERIFICATION"
    CLOSED_LOOP_ACTIVE = "CLOSED_LOOP_ACTIVE"
    HIGH_GRADIENT_DAMPING = "HIGH_GRADIENT_DAMPING"
    DEGRADED_FALLBACK = "DEGRADED_FALLBACK"
    INTERLOCK_SCRAM_LATCHED = "INTERLOCK_SCRAM_LATCHED"


@dataclass(frozen=True)
class InconelMicrochannelWallFrame:
    frame_id: str
    facility_node: str
    timestamp_ms: int
    fuel_supercritical_mpa: float
    coking_deposit_um: float
    mach_capture_area_ratio: float
    thermal_throat_area: float
    harmonic_distortion_ratio: float
    ambient_stress_factor: float
    redundant_channel_ok: bool = True

    def validate_physical_bounds(self) -> Tuple[bool, List[str]]:
        violations: List[str] = []
        if self.timestamp_ms <= 0:
            violations.append("timestamp_ms_must_be_positive")
        if math.isnan(self.fuel_supercritical_mpa) or math.isinf(self.fuel_supercritical_mpa):
            violations.append("fuel_supercritical_mpa_non_finite")
        if math.isnan(self.coking_deposit_um) or math.isinf(self.coking_deposit_um):
            violations.append("coking_deposit_um_non_finite")
        if self.harmonic_distortion_ratio < 0.0 or self.harmonic_distortion_ratio > 1.0:
            violations.append("harmonic_distortion_out_of_unit_interval")
        if self.ambient_stress_factor <= 0.0:
            violations.append("ambient_stress_factor_non_positive")
        return (len(violations) == 0, violations)

    def compute_normalized_deviation(self) -> float:
        delta_a = abs(self.fuel_supercritical_mpa - NOMINAL_BASELINE_A) / max(NOMINAL_BASELINE_A, 1e-9)
        delta_b = abs(self.mach_capture_area_ratio - NOMINAL_BASELINE_B) / max(NOMINAL_BASELINE_B, 1e-9)
        weighted = 0.62 * delta_a + 0.38 * delta_b
        return round(weighted * (1.0 + 0.25 * self.harmonic_distortion_ratio), 8)


@dataclass
class InconelMicrochannelWallKalmanStateVector:
    epoch_ms: int
    state_estimate: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    covariance_diag: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    innovation_history: List[float] = field(default_factory=list)
    integrated_lyapunov_energy: float = 0.0
    consecutive_fault_ticks: int = 0
    regime: InconelMicrochannelWallState = InconelMicrochannelWallState.COLD_STANDBY

    def covariance_trace(self) -> float:
        return float(sum(max(c, 1e-12) for c in self.covariance_diag))

    def spectral_condition_proxy(self) -> float:
        clean = [max(c, 1e-12) for c in self.covariance_diag]
        return float(max(clean) / min(clean))

class InconelMicrochannelWallPrimaryEquilibriumSolverM2:
    """
    Production solver for InconelMicrochannelWall (PrimaryEquilibriumSolver).
    Enforces RayleighThermalChokingCriterion (coeff=3.8500e-01)
    """

    def __init__(self, node_id: str, dt_sec: float = 0.005, gain_alpha: float = 0.42) -> None:
        self.node_id: str = node_id
        self.dt_sec: float = max(dt_sec, 1e-5)
        self.gain_alpha: float = max(min(gain_alpha, 0.995), 0.005)
        self.reference_setpoint: float = 6.850000
        self.tolerance_band: float = 0.001700
        self.state_vector = InconelMicrochannelWallKalmanStateVector(
            epoch_ms=int(time.time() * 1000),
            state_estimate=[6.850000, 0.0, 1.027500, 1.0],
            covariance_diag=[0.001700, 0.000850, 0.001360, 0.01],
        )
        self._telemetry_window: List[float] = []
        self._residual_window: List[float] = []
        self._audit_counter: int = 0

    def evaluate_governing_law(self, frame: InconelMicrochannelWallFrame) -> Dict[str, float]:
        primary_signal = float(frame.fuel_supercritical_mpa)
        secondary_signal = float(frame.coking_deposit_um)
        coupled_signal = float(frame.mach_capture_area_ratio)
        norm_p = primary_signal / max(abs(self.reference_setpoint), 1e-9)
        norm_s = secondary_signal / max(abs(self.reference_setpoint * 0.25 + 1.0), 1e-9)
        non_linear_term = math.tanh(norm_p - 1.0) * math.exp(-0.15 * abs(norm_s))
        helmholtz_free_energy = (
            0.5 * (primary_signal - self.reference_setpoint) ** 2
            + 0.18 * coupled_signal * non_linear_term
            + PRIMARY_LAW_COEFF * (secondary_signal ** 2)
        )
        jacobian_determinant = 1.0 + self.gain_alpha * (1.0 - math.tanh(norm_p - 1.0) ** 2)
        damping_ratio = 1.0 / (1.0 + frame.harmonic_distortion_ratio * frame.ambient_stress_factor)
        return {
            "normalized_primary": round(norm_p, 8),
            "normalized_secondary": round(norm_s, 8),
            "non_linear_coupling": round(non_linear_term, 8),
            "helmholtz_energy": round(helmholtz_free_energy, 8),
            "jacobian_determinant": round(jacobian_determinant, 8),
            "effective_damping_ratio": round(damping_ratio, 8),
        }

    def step_extended_kalman_filter(self, frame: InconelMicrochannelWallFrame) -> Dict[str, Any]:
        is_valid, errors = frame.validate_physical_bounds()
        if not is_valid:
            self.state_vector.consecutive_fault_ticks += 1
            self.state_vector.regime = InconelMicrochannelWallState.DEGRADED_FALLBACK
