"""
Domain: Hypersonic Dual-Mode Scramjet FADEC, Shock-Train Isolator & Endothermic Cooling Control
Module: core.oblique_shock_isolator_canonical_invariant_registry
Standard: DO-178C-DAL-A-HYPERSONIC
Primary Law: BilligShockTrainCorrelation (1.4000e+00)
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

LOGGER = logging.getLogger("hypersonic_scr.core.oblique_shock_is")

SIGNING_ENV_REF = os.environ.get("TELEMETRY_HMAC_REF", "dev-local-hmac-placeholder")
PRIMARY_LAW_COEFF: float = 1.40000000e+00
SECONDARY_LAW_COEFF: float = 1.12000000e+06
NOMINAL_BASELINE_A: float = 28.400000
NOMINAL_TOLERANCE_A: float = 0.004100
NOMINAL_BASELINE_B: float = 0.920000
NOMINAL_TOLERANCE_B: float = 0.001500


class BilligShockTrainDuctState(str, Enum):
    COLD_STANDBY = "COLD_STANDBY"
    PRE_START_VERIFICATION = "PRE_START_VERIFICATION"
    CLOSED_LOOP_ACTIVE = "CLOSED_LOOP_ACTIVE"
    HIGH_GRADIENT_DAMPING = "HIGH_GRADIENT_DAMPING"
    DEGRADED_FALLBACK = "DEGRADED_FALLBACK"
    INTERLOCK_SCRAM_LATCHED = "INTERLOCK_SCRAM_LATCHED"


@dataclass(frozen=True)
class BilligShockTrainDuctFrame:
    frame_id: str
    facility_node: str
    timestamp_ms: int
    pre_combustion_shock_x: float
    unstart_margin_pct: float
    damkohler_residence_no: float
    freestream_mach_est: float
    harmonic_distortion_ratio: float
    ambient_stress_factor: float
    redundant_channel_ok: bool = True

    def validate_physical_bounds(self) -> Tuple[bool, List[str]]:
        violations: List[str] = []
        if self.timestamp_ms <= 0:
            violations.append("timestamp_ms_must_be_positive")
        if math.isnan(self.pre_combustion_shock_x) or math.isinf(self.pre_combustion_shock_x):
            violations.append("pre_combustion_shock_x_non_finite")
        if math.isnan(self.unstart_margin_pct) or math.isinf(self.unstart_margin_pct):
            violations.append("unstart_margin_pct_non_finite")
        if self.harmonic_distortion_ratio < 0.0 or self.harmonic_distortion_ratio > 1.0:
            violations.append("harmonic_distortion_out_of_unit_interval")
        if self.ambient_stress_factor <= 0.0:
            violations.append("ambient_stress_factor_non_positive")
        return (len(violations) == 0, violations)

    def compute_normalized_deviation(self) -> float:
        delta_a = abs(self.pre_combustion_shock_x - NOMINAL_BASELINE_A) / max(NOMINAL_BASELINE_A, 1e-9)
        delta_b = abs(self.damkohler_residence_no - NOMINAL_BASELINE_B) / max(NOMINAL_BASELINE_B, 1e-9)
        weighted = 0.62 * delta_a + 0.38 * delta_b
        return round(weighted * (1.0 + 0.25 * self.harmonic_distortion_ratio), 8)


@dataclass
class BilligShockTrainDuctKalmanStateVector:
    epoch_ms: int
    state_estimate: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    covariance_diag: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    innovation_history: List[float] = field(default_factory=list)
    integrated_lyapunov_energy: float = 0.0
    consecutive_fault_ticks: int = 0
    regime: BilligShockTrainDuctState = BilligShockTrainDuctState.COLD_STANDBY

    def covariance_trace(self) -> float:
        return float(sum(max(c, 1e-12) for c in self.covariance_diag))

    def spectral_condition_proxy(self) -> float:
        clean = [max(c, 1e-12) for c in self.covariance_diag]
        return float(max(clean) / min(clean))

class BilligShockTrainDuctPrimaryEquilibriumSolverM1:
    """
    Production solver for BilligShockTrainDuct (PrimaryEquilibriumSolver).
    Enforces BilligShockTrainCorrelation (coeff=1.4000e+00)
    """

    def __init__(self, node_id: str, dt_sec: float = 0.005, gain_alpha: float = 0.42) -> None:
        self.node_id: str = node_id
        self.dt_sec: float = max(dt_sec, 1e-5)
        self.gain_alpha: float = max(min(gain_alpha, 0.995), 0.005)
        self.reference_setpoint: float = 28.400000
        self.tolerance_band: float = 0.004100
        self.state_vector = BilligShockTrainDuctKalmanStateVector(
            epoch_ms=int(time.time() * 1000),
            state_estimate=[28.400000, 0.0, 4.260000, 1.0],
            covariance_diag=[0.004100, 0.002050, 0.003280, 0.01],
        )
        self._telemetry_window: List[float] = []
        self._residual_window: List[float] = []
        self._audit_counter: int = 0

    def evaluate_governing_law(self, frame: BilligShockTrainDuctFrame) -> Dict[str, float]:
        primary_signal = float(frame.pre_combustion_shock_x)
        secondary_signal = float(frame.unstart_margin_pct)
        coupled_signal = float(frame.damkohler_residence_no)
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

    def step_extended_kalman_filter(self, frame: BilligShockTrainDuctFrame) -> Dict[str, Any]:
        is_valid, errors = frame.validate_physical_bounds()
        if not is_valid:
            self.state_vector.consecutive_fault_ticks += 1
            self.state_vector.regime = BilligShockTrainDuctState.DEGRADED_FALLBACK
            return {
                "status": "REJECTED_FRAME",
                "errors": errors,
                "regime": self.state_vector.regime.value,
            }
        x0, x1, x2, x3 = self.state_vector.state_estimate
        p0, p1, p2, p3 = self.state_vector.covariance_diag
        pred_x0 = x0 + self.dt_sec * x1
        pred_x1 = 0.96 * x1 - 0.12 * (x0 - self.reference_setpoint)
        pred_x2 = 0.88 * x2 + 0.12 * frame.damkohler_residence_no
        pred_x3 = max(0.05, min(2.5, x3 * (1.0 - 0.01 * frame.harmonic_distortion_ratio)))
        q_proc = max(self.tolerance_band * 0.05, 1e-7)
        pred_p0 = p0 + self.dt_sec * p1 + q_proc
        pred_p1 = p1 * 0.94 + q_proc
        pred_p2 = p2 * 0.95 + q_proc
        pred_p3 = p3 * 0.98 + q_proc * 0.1
        measurement_z = float(frame.pre_combustion_shock_x)
        r_noise = max(self.tolerance_band * (1.0 + frame.harmonic_distortion_ratio), 1e-6)
        innovation_y = measurement_z - pred_x0
        innovation_s = pred_p0 + r_noise
        kalman_gain_k0 = pred_p0 / innovation_s
        kalman_gain_k1 = (0.25 * pred_p1) / innovation_s
        upd_x0 = pred_x0 + kalman_gain_k0 * innovation_y
        upd_x1 = pred_x1 + kalman_gain_k1 * innovation_y
        upd_p0 = max((1.0 - kalman_gain_k0) * pred_p0, 1e-12)
        upd_p1 = max((1.0 - 0.25 * kalman_gain_k1) * pred_p1, 1e-12)
        mahalanobis_d2 = (innovation_y ** 2) / max(innovation_s, 1e-12)
        mahalanobis_dist = math.sqrt(max(mahalanobis_d2, 0.0))
        self.state_vector.epoch_ms = frame.timestamp_ms
        self.state_vector.state_estimate = [upd_x0, upd_x1, pred_x2, pred_x3]
        self.state_vector.covariance_diag = [upd_p0, upd_p1, pred_p2, pred_p3]
        self.state_vector.innovation_history.append(round(innovation_y, 8))
        if len(self.state_vector.innovation_history) > 128:
            self.state_vector.innovation_history.pop(0)
        self._telemetry_window.append(measurement_z)
        if len(self._telemetry_window) > 128:
            self._telemetry_window.pop(0)
        self._residual_window.append(mahalanobis_dist)
        if len(self._residual_window) > 128:
            self._residual_window.pop(0)
        self.state_vector.integrated_lyapunov_energy = round(
            0.92 * self.state_vector.integrated_lyapunov_energy + 0.08 * mahalanobis_d2, 8
        )
        self._transition_operational_regime(mahalanobis_dist, frame.redundant_channel_ok)
        self._audit_counter += 1
        return {
            "status": "CONVERGED",
            "node_id": self.node_id,
            "primary_metric": "pre_combustion_shock_x",
            "secondary_metric": "unstart_margin_pct",
            "posterior_estimate": round(upd_x0, 7),
            "posterior_velocity": round(upd_x1, 7),
            "innovation_residual": round(innovation_y, 7),
            "kalman_gain": round(kalman_gain_k0, 7),
            "mahalanobis_distance": round(mahalanobis_dist, 7),
            "covariance_trace": round(self.state_vector.covariance_trace(), 8),
            "regime": self.state_vector.regime.value,
            "audit_seq": self._audit_counter,
        }

    def _transition_operational_regime(self, m_dist: float, redundant_ok: bool) -> None:
        if not redundant_ok or m_dist > 6.5:
            self.state_vector.consecutive_fault_ticks += 1
            if self.state_vector.consecutive_fault_ticks >= 3:
                self.state_vector.regime = BilligShockTrainDuctState.INTERLOCK_SCRAM_LATCHED
            else:
                self.state_vector.regime = BilligShockTrainDuctState.HIGH_GRADIENT_DAMPING
        elif m_dist > 3.2:
            prev_ft = self.state_vector.consecutive_fault_ticks
            self.state_vector.consecutive_fault_ticks = max(0, prev_ft - 1)
            self.state_vector.regime = BilligShockTrainDuctState.HIGH_GRADIENT_DAMPING
        else:
            self.state_vector.consecutive_fault_ticks = 0
            self.state_vector.regime = BilligShockTrainDuctState.CLOSED_LOOP_ACTIVE
