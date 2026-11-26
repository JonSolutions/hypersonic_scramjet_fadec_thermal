"""
Domain: Hypersonic Dual-Mode Scramjet FADEC, Shock-Train Isolator & Endothermic Cooling Control
Module: core.endothermic_cracker_bed_deterministic_state_machine_guard
Standard: DO-178C-DAL-A-HYPERSONIC
Primary Law: FayRiddellStagnationHeating (1.1200e+06)
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

LOGGER = logging.getLogger("hypersonic_scr.core.endothermic_crac")

SIGNING_ENV_REF = os.environ.get("TELEMETRY_HMAC_REF", "dev-local-hmac-placeholder")
PRIMARY_LAW_COEFF: float = 1.12000000e+06
SECONDARY_LAW_COEFF: float = 4.65000000e+06
NOMINAL_BASELINE_A: float = 0.740000
NOMINAL_TOLERANCE_A: float = 0.002200
NOMINAL_BASELINE_B: float = 18.500000
NOMINAL_TOLERANCE_B: float = 0.002900


class ZeoliteCatalystReactorState(str, Enum):
    COLD_STANDBY = "COLD_STANDBY"
    PRE_START_VERIFICATION = "PRE_START_VERIFICATION"
    CLOSED_LOOP_ACTIVE = "CLOSED_LOOP_ACTIVE"
    HIGH_GRADIENT_DAMPING = "HIGH_GRADIENT_DAMPING"
    DEGRADED_FALLBACK = "DEGRADED_FALLBACK"
    INTERLOCK_SCRAM_LATCHED = "INTERLOCK_SCRAM_LATCHED"


@dataclass(frozen=True)
class ZeoliteCatalystReactorFrame:
    frame_id: str
    facility_node: str
    timestamp_ms: int
    jp7_conversion_frac: float
    ethylene_yield_mol: float
    radical_oh_ppm: float
    plume_underexpansion: float
    harmonic_distortion_ratio: float
    ambient_stress_factor: float
    redundant_channel_ok: bool = True

    def validate_physical_bounds(self) -> Tuple[bool, List[str]]:
        violations: List[str] = []
        if self.timestamp_ms <= 0:
            violations.append("timestamp_ms_must_be_positive")
        if math.isnan(self.jp7_conversion_frac) or math.isinf(self.jp7_conversion_frac):
            violations.append("jp7_conversion_frac_non_finite")
        if math.isnan(self.ethylene_yield_mol) or math.isinf(self.ethylene_yield_mol):
            violations.append("ethylene_yield_mol_non_finite")
        if self.harmonic_distortion_ratio < 0.0 or self.harmonic_distortion_ratio > 1.0:
            violations.append("harmonic_distortion_out_of_unit_interval")
        if self.ambient_stress_factor <= 0.0:
            violations.append("ambient_stress_factor_non_positive")
        return (len(violations) == 0, violations)

    def compute_normalized_deviation(self) -> float:
        delta_a = abs(self.jp7_conversion_frac - NOMINAL_BASELINE_A) / max(NOMINAL_BASELINE_A, 1e-9)
        delta_b = abs(self.radical_oh_ppm - NOMINAL_BASELINE_B) / max(NOMINAL_BASELINE_B, 1e-9)
        weighted = 0.62 * delta_a + 0.38 * delta_b
        return round(weighted * (1.0 + 0.25 * self.harmonic_distortion_ratio), 8)


@dataclass
class ZeoliteCatalystReactorKalmanStateVector:
    epoch_ms: int
    state_estimate: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    covariance_diag: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    innovation_history: List[float] = field(default_factory=list)
    integrated_lyapunov_energy: float = 0.0
    consecutive_fault_ticks: int = 0
    regime: ZeoliteCatalystReactorState = ZeoliteCatalystReactorState.COLD_STANDBY

    def covariance_trace(self) -> float:
        return float(sum(max(c, 1e-12) for c in self.covariance_diag))

    def spectral_condition_proxy(self) -> float:
        clean = [max(c, 1e-12) for c in self.covariance_diag]
        return float(max(clean) / min(clean))

class ZeoliteCatalystReactorPrimaryEquilibriumSolverM3:
    """
    Production solver for ZeoliteCatalystReactor (PrimaryEquilibriumSolver).
    Enforces FayRiddellStagnationHeating (coeff=1.1200e+06)
    """

    def __init__(self, node_id: str, dt_sec: float = 0.005, gain_alpha: float = 0.42) -> None:
        self.node_id: str = node_id
        self.dt_sec: float = max(dt_sec, 1e-5)
        self.gain_alpha: float = max(min(gain_alpha, 0.995), 0.005)
        self.reference_setpoint: float = 0.740000
        self.tolerance_band: float = 0.002200
        self.state_vector = ZeoliteCatalystReactorKalmanStateVector(
            epoch_ms=int(time.time() * 1000),
            state_estimate=[0.740000, 0.0, 0.111000, 1.0],
            covariance_diag=[0.002200, 0.001100, 0.001760, 0.01],
        )
        self._telemetry_window: List[float] = []
        self._residual_window: List[float] = []
        self._audit_counter: int = 0

    def evaluate_governing_law(self, frame: ZeoliteCatalystReactorFrame) -> Dict[str, float]:
        primary_signal = float(frame.jp7_conversion_frac)
        secondary_signal = float(frame.ethylene_yield_mol)
        coupled_signal = float(frame.radical_oh_ppm)
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

    def step_extended_kalman_filter(self, frame: ZeoliteCatalystReactorFrame) -> Dict[str, Any]:
        is_valid, errors = frame.validate_physical_bounds()
        if not is_valid:
            self.state_vector.consecutive_fault_ticks += 1
            self.state_vector.regime = ZeoliteCatalystReactorState.DEGRADED_FALLBACK
            return {
                "status": "REJECTED_FRAME",
                "errors": errors,
                "regime": self.state_vector.regime.value,
            }
        x0, x1, x2, x3 = self.state_vector.state_estimate
        p0, p1, p2, p3 = self.state_vector.covariance_diag
        pred_x0 = x0 + self.dt_sec * x1
        pred_x1 = 0.96 * x1 - 0.12 * (x0 - self.reference_setpoint)
        pred_x2 = 0.88 * x2 + 0.12 * frame.radical_oh_ppm
        pred_x3 = max(0.05, min(2.5, x3 * (1.0 - 0.01 * frame.harmonic_distortion_ratio)))
        q_proc = max(self.tolerance_band * 0.05, 1e-7)
        pred_p0 = p0 + self.dt_sec * p1 + q_proc
        pred_p1 = p1 * 0.94 + q_proc
        pred_p2 = p2 * 0.95 + q_proc
        pred_p3 = p3 * 0.98 + q_proc * 0.1
        measurement_z = float(frame.jp7_conversion_frac)
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
            "primary_metric": "jp7_conversion_frac",
            "secondary_metric": "ethylene_yield_mol",
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
                self.state_vector.regime = ZeoliteCatalystReactorState.INTERLOCK_SCRAM_LATCHED
            else:
                self.state_vector.regime = ZeoliteCatalystReactorState.HIGH_GRADIENT_DAMPING
        elif m_dist > 3.2:
            prev_ft = self.state_vector.consecutive_fault_ticks
            self.state_vector.consecutive_fault_ticks = max(0, prev_ft - 1)
            self.state_vector.regime = ZeoliteCatalystReactorState.HIGH_GRADIENT_DAMPING
        else:
            self.state_vector.consecutive_fault_ticks = 0
            self.state_vector.regime = ZeoliteCatalystReactorState.CLOSED_LOOP_ACTIVE

    def compute_spectral_stability_metrics(self) -> Dict[str, float]:
        if len(self._telemetry_window) < 4:
            return {
                "window_mean": self.reference_setpoint,
                "window_sigma": 0.0,
                "allan_variance_proxy": 0.0,
                "crest_factor": 1.0,
                "lyapunov_exponent_est": -0.15,
            }
        mean_val = statistics.fmean(self._telemetry_window)
        sigma_val = statistics.pstdev(self._telemetry_window)
        diffs = [
            self._telemetry_window[i] - self._telemetry_window[i - 1]
            for i in range(1, len(self._telemetry_window))
        ]
        allan_var = sum(d * d for d in diffs) / max(2.0 * len(diffs), 1.0)
        peak_abs = max(abs(v) for v in self._telemetry_window)
        rms_val = math.sqrt(sum(v * v for v in self._telemetry_window) / len(self._telemetry_window))
        crest = peak_abs / max(rms_val, 1e-9)
        lyap = math.log(max(allan_var / max(self.tolerance_band ** 2, 1e-12), 1e-6)) * 0.05
        return {
            "window_mean": round(mean_val, 7),
            "window_sigma": round(sigma_val, 7),
            "allan_variance_proxy": round(allan_var, 8),
            "crest_factor": round(crest, 6),
            "lyapunov_exponent_est": round(lyap, 6),
        }

    def synthesize_control_actuation_command(self, target_override: float = -1.0) -> Dict[str, Any]:
        setpoint = target_override if target_override > 0.0 else self.reference_setpoint
        est_pos = self.state_vector.state_estimate[0]
        est_vel = self.state_vector.state_estimate[1]
        tracking_error = setpoint - est_pos
        kp_term = 1.35 * self.gain_alpha * tracking_error
        kd_term = -0.28 * self.gain_alpha * est_vel
        integral_proxy = sum(self.state_vector.innovation_history[-16:]) * 0.04
        raw_u = kp_term + kd_term + integral_proxy
        saturation_limit = abs(self.reference_setpoint) * 0.45 + 5.0
        clamped_u = max(-saturation_limit, min(saturation_limit, raw_u))
        digest_payload = f"{self.node_id}:{self.state_vector.epoch_ms}:{clamped_u:.6f}"
        mac = hmac.new(
            SIGNING_ENV_REF.encode("utf-8"),
            digest_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:24]
        return {
            "node_id": self.node_id,
            "setpoint": round(setpoint, 6),
