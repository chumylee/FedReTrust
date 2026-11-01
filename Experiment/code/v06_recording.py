"""Affine audit for v0.6: current submitted deltas plus client-specific history.

The full applied contribution of client i is c_i*y_i, where
y_i=lambda_i*x_i+(1-lambda_i)*h_i. Returned coefficients c_i*lambda_i
alone do not reconstruct the update. No simulator intent enters this recorder.
"""
import copy
import time

import torch

from experiment_core import AggregationResult, file_sha256
from fedretrust_v06 import COEFFICIENT_CONVENTION, FedReTrustV06
from fltrust import flatten_delta


class InnovationContributionRecorder:
    def __init__(self, base):
        if not isinstance(base, FedReTrustV06):
            raise TypeError("The affine recorder requires FedReTrustV06 history semantics.")
        self.base, self.name = base, base.name
        self.settings = {**copy.deepcopy(base.settings), "measurement": {
            "name": "innovation_affine_contribution_recorder_v1",
            "implementation_sha256": file_sha256(__file__),
            "coefficient_convention": COEFFICIENT_CONVENTION,
            "returned_coefficients": "current_raw_delta_part_only",
            "gram_matrix": "full_applied_contributions_including_client_history",
            "additional_grams": "current_raw_part_historical_part_and_current_history_cross",
            "timing": "base_and_recorder_separate"}}

    def reset(self):
        self.base.reset()

    def aggregate(self, submissions, broadcast, round_number):
        # The method replaces these maps and never mutates their tensor values.
        # Retain the old state for an independent affine audit and roll back if
        # that audit fails, so a failed wrapper call cannot advance history.
        saved = (self.base.weights, self.base.history, self.base.last_round, self.base.layout_signature)
        started = time.perf_counter()
        try:
            result = self.base.aggregate(submissions, broadcast, round_number)
            base_seconds = time.perf_counter() - started
            return self._measure(submissions, broadcast, result, saved[1], base_seconds)
        except BaseException:
            self.base.weights, self.base.history, self.base.last_round, self.base.layout_signature = saved
            raise

    @torch.inference_mode()
    def _measure(self, submissions, broadcast, result, old_history, base_seconds):
        started = time.perf_counter()
        ordered = sorted(submissions, key=lambda s: s.client_id)
        if (not ordered or len({s.client_id for s in ordered}) != len(ordered) or
                set(result.coefficients) != {s.client_id for s in ordered}):
            raise ValueError("Current coefficients must cover distinct submissions exactly.")
        if result.diagnostics.get("coefficient_convention") != COEFFICIENT_CONVENTION:
            raise ValueError("Expected the explicit affine coefficient contract.")
        metrics = result.diagnostics["client_metrics"]
        raw = torch.stack([flatten_delta(s.state, broadcast) for s in ordered])
        previous = torch.stack([old_history.get(s.client_id, torch.zeros_like(raw[i]))
                                for i, s in enumerate(ordered)])
        filtered = torch.stack([self.base.history[s.client_id] for s in ordered])
        factors = torch.tensor([metrics[str(s.client_id)]["innovation_factor"] for s in ordered], dtype=torch.float64)
        coeff = torch.tensor([metrics[str(s.client_id)]["filtered_effective_coefficient"] for s in ordered], dtype=torch.float64)
        current_coeff = torch.tensor([result.coefficients[s.client_id] for s in ordered], dtype=torch.float64)
        historical_coeff = coeff * (1. - factors)
        if (not torch.isfinite(coeff).all() or (coeff < 0).any() or
                not torch.isfinite(factors).all() or (factors < 0).any() or (factors > 1).any()):
            raise ValueError("Expected finite nonnegative coefficients and unit-interval innovation factors.")
        torch.testing.assert_close(current_coeff, coeff * factors, rtol=1e-12, atol=1e-14)
        current = current_coeff[:, None] * raw
        historical = historical_coeff[:, None] * previous
        full = coeff[:, None] * filtered
        affine = current + historical
        torch.testing.assert_close(affine, full, rtol=2e-12, atol=2e-12)
        total = full.sum(0)
        affine_total = current.sum(0) + historical.sum(0)
        torch.testing.assert_close(affine_total, total, rtol=2e-12, atol=2e-12)
        affine_error = float((affine_total - total).abs().max())
        max_error, offset = 0., 0
        for name, reference in broadcast.items():
            size = reference.numel()
            expected = (reference.double() + affine_total[offset:offset + size].reshape(reference.shape)).to(reference.dtype)
            torch.testing.assert_close(expected, result.state[name], rtol=2e-6, atol=2e-8)
            max_error = max(max_error, float((expected.double() - result.state[name].double()).abs().max()))
            offset += size
        raw_norms = torch.linalg.vector_norm(raw, dim=1)
        current_norms = torch.linalg.vector_norm(current, dim=1)
        historical_norms = torch.linalg.vector_norm(historical, dim=1)
        full_norms = torch.linalg.vector_norm(full, dim=1)
        norm_sum = float(full_norms.sum())
        total_norm = float(torch.linalg.vector_norm(total))
        current_mass = float(current_coeff.sum())
        rows = {}
        for i, submission in enumerate(ordered):
            rows[str(submission.client_id)] = {**metrics[str(submission.client_id)],
                "recorder_native_coefficient": float(current_coeff[i]),
                "applied_delta_coefficient": float(current_coeff[i]),
                "raw_delta_l2_norm": float(raw_norms[i]),
                "current_contribution_l2_norm": float(current_norms[i]),
                "historical_contribution_l2_norm": float(historical_norms[i]),
                "applied_contribution_l2_norm": float(full_norms[i]),
                "applied_contribution_scope": "full_current_plus_client_history",
                "current_coefficient_share": float(current_coeff[i]) / current_mass if current_mass else None,
                "contribution_norm_share": float(full_norms[i]) / norm_sum if norm_sum else None,
                "aggregate_projection_share": float(torch.dot(full[i], total)) / total_norm**2 if total_norm else None}
        measurements = {
            "client_order": [s.client_id for s in ordered],
            "gram_matrix": (full @ full.T).tolist(),
            "gram_matrix_convention": "full_applied_current_plus_history_contributions",
            "current_gram_matrix": (current @ current.T).tolist(),
            "historical_gram_matrix": (historical @ historical.T).tolist(),
            "current_history_cross_gram_matrix": (current @ historical.T).tolist(),
            "sum_individual_norms": norm_sum, "aggregate_l2_norm": total_norm,
            "current_aggregate_l2_norm": float(torch.linalg.vector_norm(current.sum(0))),
            "historical_aggregate_l2_norm": float(torch.linalg.vector_norm(historical.sum(0))),
            "effective_coefficient_mass": current_mass,
            "effective_coefficient_mass_scope": "current_raw_delta_part_only",
            "filtered_effective_coefficient_mass": float(coeff.sum()),
            "historical_coefficient_mass": float(historical_coeff.sum()),
            "historical_term_present": bool(torch.any(historical != 0.)),
            "cancellation_ratio": total_norm / norm_sum if norm_sum else None,
            "zero_contribution_mass": norm_sum == 0.,
            "zero_aggregate_from_cancellation": norm_sum > 0. and total_norm == 0.,
            "affine_reconstruction_max_abs_error": affine_error,
            "reconstruction_max_abs_error": max_error,
            "defence_aggregation_seconds": base_seconds,
            "metric_recording_seconds": time.perf_counter() - started,
        }
        return AggregationResult(result.state, dict(result.coefficients), {
            **result.diagnostics, "client_metrics": rows, "contribution_measurements": measurements})
