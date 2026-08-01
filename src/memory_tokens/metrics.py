import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BinomialMetric:
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    def wilson95(self) -> tuple[float, float]:
        if self.total <= 0:
            raise ValueError("total must be positive")
        z = 1.959963984540054
        p = self.accuracy
        denominator = 1.0 + z * z / self.total
        center = (p + z * z / (2 * self.total)) / denominator
        radius = (
            z
            * math.sqrt(
                p * (1.0 - p) / self.total + z * z / (4.0 * self.total * self.total)
            )
            / denominator
        )
        return center - radius, center + radius

    def asdict(self) -> dict[str, int | float | list[float]]:
        return {
            "correct": self.correct,
            "total": self.total,
            "accuracy": self.accuracy,
            "wilson95": list(self.wilson95()),
        }


@dataclass(frozen=True)
class EvaluationMetric:
    """Overall accuracy with separate conflict and single-rule results."""

    correct: int
    total: int
    conflict_correct: int
    conflict_total: int
    single_rule_correct: int
    single_rule_total: int
    hot_correct: int
    hot_total: int
    cold_correct: int
    cold_total: int
    rare_context_correct: int
    rare_context_total: int
    common_context_correct: int
    common_context_total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    @property
    def conflict_accuracy(self) -> float | None:
        if self.conflict_total == 0:
            return None
        return self.conflict_correct / self.conflict_total

    @property
    def cold_accuracy(self) -> float | None:
        if self.cold_total == 0:
            return None
        return self.cold_correct / self.cold_total

    @property
    def rare_context_accuracy(self) -> float | None:
        if self.rare_context_total == 0:
            return None
        return self.rare_context_correct / self.rare_context_total

    def wilson95(self) -> tuple[float, float]:
        return BinomialMetric(self.correct, self.total).wilson95()

    @staticmethod
    def _group(correct: int, total: int) -> dict[str, object]:
        if total == 0:
            return {
                "correct": 0,
                "total": 0,
                "accuracy": None,
                "wilson95": None,
            }
        return BinomialMetric(correct, total).asdict()

    def asdict(self) -> dict[str, object]:
        return {
            **BinomialMetric(self.correct, self.total).asdict(),
            "conflict": self._group(self.conflict_correct, self.conflict_total),
            "single_rule": self._group(
                self.single_rule_correct,
                self.single_rule_total,
            ),
            "hot": self._group(self.hot_correct, self.hot_total),
            "cold": self._group(self.cold_correct, self.cold_total),
            "rare_context": self._group(
                self.rare_context_correct,
                self.rare_context_total,
            ),
            "common_context": self._group(
                self.common_context_correct,
                self.common_context_total,
            ),
        }
