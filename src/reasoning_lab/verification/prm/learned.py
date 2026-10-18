"""Shallow learned PRM head on top of symbolic verifier features.

Trains in seconds on a CPU and gives a second, data-driven signal for step
scoring and reranking. Labels come from :mod:`reasoning_lab.verification.prm.labels`
(symbolic verdicts, outcome attribution, or rollout success rates).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ...schema import Step
from .base import Aggregator, ProcessRewardModel, StepScore
from .features import FEATURE_NAMES, step_features
from .symbolic import SymbolicPRM


@dataclass
class LabeledExample:
    problem_id: str
    depth: int
    features: list[float]
    label: float
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PRMTrainReport:
    n_examples: int
    n_positive: int
    accuracy: float
    auc: float | None
    estimator: str
    C: float
    feature_names: list[str]
    coefficients: list[float]
    sources: dict[str, int]

    def to_markdown(self) -> str:
        top = sorted(zip(self.feature_names, self.coefficients, strict=False), key=lambda kv: -abs(kv[1]))[:8]
        rows = "\n".join(f"| {name} | {coef:+.3f} |" for name, coef in top)
        return (
            f"Estimator: `{self.estimator}` (C={self.C}) - {self.n_examples} examples, "
            f"{self.n_positive} positive.\n\n"
            f"accuracy {self.accuracy:.3f}" + (f", auc {self.auc:.3f}" if self.auc is not None else "") + "\n\n"
            f"| feature | weight |\n| --- | --- |\n{rows}\n"
        )


class LearnedPRM(ProcessRewardModel):
    name = "learned"

    def __init__(
        self,
        estimator: Any,
        *,
        base: SymbolicPRM | None = None,
        aggregator: Aggregator = "geomean",
        feature_names: tuple[str, ...] = FEATURE_NAMES,
    ):
        self.estimator = estimator
        self.base = base or SymbolicPRM()
        self.aggregator = aggregator
        self.feature_names = list(feature_names)

    @classmethod
    def fit(
        cls,
        examples: list[LabeledExample],
        *,
        estimator: str = "logistic",
        C: float = 1.0,
        test_fraction: float = 0.25,
        seed: int = 0,
    ) -> tuple[LearnedPRM, PRMTrainReport]:
        """Fit the head and report held-out metrics; no GPU, no deep network."""
        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if len(examples) < 8:
            raise ValueError(f"need at least 8 labelled steps, got {len(examples)}")
        X = np.asarray([e.features for e in examples], dtype=float)
        y = np.asarray([1 if e.label >= 0.5 else 0 for e in examples], dtype=int)
        if len(set(y.tolist())) < 2:
            raise ValueError("labels are degenerate: every step has the same class")

        model = (
            LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
            if estimator == "logistic"
            else GradientBoostingClassifier(n_estimators=80, max_depth=2, random_state=seed)
        )
        pipe = make_pipeline(StandardScaler(), model)

        strat = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=seed, stratify=strat
        )
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        try:
            auc = float(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]))
        except ValueError:
            auc = None
        sources: dict[str, int] = {}
        for example in examples:
            sources[example.source or "unspecified"] = sources.get(example.source or "unspecified", 0) + 1
        linear = pipe.steps[-1][1]
        coefficients = list(getattr(linear, "coef_", np.zeros(len(FEATURE_NAMES)))[0])
        report = PRMTrainReport(
            n_examples=len(examples),
            n_positive=int(y.sum()),
            accuracy=float(accuracy_score(y_test, pred)),
            auc=auc,
            estimator=estimator,
            C=C,
            feature_names=list(FEATURE_NAMES),
            coefficients=[float(c) for c in coefficients],
            sources=sources,
        )
        return cls(pipe, aggregator="geomean"), report

    async def score_step(self, problem_id: str, prefix: list[Step], step: Step) -> StepScore:
        base_score = await self.base.score_step(problem_id, prefix, step)
        features = [step_features(prefix, step, base_score)]
        probability = float(self.estimator.predict_proba(features)[0][1])
        blend = 0.5 * probability + 0.5 * base_score.score
        return StepScore(
            score=round(max(0.001, min(0.999, blend)), 5),
            status=base_score.status,
            detail=f"learned={probability:.3f} symbolic={base_score.score:.3f}",
            source=self.name,
            checks=base_score.checks,
        )

    # -- persistence -------------------------------------------------
    def save(self, path: Path | str) -> Path:
        import joblib

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, target)
        target.with_suffix(".meta.json").write_text(
            json.dumps({"feature_names": self.feature_names, "aggregator": self.aggregator}, indent=2),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Path | str) -> LearnedPRM:
        import joblib

        target = Path(path)
        meta_file = target.with_suffix(".meta.json")
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
        return cls(
            joblib.load(target),
            aggregator=meta.get("aggregator", "geomean"),
            feature_names=tuple(meta.get("feature_names", FEATURE_NAMES)),
        )
