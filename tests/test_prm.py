from __future__ import annotations

import pytest

from reasoning_lab.schema import Step, VerifyStatus
from reasoning_lab.verification.prm import build_prm
from reasoning_lab.verification.prm.base import NullPRM
from reasoning_lab.verification.prm.features import FEATURE_NAMES, step_features
from reasoning_lab.verification.prm.labels import outcome_examples, symbolic_examples
from reasoning_lab.verification.prm.learned import LearnedPRM
from reasoning_lab.verification.prm.mixed import MixedPRM
from reasoning_lab.verification.prm.symbolic import SymbolicPRM

CLEAN = ["Total packed: 3*4=12.", "Mia eats 5, so 12-5=7.", "Left x = 7."]
FLAWED = ["Total packed: 3*4=13.", "Mia eats 5, so 13-5=8.", "Left x = 8."]
FULL = ["u = 12.", "1/4 - 1/12 = 1/6.", "1/v = 1/6.", "v = 6."]


def steps(texts: list[str]) -> list[Step]:
    return [Step(index=i, text=t) for i, t in enumerate(texts)]


@pytest.mark.asyncio
async def test_symbolic_prm_separates_clean_from_flawed():
    prm = SymbolicPRM()
    clean = await prm.score_path("p", steps(CLEAN))
    flawed = await prm.score_path("p", steps(FLAWED))
    assert prm.path_value(clean) > 0.9
    assert prm.path_value(flawed) < 0.4
    assert prm.path_value(clean) > 2 * prm.path_value(flawed)
    assert [s.status for s in flawed][:1] == [VerifyStatus.FAIL]


@pytest.mark.asyncio
async def test_path_score_does_not_prefer_a_truncated_chain():
    """A completed chain must outrank its own prefix.

    The raw step-probability product is length-biased: every extra step, even a
    verified one, multiplies the path score down, so reranking preferred a
    chain that stopped before stating the answer.
    """
    full = SymbolicPRM()
    completed = await full.score_path("p", steps(FULL))
    prefix = await full.score_path("p", steps(FULL[:3]))
    assert full.path_value(completed) > full.path_value(prefix)

    product = SymbolicPRM(aggregator="product")
    assert product.path_value(completed) < product.path_value(prefix)


@pytest.mark.asyncio
async def test_shared_prefix_state_is_reused_across_branches():
    prm = SymbolicPRM()
    prefix = steps(CLEAN[:2])
    await prm.score_step("p", prefix, Step(index=2, text="Left x = 7."))
    before = prm.verify_calls
    await prm.score_step("p", prefix, Step(index=2, text="Left x = 7."))
    assert prm.verify_calls == before + 1  # prefix replay came from the memo, not regeneration
    assert len(prm._states) >= 3


@pytest.mark.asyncio
async def test_null_prm_is_flat():
    prm = NullPRM()
    scores = await prm.score_path("p", steps(CLEAN))
    assert prm.path_value(scores) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_mixed_prm_combines_components():
    mixed = MixedPRM([(SymbolicPRM(), 0.6), (NullPRM(), 0.4)])
    flawed = await mixed.score_path("p", steps(FLAWED))
    clean = await mixed.score_path("p", steps(CLEAN))
    assert mixed.path_value(flawed) < 0.5, "the flat component must not lift a failed chain above 0.5"
    assert mixed.path_value(clean) > mixed.path_value(flawed)
    assert "mixed(" in mixed.name


@pytest.mark.asyncio
async def test_learned_head_trains_and_predicts(problem):
    symbolic = SymbolicPRM()
    examples = []
    for texts, correct in [(CLEAN, True), (FLAWED, False)] * 6:
        chain = steps(texts)
        scores = await symbolic.score_path("p", chain)
        examples += symbolic_examples(problem, chain, scores)
        examples += outcome_examples(problem, chain, scores, correct=correct)

    model, report = LearnedPRM.fit(examples, estimator="logistic", seed=0)
    assert report.n_examples >= 12
    assert report.accuracy >= 0.5
    assert len(report.coefficients) == len(FEATURE_NAMES)

    clean_score = await model.score_step("p", [], Step(index=0, text=CLEAN[0]))
    flawed_score = await model.score_step("p", [], Step(index=0, text=FLAWED[0]))
    assert clean_score.score > flawed_score.score


@pytest.mark.asyncio
async def test_features_have_stable_length():
    row = step_features(steps(CLEAN[:1]), steps(CLEAN)[1], None)
    assert len(row) == len(FEATURE_NAMES)


def test_build_prm_kinds():
    assert isinstance(build_prm("symbolic"), SymbolicPRM)
    assert isinstance(build_prm("none"), NullPRM)
    # no trained head on disk yet -> falls back rather than crashing
    assert isinstance(build_prm("learned", model_path="runs/nope.joblib"), SymbolicPRM)
    with pytest.raises(ValueError):
        build_prm("bogus")
