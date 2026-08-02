"""A measured confidence score for one compression.

The eval harness can tell you that compression preserved 80.8% of key facts,
but only because it has a test set with the answers written down. On an
arbitrary input a user pastes in, there is no answer key - and that is exactly
when they most want to know whether to trust the output.

This module answers that with signals the pipeline already computes, and
nothing else. **No model is called and nothing is estimated by an LLM.** Every
component below is a ratio between two counts taken from the run that just
happened, which is the only kind of number this project reports.

**Lexical retention is measured against the stage-3 survivors, not the raw
input.** This is the single most important design decision here and the first
version got it wrong, which is worth recording. Scored against the raw original,
the 2,400-record log retained 7.7% of its distinct numbers and scored *lowest*
of the four sample contexts - despite being the one with 100% measured fact
survival. The "lost" numbers were timestamp fractions (``00.038``, ``00.040``,
...) and order ids: volatile fields that stage 2 templates and stage 3 collapses
deliberately. The score was punishing the pipeline for working correctly.

Redundancy collapse is near-lossless by construction - the representative keeps
its full text and a count marker records the repetition - whereas budget
eviction deletes content that occurred once. So the honest question is not "what
did the whole pipeline drop" but **"of the facts stage 3 preserved, how many did
stage 5 evict?"** Measured that way the log scores 1.000 and rank correlation
against known fact survival goes from 3/6 concordant pairs (chance) to 5/6.

The five signals, and why each one predicts fact survival:

``number_retention``
    Fraction of the distinct numbers *in the stage-3 survivors* that still
    appear in the compressed text. The strongest single predictor (5/6
    concordant alone), and weighted accordingly: in the shipped eval, 6 of the
    9 facts the model missed were numbers (``240ms``, ``8.4s``, ``900``,
    ``500``).

``identifier_retention``
    Same for identifiers: ``PAYMENT_POOL_SIZE``, ``payments.internal``,
    ``ReadTimeout``. Deliberately given a small weight - it ranks the four
    calibration contexts no better than chance (3/6), because identifier sets
    are dominated by common code tokens that recur throughout a document, so
    retention saturates near 1.0 and carries little signal. Kept because it is
    informative *evidence* when it does drop, not because it discriminates.

``density_retention``
    Token-weighted share of total density mass that survived selection. Asks
    "did we keep the informative material, or just the material that fit?"

``lossless_share``
    Of the tokens removed, how many went to redundancy collapse rather than
    budget eviction. These are not equivalent losses: a dedup'd cluster keeps
    its representative *and* a count marker, so the fact of the repetition
    survives. A budget eviction deletes content that occurred exactly once.
    This is why a 2,400-record log compresses 93.9% at no real cost while a
    postmortem cannot lose 70% without losing something.

``dependency_integrity``
    Whether code that survived still has the definitions it references, or
    whether stage 5 left dangling symbols it could not repair.

**What this score is not.** It is a predictor, not a guarantee. It is checked
against only **four** contexts with known fact survival - far too few to call it
calibrated, and enough that the weights below must be treated as reasoned
defaults rather than fitted parameters. What it does have is a stated failure
mode, an auditable decomposition, and a rank-correlation number that is reported
rather than assumed. A low score can always be explained by pointing at which
component produced it; ``reasons`` carries exactly that.

Re-check the correlation at any time with::

    python -m engine.confidence --calibrate
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .abstractive import _IDENTIFIER, _NUMBER
from .types import Chunk

#: Component weights. Reasoned defaults, not fitted parameters - with four
#: calibration contexts, fitting would be overfitting. Each of the top three
#: independently ranks those four contexts 5/6 concordant with measured fact
#: survival; identifier_retention manages only 3/6 (chance) and is weighted at
#: floor level accordingly. See the module docstring.
WEIGHTS = {
    "number_retention": 0.40,
    "density_retention": 0.25,
    "lossless_share": 0.20,
    "dependency_integrity": 0.10,
    "identifier_retention": 0.05,
}

#: Band thresholds on the combined score. Set so the four calibration contexts
#: do not all collapse into one band - a score that says "low" for everything
#: carries no information even if it ranks correctly.
HIGH, MODERATE = 0.75, 0.50

#: Measured fact survival per context from the shipped eval report, used to
#: sanity-check that this score ranks inputs in the same order. Reproduce with
#: `python -m engine.confidence --calibrate`.
CALIBRATION = {
    "log_incident": 1.00,
    "auth_code": 0.857,
    "tickets": 0.667,
    "postmortem": 0.625,
}


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def _identifiers(text: str) -> set[str]:
    return set(_IDENTIFIER.findall(text))


def _ratio(kept: set[str], original: set[str]) -> float:
    """Retention ratio. An original with none of a token type scores 1.0 -
    nothing could be lost, so nothing was."""
    if not original:
        return 1.0
    return len(original & kept) / len(original)


@dataclass
class Confidence:
    """One compression's trustworthiness, decomposed."""

    score: float = 0.0
    band: str = "unknown"
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    #: Counts behind the ratios, so a reader can check the arithmetic.
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "band": self.band,
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "reasons": self.reasons,
            "evidence": self.evidence,
            "method": (
                "Deterministic. Weighted ratios over counts measured during this "
                "run; no model involved. Predicts fact survival, does not "
                "guarantee it."
            ),
        }


def score_compression(
    original_text: str,
    compressed_text: str,
    chunks: list[Chunk],
    kept: list[Chunk],
    *,
    survivors: list[Chunk] | None = None,
    tokens_removed_by_redundancy: int = 0,
    tokens_removed_total: int = 0,
    broken_dependencies: int = 0,
    repaired_dependencies: int = 0,
) -> Confidence:
    """Score one compression. Pure function over already-measured quantities.

    ``survivors`` is the stage-3 output (post-dedup, pre-selection). Lexical
    retention is measured against it rather than ``original_text`` - see the
    module docstring for why that distinction is the whole ballgame. Falls back
    to the raw original when not supplied, which is the pessimistic reading.
    """
    result = Confidence()

    if not original_text.strip():
        result.band = "unknown"
        result.reasons.append("empty input; nothing to score")
        return result

    # --- lexical retention -------------------------------------------------
    # Baseline is what stage 3 preserved. Volatile fields it deliberately
    # collapsed (timestamps, request ids) are not losses and must not be
    # counted as such.
    baseline = "\n".join(c.text for c in survivors) if survivors else original_text
    original_numbers, kept_numbers = _numbers(baseline), _numbers(compressed_text)
    original_ids, kept_ids = _identifiers(baseline), _identifiers(compressed_text)
    number_retention = _ratio(kept_numbers, original_numbers)
    identifier_retention = _ratio(kept_ids, original_ids)

    # --- density retention -------------------------------------------------
    # Token-weighted: dropping one 233-token chunk matters more than dropping
    # three 14-token ones, even at the same density.
    def mass(items: list[Chunk]) -> float:
        return sum((c.density or 0.0) * c.token_count for c in items)

    total_mass = mass(chunks)
    density_retention = (mass(kept) / total_mass) if total_mass > 0 else 1.0

    # --- lossless share ----------------------------------------------------
    # Redundancy collapse preserves the fact of the repetition (representative
    # + count marker); budget eviction does not. Only meaningful if anything
    # was removed at all.
    if tokens_removed_total > 0:
        lossless_share = min(1.0, tokens_removed_by_redundancy / tokens_removed_total)
    else:
        lossless_share = 1.0

    # --- dependency integrity ----------------------------------------------
    dependency_total = broken_dependencies + repaired_dependencies
    dependency_integrity = (
        repaired_dependencies / dependency_total if dependency_total else 1.0
    )

    result.components = {
        "number_retention": number_retention,
        "identifier_retention": identifier_retention,
        "density_retention": density_retention,
        "lossless_share": lossless_share,
        "dependency_integrity": dependency_integrity,
    }
    result.score = sum(WEIGHTS[k] * v for k, v in result.components.items())
    result.band = (
        "high" if result.score >= HIGH
        else "moderate" if result.score >= MODERATE
        else "low"
    )

    lost_numbers = sorted(original_numbers - kept_numbers)
    lost_ids = sorted(original_ids - kept_ids)
    result.evidence = {
        "numbers_original": len(original_numbers),
        "numbers_kept": len(original_numbers & kept_numbers),
        "identifiers_original": len(original_ids),
        "identifiers_kept": len(original_ids & kept_ids),
        "tokens_removed_total": tokens_removed_total,
        "tokens_removed_by_redundancy": tokens_removed_by_redundancy,
        "chunks_total": len(chunks),
        "chunks_kept": len(kept),
        "example_lost_numbers": lost_numbers[:8],
        "example_lost_identifiers": lost_ids[:8],
    }
    result.reasons = _explain(result.components, result.evidence)
    return result


def _explain(components: dict[str, float], evidence: dict) -> list[str]:
    """Say what is dragging the score down, in the order it matters.

    A bare number is not actionable; "37 of 214 numbers are missing, including
    240 and 8.4" tells a user whether the thing they care about survived.
    """
    reasons: list[str] = []

    lost_numbers = evidence["numbers_original"] - evidence["numbers_kept"]
    if lost_numbers:
        sample = ", ".join(evidence["example_lost_numbers"])
        reasons.append(
            f"{lost_numbers} of {evidence['numbers_original']} distinct numbers "
            f"that survived deduplication were dropped by the budget"
            + (f" (e.g. {sample})" if sample else "")
        )

    lost_ids = evidence["identifiers_original"] - evidence["identifiers_kept"]
    if lost_ids:
        sample = ", ".join(evidence["example_lost_identifiers"])
        reasons.append(
            f"{lost_ids} of {evidence['identifiers_original']} distinct identifiers "
            f"were dropped by the budget"
            + (f" (e.g. {sample})" if sample else "")
        )

    if components["density_retention"] < 0.8:
        reasons.append(
            f"only {components['density_retention']:.0%} of the document's density "
            f"mass survived selection; the budget is forcing out scoring content"
        )

    removed = evidence["tokens_removed_total"]
    if removed and components["lossless_share"] < 0.3:
        reasons.append(
            f"{100 * (1 - components['lossless_share']):.0f}% of the {removed:,} "
            f"removed tokens were unique content evicted by the budget, not "
            f"duplicates collapsed - this input has little redundancy to exploit"
        )
    elif components["lossless_share"] >= 0.8 and removed:
        reasons.append(
            f"{components['lossless_share']:.0%} of removed tokens were redundant "
            f"duplicates, which keep a representative and a count marker"
        )

    if components["dependency_integrity"] < 1.0:
        reasons.append(
            "some kept code references definitions that were dropped and could "
            "not be re-admitted"
        )

    if not reasons:
        reasons.append(
            "every number and identifier that survived deduplication is still present"
        )
    return reasons


# --------------------------------------------------------------------------
# calibration check
# --------------------------------------------------------------------------
def calibrate(budget_ratio: float = 0.30) -> dict:
    """Score the four contexts whose true fact survival the eval measured.

    Reports rank correlation (concordant pairs) rather than a fit statistic:
    with n=4 the only defensible claim is about *ordering*, and even that rests
    on six comparisons. Printed so the number in the docs can be checked rather
    than believed.
    """
    import itertools

    from .config import PROJECT_ROOT
    from .pipeline import CompressionPipeline

    corpus = PROJECT_ROOT / "data" / "sample_corpus"
    files = {
        "log_incident": corpus / "logs" / "checkout_service.log",
        "auth_code": corpus / "code" / "auth_service.py",
        "tickets": corpus / "support" / "tickets.txt",
        "postmortem": corpus / "docs" / "incident_postmortem.md",
    }
    pipeline = CompressionPipeline()
    rows = []
    for key, path in files.items():
        text = path.read_text(encoding="utf-8")
        outcome = pipeline.compress(
            text, path.name, budget_ratio=budget_ratio, fast_mode=True
        )
        rows.append((key, CALIBRATION[key], outcome.confidence))

    pairs = list(itertools.combinations(rows, 2))
    concordant = sum(1 for a, b in pairs if (a[1] - b[1]) * (a[2].score - b[2].score) > 0)
    return {
        "budget_ratio": budget_ratio,
        "concordant_pairs": concordant,
        "total_pairs": len(pairs),
        "rows": [
            {
                "context": key,
                "measured_fact_survival": actual,
                "confidence_score": round(conf.score, 4),
                "band": conf.band,
                "components": {k: round(v, 4) for k, v in conf.components.items()},
            }
            for key, actual, conf in rows
        ],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="engine.confidence", description=__doc__)
    parser.add_argument(
        "--calibrate", action="store_true",
        help="score the four contexts with known fact survival and report rank correlation",
    )
    parser.add_argument("--budget", type=float, default=0.30)
    args = parser.parse_args(argv)

    if not args.calibrate:
        parser.print_help()
        return 0

    report = calibrate(args.budget)
    print(f"{'context':<14}{'measured':>11}{'predicted':>12}{'band':>11}")
    for row in report["rows"]:
        print(
            f"  {row['context']:<12}{row['measured_fact_survival']:>10.1%}"
            f"{row['confidence_score']:>12.3f}{row['band']:>11}"
        )
    print(
        f"\n  {report['concordant_pairs']}/{report['total_pairs']} concordant pairs "
        f"(does the score rank inputs the way measured fact survival does?)"
    )
    print("  n=4: this checks ordering only, and is not a calibration claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
