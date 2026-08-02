"""Entity and identifier density, with a regex fallback for when spaCy is absent.

Stage 4 asks "how much *concrete* information does this chunk carry?". Named
entities, numbers, identifiers and function signatures are the observable proxy:
a paragraph naming a service, a version and a latency figure is carrying facts,
while one that says "the system was then reviewed by the team" is carrying
almost none.

spaCy's ``en_core_web_sm`` supplies the NER signal. If the pipeline is not
installed - or the input is 400k characters of log text where loading it is not
worth the wall-clock - the regex proxies below run instead. They are weaker but
directionally identical, and the stage reports which one ran, so a number is
never presented as NER-backed when it was not.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

# --- regex proxies ---------------------------------------------------------
# A bare number, NOT counting one glued to a unit or embedded in an identifier
# or version string - those are handled below and would otherwise be counted
# twice, or half-matched.
#
# The previous pattern (`\b\d[\d,._]*\b`) silently lost every measurement:
# "latency rose from 240ms to 8.4s" matched only `8.` - `240` was invisible
# because the trailing word boundary failed against the unit. Chunks stating
# exactly the facts questions ask about therefore scored as ordinary prose,
# and stage 5 dropped them. Both `240ms` and `8.4s` were among the facts the
# benchmark measured as lost.
_NUMBER = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w.]*[A-Za-z_])")

# A number with a unit attached. Weighted above a bare digit: "8000ms" is a
# threshold someone will ask about, "3" on its own usually is not.
_MEASUREMENT = re.compile(
    r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?\s?"
    r"(?:ms|us|ns|s|m|h|d|kb|mb|gb|tb|kib|mib|gib|b|%|x|px|rps|qps|req/s)\b",
    re.I,
)
_SNAKE_CASE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_CAMEL_CASE = re.compile(r"\b[a-z]+[A-Z]\w*\b")
_CONSTANT = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)*\b")
_SIGNATURE = re.compile(
    r"\b(?:def|class|function|async\s+def|interface|struct|impl)\s+\w+"
    r"|\b\w+\s*\([^)]*\)\s*(?:->|\{|:)"
)
_DOTTED = re.compile(r"\b\w+(?:\.\w+){1,}\b")  # module.attr, requests.exceptions.X
_URLISH = re.compile(r"https?://\S+|/(?:[\w.-]+/)+[\w.-]+|[\w.+-]+@[\w-]+\.[\w.]+")
_PROPER_NOUN = re.compile(r"\b[A-Z][a-z]{2,}\b")

# Relative worth of each observed feature. Entities and signatures are the
# strongest evidence that a chunk states a fact rather than describes one.
_WEIGHTS = {
    "entities": 2.0,
    "signatures": 3.0,
    "numbers": 1.0,
    # A measurement is a stronger fact signal than a bare digit - it is the
    # shape of a threshold, a duration or a limit, which is what an incident
    # question asks about.
    "measurements": 2.5,
    "identifiers": 1.0,
    "dotted": 0.75,
    "urls": 1.0,
    "proper_nouns": 0.5,
}


@dataclass
class EntityCounts:
    entities: int = 0
    signatures: int = 0
    numbers: int = 0
    measurements: int = 0
    identifiers: int = 0
    dotted: int = 0
    urls: int = 0
    proper_nouns: int = 0

    def weighted(self) -> float:
        return (
            _WEIGHTS["entities"] * self.entities
            + _WEIGHTS["signatures"] * self.signatures
            + _WEIGHTS["numbers"] * self.numbers
            + _WEIGHTS["measurements"] * self.measurements
            + _WEIGHTS["identifiers"] * self.identifiers
            + _WEIGHTS["dotted"] * self.dotted
            + _WEIGHTS["urls"] * self.urls
            + _WEIGHTS["proper_nouns"] * self.proper_nouns
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "entities": self.entities,
            "signatures": self.signatures,
            "numbers": self.numbers,
            "measurements": self.measurements,
            "identifiers": self.identifiers,
            "dotted": self.dotted,
            "urls": self.urls,
            "proper_nouns": self.proper_nouns,
        }


def regex_counts(text: str) -> EntityCounts:
    """Model-free approximation of the NER signal."""
    return EntityCounts(
        entities=0,
        signatures=len(_SIGNATURE.findall(text)),
        numbers=len(_NUMBER.findall(text)),
        measurements=len(_MEASUREMENT.findall(text)),
        identifiers=len(_SNAKE_CASE.findall(text))
        + len(_CAMEL_CASE.findall(text))
        + len(_CONSTANT.findall(text)),
        dotted=len(_DOTTED.findall(text)),
        urls=len(_URLISH.findall(text)),
        proper_nouns=len(_PROPER_NOUN.findall(text)),
    )


class EntityScorer:
    """Counts informative features per chunk. Never raises."""

    def __init__(self, model: str = "en_core_web_sm", max_chars: int = 400_000) -> None:
        self.model_name = model
        self.max_chars = max_chars
        self._nlp = None
        self._loaded = False
        self._error: str | None = None
        self.backend = "regex"

    def _load(self):
        if self._loaded:
            return self._nlp
        self._loaded = True
        try:
            import spacy

            # Only NER is needed; excluding the parser and lemmatizer roughly
            # triples throughput and we never use their output.
            self._nlp = spacy.load(
                self.model_name,
                exclude=["parser", "lemmatizer", "attribute_ruler", "senter"],
            )
            self.backend = f"spacy:{self.model_name}"
        except Exception as exc:
            self._error = str(exc)
            log.warning(
                "spaCy model %s unavailable (%s); using regex entity proxies",
                self.model_name,
                exc,
            )
            self._nlp = None
        return self._nlp

    @property
    def available(self) -> bool:
        return self._load() is not None

    def count_all(self, texts: list[str]) -> tuple[list[EntityCounts], str]:
        """Count features for every text. Returns (counts, backend actually used)."""
        if not texts:
            return [], self.backend

        total_chars = sum(len(t) for t in texts)
        if total_chars > self.max_chars:
            log.info(
                "input is %d chars (> %d); using regex proxies instead of spaCy",
                total_chars,
                self.max_chars,
            )
            return [regex_counts(t) for t in texts], "regex:oversized"

        nlp = self._load()
        if nlp is None:
            return [regex_counts(t) for t in texts], "regex:unavailable"

        try:
            counts: list[EntityCounts] = []
            for text, doc in zip(texts, nlp.pipe(texts, batch_size=64)):
                # Start from the regex proxies - they catch code constructs that
                # a prose-trained NER model has no labels for - then layer the
                # real named entities on top.
                base = regex_counts(text)
                base.entities = len(doc.ents)
                counts.append(base)
            return counts, self.backend
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("spaCy pipe failed (%s); falling back to regex", exc)
            self._error = str(exc)
            return [regex_counts(t) for t in texts], "regex:failed"

    def describe(self) -> dict:
        return {
            "backend": self.backend,
            "model": self.model_name,
            "available": self.available,
            "error": self._error,
        }
