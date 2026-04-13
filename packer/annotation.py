"""
annotation.py — Rule-based mapping from Witvliet individual neuron names
                to Packer 2019 cell-subtype annotation labels.

The central challenge: Packer annotates cells by neuron class ('AWC', 'DA'),
while Witvliet tracks individual neurons ('AWCL', 'AWCR', 'DA1'…'DA9').
A purely hardcoded lookup table would need ~500 entries, most of which follow
predictable suffix-stripping patterns.  This module encodes those patterns
as ordered rules so the mapping generalises to any neuron list without
manual curation of each entry.

The bilateral symmetry assumption is documented explicitly: the model
cannot distinguish AWCL from AWCR by expression because Packer does not
resolve left/right for most neuron classes.
"""

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

_DIRECTIONAL_SUFFIXES: tuple[str, ...] = ("DL", "DR", "VL", "VR")

_DEFAULT_MANUAL_EXCEPTIONS: dict[str, str] = {}

RULE_NAMES: dict[str, str] = {
    "direct": "direct match (name identical in both datasets)",
    "A": "strip bilateral suffix (L or R)",
    "B": "strip directional suffix (DL/DR/VL/VR ± trailing D or V)",
    "C": "strip numeric suffix",
    "D": "strip L/R then trailing D or V (two-level bilateral)",
    "E": "manual exception",
    "none": "no mapping found",
}


@dataclass
class MappingReport:
    """
    Structured summary of the neuron mapping result.

    Designed to be JSON-serialisable (all values are primitives or plain
    dicts/lists) so the build_atlas script can write it to the manifest.
    """

    n_neurons_total: int
    n_neurons_mapped: int
    n_neurons_unmapped: int
    coverage_fraction: float
    by_rule: dict[str, list[str]]
    bilateral_pairs: dict[str, list[str]]
    unmapped: list[str]
    rule_descriptions: dict[str, str]


def _generate_candidates(name: str) -> list[tuple[str, str]]:
    """
    Produce an ordered list of (candidate_packer_subtype, rule_name) for one
    Witvliet neuron name, without checking against any subtype set.

    Ordering encodes priority: more specific rules first so that a neuron
    that satisfies multiple rules is classified by the most predictive one.
    """
    candidates: list[tuple[str, str]] = []

    if len(name) > 1 and name[-1] in ("L", "R"):
        candidates.append((name[:-1], "A"))

    for dsuf in _DIRECTIONAL_SUFFIXES:
        if name.endswith(dsuf) and len(name) > len(dsuf):
            base = name[: -len(dsuf)]
            candidates.append((base, "B"))
            if base and base[-1] in ("D", "V"):
                candidates.append((base[:-1], "B"))

    base_no_digits = name.rstrip("0123456789")
    if base_no_digits != name and base_no_digits:
        candidates.append((base_no_digits, "C"))
        if base_no_digits[-1] in ("L", "R"):
            candidates.append((base_no_digits[:-1], "C"))

    if len(name) > 2 and name[-1] in ("L", "R"):
        base_lr = name[:-1]
        if base_lr[-1] in ("D", "V"):
            candidates.append((base_lr[:-1], "D"))

    return candidates


class NeuronMapper:
    """
    Maps Witvliet individual neuron names to Packer 2019 cell-subtype labels.

    Construction is separated from mapping so you can inspect the Packer
    subtype set before committing to any particular mapping strategy.

    Parameters
    ----------
    packer_subtypes : collection of str
        All subtype labels present in the Packer annotation column.
        Typically obtained from PackerDataset.adata.obs[ann_col].unique().
    manual_exceptions : dict[str, str], optional
        Override mapping for specific neurons that defeat rules A-D.
        Merged with the built-in _DEFAULT_MANUAL_EXCEPTIONS; caller-supplied
        entries take precedence.  Populate only after empirically verifying
        that no rule resolves the neuron.
    """

    def __init__(
        self,
        packer_subtypes: list[str] | set[str],
        manual_exceptions: Optional[dict[str, str]] = None,
    ) -> None:
        self._subtypes: frozenset[str] = frozenset(
            s for s in packer_subtypes if isinstance(s, str) and s != "nan"
        )
        self._manual: dict[str, str] = {**_DEFAULT_MANUAL_EXCEPTIONS}
        if manual_exceptions:
            self._manual.update(manual_exceptions)

        log.info("NeuronMapper initialised with %d Packer subtypes", len(self._subtypes))

    def map_one(self, name: str) -> tuple[str, str]:
        """
        Return (packer_subtype, rule) for a single Witvliet neuron name.

        Returns ('', 'none') when no mapping can be found.  The rule string
        identifies which naming convention resolved the match, enabling
        downstream auditing of how reliable each mapping is (direct matches
        are more trustworthy than Rule-D two-step strips).
        """
        if name in self._subtypes:
            return name, "direct"

        for candidate, rule in _generate_candidates(name):
            if candidate in self._subtypes:
                return candidate, rule

        if name in self._manual:
            mapped = self._manual[name]
            if mapped in self._subtypes:
                return mapped, "E"
            log.warning(
                "Manual exception '%s' → '%s' but '%s' not in Packer subtypes",
                name,
                mapped,
                mapped,
            )

        return "", "none"

    def build_mapping(
        self, witvliet_neurons: list[str]
    ) -> dict[str, str]:
        """
        Map a list of Witvliet neuron names and return {witvliet_name: packer_subtype}.

        Only successfully mapped neurons appear as keys.  Unmapped neurons are
        logged at WARNING level and excluded; callers should use report() to
        obtain the full unmapped list.
        """
        mapping: dict[str, str] = {}
        for name in witvliet_neurons:
            subtype, rule = self.map_one(name)
            if subtype:
                mapping[name] = subtype
            else:
                log.warning("No Packer subtype found for Witvliet neuron '%s'", name)
        return mapping

    def report(self, witvliet_neurons: list[str]) -> MappingReport:
        """
        Build a full mapping and return a structured coverage report.

        Bilateral pairs are reported explicitly to make the left/right
        ambiguity visible in downstream analysis and the final manifest.
        """
        by_rule: dict[str, list[str]] = {r: [] for r in RULE_NAMES}
        bilateral_pairs: dict[str, list[str]] = {}
        unmapped: list[str] = []

        for name in witvliet_neurons:
            subtype, rule = self.map_one(name)
            if subtype:
                by_rule[rule].append(name)
                bilateral_pairs.setdefault(subtype, []).append(name)
            else:
                by_rule["none"].append(name)
                unmapped.append(name)

        n_mapped = len(witvliet_neurons) - len(unmapped)

        log.info(
            "Mapping: %d/%d neurons resolved (%.1f%%)",
            n_mapped,
            len(witvliet_neurons),
            100 * n_mapped / max(1, len(witvliet_neurons)),
        )
        log.info(
            "By rule: %s",
            {r: len(v) for r, v in by_rule.items() if v},
        )
        if unmapped:
            log.warning("%d neurons unmapped: %s", len(unmapped), unmapped[:10])

        return MappingReport(
            n_neurons_total=len(witvliet_neurons),
            n_neurons_mapped=n_mapped,
            n_neurons_unmapped=len(unmapped),
            coverage_fraction=n_mapped / max(1, len(witvliet_neurons)),
            by_rule={r: v for r, v in by_rule.items()},
            bilateral_pairs=bilateral_pairs,
            unmapped=unmapped,
            rule_descriptions=RULE_NAMES,
        )
