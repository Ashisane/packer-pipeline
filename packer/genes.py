"""
genes.py — Gene set selection for C. elegans connectome GNN features.

Two complementary strategies:

  Strategy 1 — Curated guidance genes:  A structured registry of axon
  guidance and synaptic specificity genes organised by pathway.  These
  genes have known roles in determining synaptic partner choice and are
  the mechanistically interpretable subset of the feature space.  Using
  them allows W-matrix entries to be attributed to specific molecular
  programmes (e.g. 'netrin receptor pair avoidance').

  Strategy 2 — Data-driven high-variance genes:  Genes whose expression
  varies most across neuron subtypes (not across individual cells) are
  selected because high inter-subtype variance implies the gene encodes
  neuronal identity that distinguishes one class from another — exactly
  the signal needed for predicting class-specific connectivity.

The final feature set is the union of both strategies.  Storing gene
origin (curated / variance / both) enables pathway-level interpretation
of learned W matrices.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_N_VARIANCE_GENES: int = 200
MIN_VARIANCE_THRESHOLD: float = 1e-6


class GeneOrigin(str, Enum):
    """Provenance of a gene in the final feature set."""
    CURATED = "curated"
    VARIANCE = "variance"
    BOTH = "both"


@dataclass(frozen=True)
class GuidanceGene:
    """
    Metadata for one curated axon guidance / synaptic specificity gene.

    wormbase_id is the WBGene identifier used as var_name in the Packer h5ad.
    It is looked up at runtime from the gene-name→ID map; 'unknown' means
    the gene name was not found in the Packer variable index.
    """
    name: str
    pathway: str
    role: str
    wormbase_id: str


@dataclass
class GeneSetResult:
    """Output of GeneSelector.select()."""
    gene_ids: list[str]
    gene_names: list[str]
    origin: dict[str, GeneOrigin]
    curated_found: list[GuidanceGene]
    curated_missing: list[GuidanceGene]
    n_variance_selected: int
    pathway_index: dict[str, list[str]]


_REGISTRY: list[tuple[str, str, str]] = [
    ("unc-6",   "Netrin",      "ligand"),
    ("unc-5",   "Netrin",      "receptor"),
    ("unc-40",  "Netrin",      "receptor"),
    ("unc-129", "Netrin",      "ligand"),
    ("vab-1",   "Ephrin",      "receptor"),
    ("vab-2",   "Ephrin",      "ligand"),
    ("efn-1",   "Ephrin",      "ligand"),
    ("efn-2",   "Ephrin",      "ligand"),
    ("efn-3",   "Ephrin",      "ligand"),
    ("efn-4",   "Ephrin",      "ligand"),
    ("lin-44",  "Wnt",         "ligand"),
    ("egl-20",  "Wnt",         "ligand"),
    ("mom-5",   "Wnt",         "receptor"),
    ("lin-17",  "Wnt",         "receptor"),
    ("cfz-2",   "Wnt",         "receptor"),
    ("mig-1",   "Wnt",         "effector"),
    ("mig-5",   "Wnt",         "effector"),
    ("slt-1",   "Slit/Robo",   "ligand"),
    ("sax-3",   "Slit/Robo",   "receptor"),
    ("syd-2",   "IgCAM",       "scaffold"),
    ("rig-3",   "IgCAM",       "effector"),
    ("rig-6",   "IgCAM",       "effector"),
    ("nlg-1",   "Neurexin",    "scaffold"),
    ("nrx-1",   "Neurexin",    "scaffold"),
    ("syg-1",   "Specificity", "receptor"),
    ("syg-2",   "Specificity", "ligand"),
    ("nid-1",   "Specificity", "scaffold"),
    ("smp-1",   "Semaphorin",  "ligand"),
    ("smp-2",   "Semaphorin",  "ligand"),
    ("plx-1",   "Semaphorin",  "receptor"),
    ("plx-2",   "Semaphorin",  "receptor"),
    ("cdh-3",   "Cadherin",    "receptor"),
    ("cdh-4",   "Cadherin",    "receptor"),
    ("hmr-1",   "Cadherin",    "receptor"),
    ("sax-7",   "Cadherin",    "scaffold"),
]


class GeneSelector:
    """
    Selects the gene feature set from curated guidance genes and data-driven
    high-variance genes.

    Parameters
    ----------
    gene_id_to_name : dict[str, str]
        WormBase ID → gene name map from PackerDataset.gene_name_map().
    """

    def __init__(self, gene_id_to_name: dict[str, str]) -> None:
        self._id_to_name: dict[str, str] = gene_id_to_name
        self._name_to_id: dict[str, str] = {v: k for k, v in gene_id_to_name.items()}

    def _resolve_curated(self) -> tuple[list[GuidanceGene], list[GuidanceGene]]:
        """
        Match curated registry entries against the Packer gene index.

        Returns (found, missing).  Genes not present in Packer are documented
        in 'missing' rather than silently dropped; they may appear in the
        Packer data under a different alias or may be genuinely absent from
        the embryonic atlas.
        """
        found: list[GuidanceGene] = []
        missing: list[GuidanceGene] = []

        for name, pathway, role in _REGISTRY:
            wbid = self._name_to_id.get(name, "")
            gene = GuidanceGene(
                name=name,
                pathway=pathway,
                role=role,
                wormbase_id=wbid if wbid else "unknown",
            )
            if wbid:
                found.append(gene)
                log.debug("Curated gene found: %s (%s)", name, wbid)
            else:
                missing.append(gene)
                log.warning(
                    "Curated gene '%s' not found in Packer var_names; "
                    "it may be absent from this embryonic atlas or listed "
                    "under a different alias.",
                    name,
                )

        log.info(
            "Curated genes: %d found, %d missing from Packer",
            len(found), len(missing),
        )
        return found, missing

    def _select_high_variance(
        self,
        subtype_means: np.ndarray,
        all_gene_ids: list[str],
        n_genes: int,
    ) -> list[str]:
        """
        Select top-n genes by variance of mean expression across neuron subtypes.

        Variance is computed across subtypes (rows of subtype_means), not across
        individual cells.  This measures how much a gene distinguishes neuron
        classes from each other — the biologically relevant axis for predicting
        class-specific connectivity.
        """
        variances = subtype_means.var(axis=0)
        top_idx = np.argsort(variances)[::-1][:n_genes]
        selected = [all_gene_ids[i] for i in top_idx if variances[i] >= MIN_VARIANCE_THRESHOLD]

        log.info(
            "Variance selection: %d genes with var ≥ %g (requested %d)",
            len(selected), MIN_VARIANCE_THRESHOLD, n_genes,
        )
        return selected

    def select(
        self,
        subtype_means: np.ndarray,
        all_gene_ids: list[str],
        n_variance_genes: int = DEFAULT_N_VARIANCE_GENES,
    ) -> GeneSetResult:
        """
        Compute the final gene feature set and return structured metadata.

        Parameters
        ----------
        subtype_means
            (N_subtypes, N_all_genes) array of mean expression per subtype.
            Typically computed before subsetting to avoid the chicken-and-egg
            problem of needing gene selection before the expression matrix.
        all_gene_ids
            WormBase IDs corresponding to columns of subtype_means.
        n_variance_genes
            How many top-variance genes to include from Strategy 2.
        """
        curated_found, curated_missing = self._resolve_curated()
        curated_ids = {g.wormbase_id for g in curated_found}

        variance_ids = self._select_high_variance(subtype_means, all_gene_ids, n_variance_genes)

        all_selected_ids: list[str] = []
        origin: dict[str, GeneOrigin] = {}

        for gid in curated_ids:
            if gid in all_gene_ids:
                all_selected_ids.append(gid)
                origin[gid] = GeneOrigin.CURATED

        for gid in variance_ids:
            if gid in origin:
                origin[gid] = GeneOrigin.BOTH
            else:
                all_selected_ids.append(gid)
                origin[gid] = GeneOrigin.VARIANCE

        pathway_index: dict[str, list[str]] = {}
        for gene in curated_found:
            pathway_index.setdefault(gene.pathway, []).append(gene.wormbase_id)

        gene_names = [self._id_to_name.get(gid, gid) for gid in all_selected_ids]

        n_both = sum(1 for o in origin.values() if o == GeneOrigin.BOTH)
        log.info(
            "Final gene set: %d total (%d curated-only, %d variance-only, %d in both)",
            len(all_selected_ids),
            len(curated_ids) - n_both,
            len(all_selected_ids) - len(curated_ids),
            n_both,
        )

        return GeneSetResult(
            gene_ids=all_selected_ids,
            gene_names=gene_names,
            origin=origin,
            curated_found=curated_found,
            curated_missing=curated_missing,
            n_variance_selected=len(variance_ids),
            pathway_index=pathway_index,
        )
