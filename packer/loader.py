"""
loader.py — Lazy loading and validation of the Packer 2019 h5ad atlas.

The Packer 2019 file is ~2 GB and contains 89,701 cells × 20,222 genes.
We defer matrix loading until explicitly requested because downstream
steps (mapping, gene selection) often only need the obs metadata.
Column names are auto-detected so this module remains robust if the
h5ad was re-exported with slightly different field names.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anndata
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PACKER_DOWNLOAD_URL = "https://data.caltech.edu/records/b1kj4-nh475"

_ANN_COL_CANDIDATES: list[str] = [
    "cell_subtype",
    "cell.type",
    "assigned.cell.type",
    "plot_cell_type",
    "plot.cell.type",
]

_TIME_COL_CANDIDATES: list[str] = [
    "embryo_time",
    "raw_embryo_time",
    "embryo_time_bin",
]

_REQUIRED_OBS_COLS: frozenset[str] = frozenset({"cell_type", "n_umi"})

_NEURON_SUBTYPE_PATTERN = re.compile(r"^[A-Z]{2,8}[0-9]?$")


@dataclass
class DatasetSummary:
    """Structured validation summary returned by PackerDataset.validate()."""

    n_cells: int
    n_genes: int
    n_neuron_cells: int
    obs_fields: list[str]
    var_fields: list[str]
    cell_type_counts: dict[str, int]
    embryo_time_range: tuple[float, float]
    annotation_column: str
    time_column: str
    missing_required_cols: list[str]


class PackerDataset:
    """
    Lazy-loading wrapper for the Packer 2019 C. elegans scRNA-seq atlas.

    The underlying AnnData object is loaded on first access to any property
    that needs it, keeping import time fast for scripts that only inspect
    the mapping or gene list.

    Parameters
    ----------
    path : str or Path
        Filesystem path to the .h5ad file.  Raises FileNotFoundError with
        a download URL if the file is absent.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._adata: Optional[anndata.AnnData] = None
        self._ann_col: Optional[str] = None
        self._time_col: Optional[str] = None

        if not self._path.exists():
            raise FileNotFoundError(
                f"Packer h5ad file not found at '{self._path}'. "
                f"Download the file from {PACKER_DOWNLOAD_URL} and place it "
                f"at the configured path before running the pipeline."
            )

    @property
    def adata(self) -> anndata.AnnData:
        """Return the loaded AnnData object, loading from disk on first access."""
        if self._adata is None:
            log.info("Loading '%s' (~2 GB, first access may take ~10 s)", self._path)
            self._adata = anndata.read_h5ad(self._path)
            self._detect_columns()
            log.info(
                "Loaded %d cells × %d genes", self._adata.n_obs, self._adata.n_vars
            )
        return self._adata

    def _detect_columns(self) -> None:
        """
        Identify which obs columns hold the cell-type annotation and embryo time.

        Auto-detecting rather than hardcoding makes the loader tolerant of
        h5ad files that were re-exported or renamed.  Detection logs what
        was found so results are reproducible.
        """
        obs = self._adata.obs

        for col in _ANN_COL_CANDIDATES:
            if col in obs.columns:
                self._ann_col = col
                log.info("Annotation column: '%s'", col)
                break

        for col in _TIME_COL_CANDIDATES:
            if col in obs.columns:
                try:
                    obs[col].astype(float)
                    self._time_col = col
                    log.info("Embryo-time column: '%s'", col)
                    break
                except (ValueError, TypeError):
                    continue

        if self._ann_col is None:
            raise ValueError(
                f"Could not identify a cell-type annotation column in obs. "
                f"Tried: {_ANN_COL_CANDIDATES}. "
                f"Available columns: {list(obs.columns)}"
            )
        if self._time_col is None:
            raise ValueError(
                f"Could not identify a numeric embryo-time column in obs. "
                f"Tried: {_TIME_COL_CANDIDATES}. "
                f"Available columns: {list(obs.columns)}"
            )

    @property
    def annotation_column(self) -> str:
        """Name of the obs column holding terminal cell-type annotations."""
        _ = self.adata
        return self._ann_col  # type: ignore[return-value]

    @property
    def time_column(self) -> str:
        """Name of the obs column holding embryo time in minutes p.f.c."""
        _ = self.adata
        return self._time_col  # type: ignore[return-value]

    @property
    def n_cells(self) -> int:
        return self.adata.n_obs

    @property
    def n_genes(self) -> int:
        return self.adata.n_vars

    @property
    def obs_fields(self) -> list[str]:
        return list(self.adata.obs.columns)

    @property
    def var_fields(self) -> list[str]:
        return list(self.adata.var.columns)

    def neuron_view(self) -> anndata.AnnData:
        """
        Return the subset of adata containing only neuron-annotated cells.

        Packer 2019 tissue annotation is inconsistent: many individually-named
        neuron subtypes (AIA, AIB, AIY etc.) have cell_type = 'not provided'
        because they come from a different dissection protocol than the bulk
        cell-type-labelled cells.  The reliable annotation column for these
        cells is cell_subtype, which contains the neuron class name.

        Strategy: include any cell whose cell_subtype field is a short all-caps
        identifier (≤ 8 chars, matches the pattern of C. elegans neuron names)
        OR whose cell_type field contains 'neuron'.  This maximises recall of
        neuron cells while excluding non-neural subtypes (which use long
        descriptive labels like 'BWM_head_row_1').
        """
        ad = self.adata
        obs = ad.obs

        ann = obs[self._ann_col].astype(str)
        subtype_mask = ann.map(lambda x: bool(_NEURON_SUBTYPE_PATTERN.match(x)))

        tissue_col: Optional[str] = None
        for col in ("tissue", "cell_type", "lineage"):
            if col in obs.columns:
                tissue_col = col
                break

        tissue_mask = obs.index.map(lambda _: False)
        if tissue_col is not None:
            tissue_mask = obs[tissue_col].astype(str).str.contains(
                "neuron", case=False, na=False
            )

        mask = subtype_mask | tissue_mask

        log.info(
            "neuron_view: %d cells (subtype-pattern: %d, tissue-label: %d)",
            int(mask.sum()), int(subtype_mask.sum()), int(tissue_mask.sum()),
        )
        return ad[mask].copy()

    def gene_name_map(self) -> dict[str, str]:
        """
        Return a dict mapping WormBase gene ID → gene name (e.g. WBGene00006849 → unc-6).

        The Packer h5ad uses WormBase IDs as var_names and stores human-readable
        names in the 'gene_name' column of var.  This map allows the rest of the
        pipeline to work with familiar gene names while keeping the matrix indexed
        by stable WormBase IDs.
        """
        ad = self.adata
        if "gene_name" in ad.var.columns:
            return dict(zip(ad.var_names, ad.var["gene_name"].astype(str)))
        log.warning(
            "No 'gene_name' column in var; gene IDs will be used as names."
        )
        return {g: g for g in ad.var_names}

    def gene_id_map(self) -> dict[str, str]:
        """
        Return a dict mapping gene name → WormBase gene ID (inverse of gene_name_map).

        Used by GeneSelector to look up guidance genes by familiar name.
        """
        return {v: k for k, v in self.gene_name_map().items()}

    def validate(self) -> DatasetSummary:
        """
        Check data integrity and return a structured summary.

        Does not raise on soft failures (missing optional columns);
        those are surfaced in the returned DatasetSummary for the caller
        to decide whether to abort or continue.
        """
        ad = self.adata
        obs = ad.obs

        missing = [c for c in _REQUIRED_OBS_COLS if c not in obs.columns]
        if missing:
            log.warning("Required obs columns missing: %s", missing)

        try:
            n_neuron_cells = len(self.neuron_view())
        except Exception as exc:
            log.warning("neuron_view() failed during validation: %s", exc)
            n_neuron_cells = 0

        cell_type_counts = (
            obs[self._ann_col].value_counts().to_dict()
            if self._ann_col in obs.columns
            else {}
        )

        t = obs[self._time_col].astype(float)

        return DatasetSummary(
            n_cells=ad.n_obs,
            n_genes=ad.n_vars,
            n_neuron_cells=n_neuron_cells,
            obs_fields=list(obs.columns),
            var_fields=list(ad.var.columns),
            cell_type_counts={k: int(v) for k, v in cell_type_counts.items()},
            embryo_time_range=(float(t.min()), float(t.max())),
            annotation_column=self._ann_col or "",
            time_column=self._time_col or "",
            missing_required_cols=missing,
        )
