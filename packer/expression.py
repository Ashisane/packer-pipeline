"""
expression.py — Constructs the (N_neurons × N_genes) feature matrix X
                where each row is the expression profile of one C. elegans
                neuron at terminal differentiation.

"Terminal differentiation" is operationalised per-subtype rather than with
a global time cutoff because different neuron classes complete differentiation
at different absolute times in the embryo.  Using a global cutoff (e.g. all
cells after 400 min) would mix post-mitotic neurons with progenitors for
early-differentiating classes like the touch neurons.

The per-subtype 75th-percentile cutoff (TERMINAL_DIFF_PERCENTILE) is a
principled approximation: it selects the latest-time cells within each class,
which are most likely to have reached the terminal transcriptional state.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

log = logging.getLogger(__name__)

TERMINAL_DIFF_PERCENTILE: int = 75
MIN_CELLS_PER_TYPE: int = 3

MARKER_GENES: dict[str, list[str]] = {
    "AWC": ["str-2", "ceh-36"],
    "ASE": ["gcy-5", "gcy-6"],
    "AVA": ["rig-3", "glr-1"],
    "AIY": ["ttx-3", "lim-6"],
    "DA":  ["unc-3", "acr-2"],
}


@dataclass
class NeuronProfile:
    """Metadata about how one neuron's expression row was constructed."""

    witvliet_name: str
    packer_subtype: str
    n_cells_total: int
    n_cells_used: int
    time_threshold: float
    used_fallback: bool


@dataclass
class MarkerCheckResult:
    """
    Biological validation of one marker gene in its expected neuron type.

    fold_change > 2 is consistent with correct cell-type resolution.
    gene_found = False means the gene is absent from the selected feature set,
    not from the atlas; check_markers is a diagnostic, not a feature filter.
    """

    neuron_type: str
    marker_gene: str
    expression_in_type: float
    expression_in_others: float
    fold_change: float
    gene_found: bool


@dataclass
class ExpressionMatrixResult:
    """Full output of ExpressionMatrix.build()."""

    X: np.ndarray
    neuron_names: list[str]
    gene_names: list[str]
    profiles: dict[str, NeuronProfile]
    unmapped_neurons: list[str]


class ExpressionMatrix:
    """
    Constructs a per-neuron expression matrix from Packer 2019.

    Parameters
    ----------
    dataset : anndata.AnnData
        The neuron-subset view from PackerDataset.neuron_view().
    mapping : dict[str, str]
        {witvliet_name: packer_subtype} from NeuronMapper.build_mapping().
    ann_col : str
        obs column holding cell subtype labels.
    time_col : str
        obs column holding embryo time in minutes p.f.c.
    gene_id_to_name : dict[str, str], optional
        Maps WormBase IDs (var_names) to familiar gene names.
        Obtained from PackerDataset.gene_name_map().  If not provided,
        WormBase IDs are used as column labels.
    """

    def __init__(
        self,
        dataset: anndata.AnnData,
        mapping: dict[str, str],
        ann_col: str,
        time_col: str,
        gene_id_to_name: Optional[dict[str, str]] = None,
    ) -> None:
        self._ad = dataset
        self._mapping = mapping
        self._ann_col = ann_col
        self._time_col = time_col
        self._gene_id_to_name = gene_id_to_name or {g: g for g in dataset.var_names}

        self._subtype_to_neurons: dict[str, list[str]] = {}
        for wn, pt in mapping.items():
            self._subtype_to_neurons.setdefault(pt, []).append(wn)

    def _time_threshold(self, subtype_times: pd.Series) -> float:
        """
        Per-subtype terminal differentiation time threshold.

        Using numpy.percentile per subtype (rather than a global cutoff) is
        the key design choice: it means 'late cells for this neuron type'
        rather than 'cells past an arbitrary absolute age'.
        """
        return float(np.percentile(subtype_times.astype(float), TERMINAL_DIFF_PERCENTILE))

    def _mean_expression(self, cell_indices: np.ndarray) -> np.ndarray:
        """Dense mean expression for a boolean or integer index array."""
        block = self._ad.X[cell_indices, :]
        if sp.issparse(block):
            return np.asarray(block.mean(axis=0)).ravel().astype(np.float32)
        return block.mean(axis=0).ravel().astype(np.float32)

    def build(
        self,
        target_neurons: list[str],
        gene_ids: Optional[list[str]] = None,
    ) -> ExpressionMatrixResult:
        """
        Build the expression matrix for a specified list of Witvliet neurons.

        Parameters
        ----------
        target_neurons
            Ordered list of Witvliet neuron names to include as rows.
            Neurons without Packer coverage receive a zero row and are
            listed in ExpressionMatrixResult.unmapped_neurons.
        gene_ids
            WormBase IDs of genes to include as columns.  Defaults to all
            genes in the dataset.
        """
        obs = self._ad.obs
        var_names = list(self._ad.var_names)

        if gene_ids is not None:
            col_idx = [var_names.index(g) for g in gene_ids if g in var_names]
            col_labels = [self._gene_id_to_name.get(var_names[i], var_names[i]) for i in col_idx]
        else:
            col_idx = list(range(len(var_names)))
            col_labels = [self._gene_id_to_name.get(g, g) for g in var_names]

        n_genes = len(col_idx)
        rows: list[np.ndarray] = []
        profiles: dict[str, NeuronProfile] = {}
        unmapped: list[str] = []

        times = obs[self._time_col].astype(float)

        for neuron in target_neurons:
            packer_subtype = self._mapping.get(neuron)

            if packer_subtype is None:
                rows.append(np.zeros(n_genes, dtype=np.float32))
                unmapped.append(neuron)
                log.debug("No Packer coverage for '%s'; using zero vector", neuron)
                continue

            subtype_mask = obs[self._ann_col] == packer_subtype
            subtype_times = times[subtype_mask]
            n_total = int(subtype_mask.sum())

            if n_total == 0:
                rows.append(np.zeros(n_genes, dtype=np.float32))
                unmapped.append(neuron)
                log.warning("Packer subtype '%s' has no cells; skipping '%s'", packer_subtype, neuron)
                continue

            threshold = self._time_threshold(subtype_times)
            late_mask = subtype_mask & (times >= threshold)
            n_late = int(late_mask.sum())
            used_fallback = False

            if n_late < MIN_CELLS_PER_TYPE:
                used_fallback = True
                late_mask = subtype_mask
                n_late = n_total
                log.debug(
                    "Fallback for '%s' (%s): only %d cells past %.0f min; using all %d",
                    neuron, packer_subtype, n_late, threshold, n_total,
                )
            else:
                log.debug(
                    "'%s' (%s): %d/%d cells past %.0f min",
                    neuron, packer_subtype, n_late, n_total, threshold,
                )

            cell_idx = np.where(late_mask.values)[0]
            expr = self._mean_expression(cell_idx)

            if gene_ids is not None:
                expr = expr[col_idx]

            rows.append(expr)
            profiles[neuron] = NeuronProfile(
                witvliet_name=neuron,
                packer_subtype=packer_subtype,
                n_cells_total=n_total,
                n_cells_used=n_late,
                time_threshold=threshold,
                used_fallback=used_fallback,
            )

        X = np.vstack(rows)

        log.info(
            "Expression matrix built: %d neurons × %d genes  "
            "(%d unmapped → zero rows)",
            len(target_neurons), n_genes, len(unmapped),
        )

        return ExpressionMatrixResult(
            X=X,
            neuron_names=target_neurons,
            gene_names=col_labels,
            profiles=profiles,
            unmapped_neurons=unmapped,
        )

    def check_markers(
        self, result: ExpressionMatrixResult
    ) -> list[MarkerCheckResult]:
        """
        Validate expression profiles against known cell-type marker genes.

        For each (neuron_type, marker_gene) pair in MARKER_GENES, compute the
        fold change of expression in the target type versus all other neurons.
        A fold change > 2 is consistent with correct cell-type resolution.
        This is a biological sanity check, not a statistical test.
        """
        gene_col_map = {name: i for i, name in enumerate(result.gene_names)}
        neuron_row_map = {name: i for i, name in enumerate(result.neuron_names)}

        checks: list[MarkerCheckResult] = []

        for neuron_type, markers in MARKER_GENES.items():
            type_rows = [
                i for name, i in neuron_row_map.items()
                if self._mapping.get(name) == neuron_type
            ]
            other_rows = [
                i for i in range(len(result.neuron_names))
                if i not in type_rows
            ]

            for gene_name in markers:
                if gene_name not in gene_col_map:
                    checks.append(
                        MarkerCheckResult(
                            neuron_type=neuron_type,
                            marker_gene=gene_name,
                            expression_in_type=0.0,
                            expression_in_others=0.0,
                            fold_change=0.0,
                            gene_found=False,
                        )
                    )
                    continue

                col = gene_col_map[gene_name]
                expr_type = float(result.X[type_rows, col].mean()) if type_rows else 0.0
                expr_others = float(result.X[other_rows, col].mean()) if other_rows else 0.0
                fc = expr_type / max(expr_others, 1e-9)

                checks.append(
                    MarkerCheckResult(
                        neuron_type=neuron_type,
                        marker_gene=gene_name,
                        expression_in_type=expr_type,
                        expression_in_others=expr_others,
                        fold_change=fc,
                        gene_found=True,
                    )
                )

                log.debug(
                    "Marker %s in %s: in-type=%.3f, others=%.3f, FC=%.2f",
                    gene_name, neuron_type, expr_type, expr_others, fc,
                )

        return checks
