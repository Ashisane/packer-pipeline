"""
preprocess.py — Standardises the expression matrix and serialises
                GNN-ready tensors to disk.

Standardisation rationale: genes with high absolute expression (e.g.
ribosomal or structural genes) would dominate the learned W matrix if
features were left on their natural scale.  Standardising each gene
across neurons ensures that a guidance gene expressed at low levels but
consistently enriched in one neuron class contributes equally to its
high-expression counterpart.  This is the correct axis of normalisation
because the GNN compares neuron pairs, not gene values across cells.

Missing neuron handling: neurons absent from Packer get a zero vector by
default.  The zero-vector assumption is conservative and may cause the GNN
to learn 'unknown → no edge' rather than 'unknown → uncertain'.  The
'learned_default' strategy addresses this by making the zero vector a
trainable parameter, but that requires modifications to the GNN code.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

TENSOR_DTYPE = torch.float32

FNAME_X         = "X_tensor.pt"
FNAME_MASK      = "coverage_mask.pt"
FNAME_NEURONS   = "neuron_index.json"
FNAME_GENES     = "gene_index.json"
FNAME_PROFILES  = "neuron_profiles.json"
FNAME_GENE_META = "gene_metadata.json"


@dataclass
class PreprocessorOutput:
    """Tensors and indices produced by GNNPreprocessor.build()."""

    X_tensor: torch.Tensor
    coverage_mask: torch.Tensor
    neuron_index: dict[str, int]
    gene_index: dict[str, int]
    n_neurons: int
    n_genes: int
    n_covered: int
    n_missing: int


class GNNPreprocessor:
    """
    Standardises the expression matrix and produces serialised GNN inputs.

    Parameters
    ----------
    X : np.ndarray
        (N_neurons, N_genes) raw expression matrix from ExpressionMatrix.build().
    neuron_names : list[str]
        Row labels, in order.
    gene_names : list[str]
        Column labels, in order.
    coverage_mask : np.ndarray or list[bool]
        Boolean array indicating which rows have real Packer data.
        Rows where this is False received a zero vector in ExpressionMatrix.
    """

    def __init__(
        self,
        X: np.ndarray,
        neuron_names: list[str],
        gene_names: list[str],
        coverage_mask: list[bool] | np.ndarray,
    ) -> None:
        if X.shape[0] != len(neuron_names):
            raise ValueError(
                f"X has {X.shape[0]} rows but neuron_names has {len(neuron_names)} entries. "
                f"They must correspond 1-to-1."
            )
        if X.shape[1] != len(gene_names):
            raise ValueError(
                f"X has {X.shape[1]} columns but gene_names has {len(gene_names)} entries."
            )

        self._X = X.astype(np.float32)
        self._neuron_names = neuron_names
        self._gene_names = gene_names
        self._coverage_mask = np.asarray(coverage_mask, dtype=bool)

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        """
        Z-score each gene column using statistics computed only from covered neurons.

        Using only real (non-zero) rows prevents missing neurons from pulling the
        mean toward zero and inflating the apparent variance of the gene.
        """
        covered_rows = self._coverage_mask
        mean = X[covered_rows].mean(axis=0)
        std  = X[covered_rows].std(axis=0)

        std[std < 1e-9] = 1.0

        X_z = (X - mean) / std
        X_z[~covered_rows] = 0.0

        return X_z.astype(np.float32)

    def build(
        self,
        missing_strategy: Literal["zero_vector", "learned_default"] = "zero_vector",
    ) -> PreprocessorOutput:
        """
        Standardise and package tensors ready for GNN training.

        Parameters
        ----------
        missing_strategy
            'zero_vector'      — missing neurons get a fixed zero vector.
                                 Simple, conservative, currently used.
            'learned_default'  — placeholder flag; the returned X_tensor
                                 still contains zeros, but training code
                                 should replace those rows with a trainable
                                 nn.Parameter.  This module cannot implement
                                 the trainable parameter itself.
        """
        if missing_strategy == "learned_default":
            log.warning(
                "missing_strategy='learned_default' is a placeholder: X_tensor "
                "will contain zeros for missing neurons.  Replace those rows with "
                "a trainable nn.Parameter in your GNN model's forward() pass. "
                "Use coverage_mask to identify which rows need replacement."
            )

        X_std = self._standardise(self._X)
        X_tensor = torch.tensor(X_std, dtype=TENSOR_DTYPE)
        coverage = torch.tensor(self._coverage_mask, dtype=torch.bool)

        neuron_index = {name: i for i, name in enumerate(self._neuron_names)}
        gene_index   = {name: i for i, name in enumerate(self._gene_names)}

        n_covered = int(self._coverage_mask.sum())
        n_missing = int((~self._coverage_mask).sum())

        log.info(
            "Preprocessor: %d neurons (%d covered, %d missing) × %d genes",
            len(self._neuron_names), n_covered, n_missing, len(self._gene_names),
        )

        return PreprocessorOutput(
            X_tensor=X_tensor,
            coverage_mask=coverage,
            neuron_index=neuron_index,
            gene_index=gene_index,
            n_neurons=len(self._neuron_names),
            n_genes=len(self._gene_names),
            n_covered=n_covered,
            n_missing=n_missing,
        )

    def save(
        self,
        output: PreprocessorOutput,
        output_dir: str | Path,
        gene_metadata: Optional[dict] = None,
        neuron_profiles: Optional[dict] = None,
    ) -> dict[str, str]:
        """
        Write all GNN-ready artefacts to disk.

        The GNN training script can load X_tensor.pt and the two JSON index
        files without re-running the pipeline.  Returns a dict of
        {artifact_name: filepath} for inclusion in the pipeline manifest.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        torch.save(output.X_tensor, out / FNAME_X)
        torch.save(output.coverage_mask, out / FNAME_MASK)

        with open(out / FNAME_NEURONS, "w") as f:
            json.dump(output.neuron_index, f, indent=2)
        with open(out / FNAME_GENES, "w") as f:
            json.dump(output.gene_index, f, indent=2)

        if gene_metadata:
            with open(out / FNAME_GENE_META, "w") as f:
                json.dump(gene_metadata, f, indent=2)
        if neuron_profiles:
            with open(out / FNAME_PROFILES, "w") as f:
                json.dump(neuron_profiles, f, indent=2)

        produced = {
            "X_tensor":      str(out / FNAME_X),
            "coverage_mask": str(out / FNAME_MASK),
            "neuron_index":  str(out / FNAME_NEURONS),
            "gene_index":    str(out / FNAME_GENES),
        }
        if gene_metadata:
            produced["gene_metadata"] = str(out / FNAME_GENE_META)
        if neuron_profiles:
            produced["neuron_profiles"] = str(out / FNAME_PROFILES)

        log.info("Saved %d artefacts to '%s'", len(produced), out)
        return produced

    def verify_against_adjacency(
        self,
        output: PreprocessorOutput,
        adjacency_neurons: list[str],
    ) -> dict[str, list[str]]:
        """
        Check that every neuron in an adjacency matrix has a row in X_tensor.

        Returns a dict with keys 'missing_from_X' (neurons in the adjacency
        but not in output.neuron_index) and 'extra_in_X' (neurons in X but
        not in the adjacency).  Both lists should ideally be empty.
        """
        adj_set = set(adjacency_neurons)
        x_set   = set(output.neuron_index.keys())

        missing = sorted(adj_set - x_set)
        extra   = sorted(x_set - adj_set)

        if missing:
            log.warning(
                "%d adjacency neurons have no row in X_tensor: %s",
                len(missing), missing[:10],
            )
        if extra:
            log.info(
                "%d neurons in X_tensor are not in the adjacency (expected"
                " for complete-neuron atlases): %s",
                len(extra), extra[:10],
            )

        return {"missing_from_X": missing, "extra_in_X": extra}
