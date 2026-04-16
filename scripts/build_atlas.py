#!/usr/bin/env python
"""
build_atlas.py — End-to-end pipeline: Packer 2019 → GNN-ready tensors.

Runs all five pipeline stages in order and writes a manifest.json
recording what was produced, the coverage numbers, and provenance.

Usage
-----
    python scripts/build_atlas.py \\
        --packer_path  data/packer2019.h5ad \\
        --witvliet_dir data/witvliet/ \\
        --output_dir   data/processed/ \\
        --n_variance_genes 200

The Witvliet directory should contain Dataset1_synapses.json through
Dataset8_synapses.json from the Witvliet 2021 Nature dataset.
"""

import argparse
import datetime
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# ── Ensure the parent directory is importable regardless of cwd ───────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packer.loader import PackerDataset
from packer.annotation import NeuronMapper
from packer.expression import ExpressionMatrix
from packer.genes import GeneSelector
from packer.preprocess import GNNPreprocessor

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_atlas")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Packer 2019 → GNN-ready feature tensors for C. elegans connectome."
    )
    p.add_argument(
        "--packer_path",
        required=True,
        type=Path,
        help="Path to the Packer 2019 h5ad file (~2 GB).",
    )
    p.add_argument(
        "--witvliet_dir",
        required=True,
        type=Path,
        help="Directory containing Dataset1_synapses.json … Dataset8_synapses.json.",
    )
    p.add_argument(
        "--output_dir",
        default=Path("data/processed"),
        type=Path,
        help="Directory where GNN-ready tensors and manifest are written.",
    )
    p.add_argument(
        "--n_variance_genes",
        type=int,
        default=200,
        help="Number of high-variance genes to include from Strategy 2 selection.",
    )
    p.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return p.parse_args()

# ── Witvliet helpers ──────────────────────────────────────────────────────────

def load_witvliet_neurons(witvliet_dir: Path) -> list[str]:
    """Collect the union of all neuron names across all 8 Witvliet stages."""
    all_neurons: set[str] = set()
    for stage in range(1, 9):
        fpath = witvliet_dir / f"Dataset{stage}_synapses.json"
        if not fpath.exists():
            log.warning("Missing Witvliet file: %s — skipping stage %d", fpath, stage)
            continue
        with open(fpath) as f:
            records = json.load(f)
        for r in records:
            all_neurons.add(r["pre"])
            post = r["post"] if isinstance(r["post"], list) else [r["post"]]
            all_neurons.update(post)
    neurons = sorted(all_neurons)
    log.info("Witvliet: %d unique neurons across available stages", len(neurons))
    return neurons


def classify_circuit(name: str) -> str:
    """Classify a neuron into sensory / interneuron / motor / other."""
    from packer.annotation import (
        SENSORY_PREFIXES as _S,
        MOTOR_PREFIXES as _M,
        INTERNEURON_PREFIXES as _I,
    )
    for p in _S:
        if name.startswith(p): return "sensory"
    for p in _M:
        if name.startswith(p): return "motor"
    for p in _I:
        if name.startswith(p): return "interneuron"
    return "other"

# Import prefix lists from annotation module
# (they are not part of the public __init__ API, so we import directly)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    print("=" * 60)
    print("  Packer → C. elegans Wiring Atlas Pipeline")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')}")
    print("=" * 60)

    # ── Stage 1: Load and validate ────────────────────────────────────────────
    print("\n[1/5] Loading Packer 2019 dataset …")
    dataset = PackerDataset(args.packer_path)
    summary = dataset.validate()
    print(
        f"    {summary.n_cells:,} cells  ×  {summary.n_genes:,} genes  |  "
        f"{summary.n_neuron_cells:,} neuron cells  |  "
        f"time {summary.embryo_time_range[0]:.0f}–{summary.embryo_time_range[1]:.0f} min"
    )
    if summary.missing_required_cols:
        log.warning("Missing expected obs columns: %s", summary.missing_required_cols)

    # ── Stage 2: Build neuron mapping ─────────────────────────────────────────
    print("\n[2/5] Building neuron mapping (Witvliet → Packer subtypes) …")
    witvliet_neurons = load_witvliet_neurons(args.witvliet_dir)

    packer_subtypes = (
        dataset.adata.obs[dataset.annotation_column]
        .dropna()
        .unique()
        .tolist()
    )
    mapper = NeuronMapper(packer_subtypes=packer_subtypes)
    mapping = mapper.build_mapping(witvliet_neurons)
    report = mapper.report(witvliet_neurons)

    print(
        f"    {report.n_neurons_mapped}/{report.n_neurons_total} neurons mapped "
        f"({100 * report.coverage_fraction:.1f}%)  |  "
        f"{report.n_neurons_unmapped} unmapped"
    )
    print(f"    Rule breakdown: { {r: len(v) for r, v in report.by_rule.items() if v} }")

    # ── Stage 3: Construct expression matrix ──────────────────────────────────
    print("\n[3/5] Constructing expression matrix (per-neuron terminal profiles) …")
    neuron_view = dataset.neuron_view()
    gene_id_map = dataset.gene_name_map()

    expr_builder = ExpressionMatrix(
        dataset=neuron_view,
        mapping=mapping,
        ann_col=dataset.annotation_column,
        time_col=dataset.time_column,
        gene_id_to_name=gene_id_map,
    )

    # First build with all genes so we can compute variance for gene selection.
    full_result = expr_builder.build(target_neurons=witvliet_neurons)
    print(
        f"    Full matrix: {full_result.X.shape[0]} neurons × {full_result.X.shape[1]} genes  |  "
        f"{len(full_result.unmapped_neurons)} zero rows"
    )

    # ── Stage 4: Select gene set ──────────────────────────────────────────────
    print(f"\n[4/5] Selecting gene features (curated + top-{args.n_variance_genes} variance) …")
    selector = GeneSelector(gene_id_to_name=gene_id_map)

    # Compute subtype-level means for variance calculation using covered rows only.
    covered_mask = np.array(
        [n not in full_result.unmapped_neurons for n in full_result.neuron_names]
    )
    var_names_all = list(dataset.adata.var_names)

    # Load the neuron-view expression matrix once for variance computation.
    # This requires the full gene dimension, so we use the raw adata.
    print("    Computing gene variances across neuron subtypes …")
    subtype_labels = list({mapping[n] for n in witvliet_neurons if n in mapping})
    obs = neuron_view.obs
    ann_col = dataset.annotation_column
    time_col = dataset.time_column
    from packer.expression import TERMINAL_DIFF_PERCENTILE, MIN_CELLS_PER_TYPE
    import scipy.sparse as sp

    subtype_means_list = []
    for st in subtype_labels:
        mask = obs[ann_col] == st
        times = obs[time_col].astype(float)[mask]
        n_st = int(mask.sum())
        if n_st == 0:
            continue
        thr = float(np.percentile(times, TERMINAL_DIFF_PERCENTILE))
        late = mask & (obs[time_col].astype(float) >= thr)
        n_late = int(late.sum())
        use_mask = late if n_late >= MIN_CELLS_PER_TYPE else mask
        idx = np.where(use_mask.values)[0]
        if len(idx) == 0:
            continue
        blk = neuron_view.X[idx, :]
        mean_expr = np.asarray(blk.mean(0)).ravel() if sp.issparse(blk) else blk.mean(0).ravel()
        subtype_means_list.append(mean_expr.astype(np.float32))

    subtype_means = np.vstack(subtype_means_list) if subtype_means_list else np.zeros((1, len(var_names_all)))

    gene_set = selector.select(
        subtype_means=subtype_means,
        all_gene_ids=var_names_all,
        n_variance_genes=args.n_variance_genes,
    )
    print(
        f"    {len(gene_set.curated_found)} curated genes found  |  "
        f"{len(gene_set.curated_missing)} missing from Packer  |  "
        f"{gene_set.n_variance_selected} variance genes  |  "
        f"{len(gene_set.gene_ids)} total"
    )
    if gene_set.curated_missing:
        print(f"    Missing curated genes: {[g.name for g in gene_set.curated_missing]}")

    # ── Rebuild expression matrix with selected genes ─────────────────────────
    print("    Rebuilding expression matrix with selected gene set …")
    final_result = expr_builder.build(
        target_neurons=witvliet_neurons,
        gene_ids=gene_set.gene_ids,
    )
    print(f"    Final matrix: {final_result.X.shape[0]} × {final_result.X.shape[1]}")

    # Marker gene validation
    markers = expr_builder.check_markers(final_result)
    for m in markers:
        status = "✓" if m.fold_change >= 2 else ("✗" if m.gene_found else "missing")
        print(
            f"    Marker {m.marker_gene:10s} in {m.neuron_type}: "
            f"FC={m.fold_change:.2f}  [{status}]"
        )

    # ── Stage 5: Preprocess for GNN ───────────────────────────────────────────
    print("\n[5/5] Standardising and serialising GNN tensors …")
    coverage_mask = [n not in final_result.unmapped_neurons for n in final_result.neuron_names]

    preprocessor = GNNPreprocessor(
        X=final_result.X,
        neuron_names=final_result.neuron_names,
        gene_names=final_result.gene_names,
        coverage_mask=coverage_mask,
    )
    output = preprocessor.build(missing_strategy="zero_vector")

    # Verify consistency with Witvliet adjacency
    check = preprocessor.verify_against_adjacency(output, witvliet_neurons)
    if check["missing_from_X"]:
        print(
            f"    WARNING: {len(check['missing_from_X'])} adjacency neurons "
            f"missing from X_tensor: {check['missing_from_X'][:5]}"
        )

    # Neuron profiles for manifest
    neuron_profiles_json = {
        name: {
            "packer_subtype": p.packer_subtype,
            "n_cells_total":  p.n_cells_total,
            "n_cells_used":   p.n_cells_used,
            "time_threshold": p.time_threshold,
            "used_fallback":  p.used_fallback,
        }
        for name, p in final_result.profiles.items()
    }

    # Gene metadata for manifest
    gene_metadata_json = {
        gid: {
            "name":   gname,
            "origin": gene_set.origin.get(gid, "variance").value
                      if hasattr(gene_set.origin.get(gid, "variance"), "value")
                      else str(gene_set.origin.get(gid, "variance")),
        }
        for gid, gname in zip(gene_set.gene_ids, gene_set.gene_names)
    }

    produced = preprocessor.save(
        output=output,
        output_dir=args.output_dir,
        gene_metadata=gene_metadata_json,
        neuron_profiles=neuron_profiles_json,
    )
    print(f"    Tensors: X_tensor {tuple(output.X_tensor.shape)}, "
          f"coverage_mask {tuple(output.coverage_mask.shape)}")

    # ── Manifest ──────────────────────────────────────────────────────────────
    print("\n  Writing manifest …")
    circuit_coverage: dict[str, dict[str, int]] = {
        "sensory":     {"covered": 0, "total": 0},
        "interneuron": {"covered": 0, "total": 0},
        "motor":       {"covered": 0, "total": 0},
        "other":       {"covered": 0, "total": 0},
    }
    for neuron in witvliet_neurons:
        cls = classify_circuit(neuron)
        circuit_coverage[cls]["total"] += 1
        if neuron not in final_result.unmapped_neurons:
            circuit_coverage[cls]["covered"] += 1

    manifest = {
        "generated_at":           datetime.datetime.now().isoformat(),
        "packer_path":            str(args.packer_path),
        "witvliet_dir":           str(args.witvliet_dir),
        "n_neurons_total":        report.n_neurons_total,
        "n_neurons_covered":      output.n_covered,
        "n_neurons_missing":      output.n_missing,
        "coverage_fraction":      round(report.coverage_fraction, 4),
        "n_genes_curated_found":  len(gene_set.curated_found),
        "n_genes_curated_missing":len(gene_set.curated_missing),
        "n_genes_variance":       gene_set.n_variance_selected,
        "n_genes_total":          output.n_genes,
        "coverage_by_circuit":    circuit_coverage,
        "X_tensor_shape":         list(output.X_tensor.shape),
        "output_files":           produced,
        "unmapped_neurons":       final_result.unmapped_neurons,
        "curated_missing_genes":  [g.name for g in gene_set.curated_missing],
        "mapping_by_rule": {
            r: len(v) for r, v in report.by_rule.items() if v
        },
    }

    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"    Manifest written to {manifest_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Neurons: {output.n_covered}/{report.n_neurons_total} covered")
    print(f"  Genes:   {output.n_genes} ({len(gene_set.curated_found)} curated + "
          f"{gene_set.n_variance_selected} variance)")
    print(f"  X_tensor: {tuple(output.X_tensor.shape)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
