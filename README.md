# packer-wiring-atlas

A pipeline for extracting embryonic transcriptomic
identity from the **Packer 2019** *C. elegans* scRNA-seq atlas and converting
it into node-feature tensors for a developmental Graph Neural Network that
predicts synaptic connectivity from Witvliet et al. 2021.

The scientific claim under investigation: embryonic gene expression at
neuronal terminal differentiation predicts which neurons form synapses at
hatching (Witvliet D1).  The GNN learns a compatibility matrix **W** that
encodes which transcriptomic profiles tend to co-occur in synaptically
connected neuron pairs, with axon-guidance and synaptic-specificity genes
as the mechanistically interpretable subset of the feature space.

---

## Data

### Packer 2019 (required)

| Field | Value |
|-------|-------|
| Paper | Packer et al. 2019, *Science* 365, 168–173 |
| DOI   | 10.1126/science.aax1971 |
| Download | https://data.caltech.edu/records/b1kj4-nh475 |
| File | `GSE126954_gene_annotation_20191126.h5ad` (≈ 2 GB) |

Place the file at `data/packer2019.h5ad` (or pass `--packer_path` to override).

### Witvliet 2021 (required)

| Field | Value |
|-------|-------|
| Paper | Witvliet et al. 2021, *Nature* 596, 257–261 |
| DOI   | 10.1038/s41586-021-03778-8 |
| Source | Supplementary Data, synaptic connectivity tables |
| Files | `Dataset1_synapses.json` … `Dataset8_synapses.json` |

Place all eight files in `data/witvliet/`.

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or later required.

---

## Usage

```bash
python scripts/build_atlas.py \
    --packer_path  data/packer2019.h5ad \
    --witvliet_dir data/witvliet/ \
    --output_dir   data/processed/ \
    --n_variance_genes 200
```

All options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--packer_path` | required | Path to Packer h5ad |
| `--witvliet_dir` | required | Directory with Witvliet JSON files |
| `--output_dir` | `data/processed` | Where to write tensors and manifest |
| `--n_variance_genes` | `200` | High-variance genes from Strategy 2 |
| `--log_level` | `INFO` | Logging verbosity |

---

## Output

All files are written to `--output_dir`:

| File | Shape / Type | Description |
|------|-------------|-------------|
| `X_tensor.pt` | `(N, D)` float32 | Standardised expression matrix for GNN |
| `coverage_mask.pt` | `(N,)` bool | True where real Packer data exists |
| `neuron_index.json` | dict | `{neuron_name: row_index}` |
| `gene_index.json` | dict | `{gene_name: col_index}` |
| `gene_metadata.json` | dict | Gene origin (curated/variance/both) |
| `neuron_profiles.json` | dict | Per-neuron profiling metadata |
| `manifest.json` | JSON | Coverage statistics and provenance record |

### Loading in a GNN training script

```python
import torch, json

X = torch.load('data/processed/X_tensor.pt')          # (234, 234)
mask = torch.load('data/processed/coverage_mask.pt')  # (234,) bool
with open('data/processed/neuron_index.json') as f:
    neuron_index = json.load(f)
```

---

## Exploratory Analysis

Open the notebook for a guided tour of the data:

```bash
cd notebooks/
jupyter notebook exploration.ipynb
```

Sections: dataset overview · neuron coverage · embryo-time distributions ·
expression heatmap · marker gene validation · GNN coverage summary.

---

## Limitations

### 1. Bilateral symmetry assumption
Packer annotates cells by neuron class ('AWC') rather than individual neuron
('AWCL', 'AWCR').  Both left and right neurons of the same class receive
identical expression profiles.  The GNN cannot use expression to distinguish
bilaterally symmetric pairs.  This affects approximately 85 neuron pairs.

### 2. Motor neuron coverage gap
The Packer 2019 atlas does not include resolved subtypes for the serial motor
neurons (VB1, VB2) that appear in Witvliet D1.  These neurons receive zero-
vector features.  The GNN's performance on motor neuron connectivity is
therefore driven entirely by structural features, not expression.

### 3. Terminal differentiation approximation
"Terminal differentiation" is operationalised as the 75th-percentile of
embryo_time within each neuron class.  This is a reasonable approximation
but does not reflect the true neuronal identity transition time, which would
require marker-gene inflection-point analysis per class.

### 4. Single-cell RNA-seq technical variation
Mean expression across cells at terminal differentiation suppresses
within-class technical noise but also removes potentially informative
intra-class variability.  The current pipeline treats each neuron class as
a single homogeneous entity.

---

## Citation

If you use this pipeline, please cite:

- Packer JS et al. (2019) A lineage-resolved molecular atlas of C. elegans
  embryogenesis at single-cell resolution. *Science* 365, 168–173.
- Witvliet D et al. (2021) Connectomes across development reveal principles
  of brain maturation. *Nature* 596, 257–261.

Pipeline developed as part of the DevoWorm project (Brad Alicea, PI).
