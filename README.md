# DoTPFN: Document-Oriented TabPFN for Clinical Predictive Models

DoTPFN is a modular, high-performance, prior-data fitted network designed to solve clinical prediction tasks by cleanly fusing low-dimensional clinical tabular records and high-dimensional document/image representations (such as sleep hypnograms and clinical charts extracted by state-of-the-art vision-language indexers like ColQwen2/ColPali).

---

## Key Features

* **Episodic In-Context Causal Decoder**: Incorporates strict causal masking rules for Bayesian-like in-context episodic inference.
* **Modality Alignment Regularization**: Supports loading pre-trained document embeddings, creating frozen "alignment anchors," and penalizing attention weight drift during joint training to preserve spatial mappings.
* **Colorized and Robust Logger**: Complete terminal outputs.
* **Config-Driven Operations**: Argparse command overrides coupled with structured YAML config systems to customize data paths, models, training epochs, and hyperparameters.

---

## Codebase Architecture

The repository is structured identically to modern state-of-the-art AI research systems:

```text
DoTPFN/
│
├── config/                  # YAML configurations
│   ├── multimodal.yaml      # DoTPFN multimodal CV config
│   ├── image_only.yaml      # Baseline Attention Classifier config
│   ├── image_pfn.yaml       # Baseline Image-Only PFN config
│   └── explain.yaml          # Saliency explaining config
│
├── src/
│   └── dotpfn/              # Core Library
│       ├── data/            # Datasets and Patient Group CV Splitting
│       ├── models/          # Cross Attention, Seq Decoders, Multimodal Fusers
│       ├── utils/           # Argparse YAML Loaders, Logger, zero-safe Metrics
│       └── scripts/         # Engine orchestrators (Train, Extract, Merge, Explain)
│
├── train.py                 # Unified cross-validation training entrypoint
├── explain.py               # Unified explainability report entrypoint
├── extract.py               # Preprocessing (ColQwen2 extract & CSV Merging) CLI
└── README.md                # This documentation

```

---

## Getting Started

### 1. Document Indexing & Feature Merging

First, extract document embedding tensors (using ColQwen2 vision model) from a directory of clinical charts, and merge the index list with clinical patient metadata:

```bash
# Extract raw chart hypnograms to tensor embeddings
python extract.py extract \
    --input_folder /path/to/raw/pngs \
    --output_dir ./outputs \
    --device cuda

# Merge extracted index list with clinical metadata
python extract.py merge \
    --images_path ./outputs/embedding_index.csv \
    --metadata_path ./patient_metadata.csv \
    --output_path ./final_combined_dataset.csv

```

### 2. Launch Cross-Validation Training

Run stratified K-fold cross-validation or episodic meta-learning for different model architectures:

```bash
# Run Multimodal DoTPFN Joint Training
python train.py --config config/multimodal.yaml --device cuda --epochs 120

# Run Image-Only Baseline Attention Classifier
python train.py --config config/image_only.yaml --lr 0.0002

```

### 3. Generate Patient Saliency Explanations

Calculate Macro Driver modulations via KernelSHAP and Micro Spatial Saliency via cross-attention weights overlaid directly on raw clinical charts:

```bash
python explain.py --config config/explain.yaml --patient_id 10008

```

---

## Theory & Mechanics

### Causal Bayesian In-Context Masking

The decoder constructs a unified sequence containing support (train) features and query (test) features:

$$\text{Combined Sequence} = [X_{\text{supp}} + E(y_{\text{supp}}), X_{\text{query}}]$$

To maintain mathematically valid in-context Bayesian inference, we apply a strict block-causal attention mask:

1. Support instances can attend to all other Support instances.
2. Query instances can attend to Support instances.
3. Query instances **cannot** attend to other Query instances (ensuring conditional prediction independence).

$$\text{Attention Mask}_{i, j} = \begin{cases} 
0 & j < N_{\text{supp}} \\
0 & i = j \\
-\infty & \text{otherwise}
\end{cases}$$

### Modality Drift Regularization

When training multimodal models end-to-end, high-capacity projectors can suffer from representation drift. We apply a structural constraint regularizing the active cross-attention weights ($A_{\text{after}}$) against a frozen pre-trained anchor pooler ($A_{\text{before}}$):

$$\mathcal{L}_{\text{reg}} = \frac{1}{B \cdot N} \sum_{b, n} \left\| A_{\text{after}}^{(b, n)} - A_{\text{before}}^{(b, n)} \right\|_2^2$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}}$$
