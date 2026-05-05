# SA-RAG: Structure-Aware Retrieval-Augmented Generation for Text-to-SQL

[![Paper](https://img.shields.io/badge/Paper-Knowledge--Based%20Systems-blue)](https://doi.org/placeholder)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

> **SA-RAG** treats database schemas as heterogeneous graphs and uses specialized graph neural networks to filter out irrelevant schema elements before an LLM generates SQL — achieving **82.0% execution accuracy** on Spider with an open-source model at **10× lower cost** than GPT-4 methods.

<p align="center">
  <img src="assets/sa_rag_teaser_enhanced.png" width="90%" alt="SA-RAG Overview"/>
</p>

---

## Highlights

- **82.0% execution accuracy** on Spider using Llama 3.3-70B (only 3.3 points below GPT-4-based SOTA)
- **100% table recall** and **96.1% column recall** through structure-aware schema filtering
- **51.7% on BIRD Mini-Dev** with GPT-4o, exceeding the GPT-4 zero-shot baseline
- **Zero-shot cross-benchmark transfer** — the GNN trained on Spider transfers to BIRD without retraining
- **10× cheaper** per query compared to GPT-4-based approaches
- **~8.7M parameter** GNN that trains in ~6 hours on a single A100

## Architecture

<p align="center">
  <img src="assets/saragOriginal.png" width="90%" alt="SA-RAG Architecture"/>
</p>

SA-RAG operates in two stages:

**Stage 1 — Structure-Aware Schema Retrieval:** The database schema is modeled as a heterogeneous graph with 2 node types (tables, columns) and 8 edge types (ownership, foreign keys, semantic similarity, etc.). A Heterogeneous Graph Transformer (HGT) with Structure-Aware Positional Encoding (SAPE) scores every schema element for relevance to the input question. Bidirectional cross-attention fuses natural language tokens with schema node representations.

**Stage 2 — Confidence-Adaptive SQL Generation:** Based on the retrieval confidence score ρ, the system routes to one of three generation strategies:
- **High confidence (ρ > 0.7):** Direct single-pass generation
- **Medium confidence (0.4 ≤ ρ ≤ 0.7):** Sample K=5 candidates + majority vote
- **Low confidence (ρ < 0.4):** Expand schema context + chain-of-thought + majority vote

## Key Innovations

| Component | Description |
|-----------|-------------|
| **SAPE** | Structure-Aware Positional Encoding combining random walk statistics, degree encoding, and hierarchy distance — designed specifically for relational database topology |
| **Heterogeneous Schema Graph** | 8 typed edge relations capturing ownership, foreign keys, same-table co-occurrence, primary keys, and learned semantic similarity |
| **Bidirectional Fusion** | Gated cross-attention bridging natural language ("youngest student") and database terminology (`birth_date`) |
| **Score Propagation** | Column→table boosting and FK propagation ensuring structurally consistent retrieval |
| **CAP** | Confidence-Adaptive Prompting that allocates more compute only when the retrieval model is uncertain |

## Results

### Spider Dev Set

| Method | Model | EX (%) |
|--------|-------|--------|
| DIN-SQL+GPT-4 | GPT-4 | 85.3 |
| DAIL-SQL+GPT-4 | GPT-4 | 84.4 |
| PURPLE+GPT-4 | GPT-4 | 83.1 |
| **SA-RAG (Ours)** | **Llama 3.3-70B** | **82.0** |
| MAC-SQL | GPT-3.5 | 81.8 |
| GPT-4 Zero-shot | GPT-4 | 70.0 |
| Embedding-Only | Llama 3.3-70B | 68.0 |

### BIRD Mini-Dev

| Method | Model | EX (%) |
|--------|-------|--------|
| GPT-4 Zero-shot | GPT-4 | 49.2 |
| **SA-RAG (Full)** | **GPT-4o** | **51.7** |
| SA-RAG w/o FK injection | GPT-4o | 34.1 |

### Retrieval Quality

| Metric | Spider | BIRD |
|--------|--------|------|
| Table Recall@5 | 100.0% | ~100% |
| Column Recall@15 | 96.1% | 93.8% |

## Installation

### Prerequisites

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA 11.8+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/alphaomar/SA-RAG.git
cd SA-RAG

# Create conda environment
conda create -n sarag python=3.10 -y
conda activate sarag

# Install dependencies
pip install -r requirements.txt

# Download datasets
python scripts/download_data.py
```

### Dependencies

```
torch>=2.0
torch-geometric>=2.4
transformers>=4.36
sentence-transformers>=2.2
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
networkx>=3.1
groq>=0.4              # For Llama 3.3-70B inference
openai>=1.12            # For GPT-4o inference (BIRD)
tqdm>=4.65
```

## Quick Start

### 1. Train the GNN

```bash
python train.py \
    --dataset spider \
    --hidden_dim 256 \
    --num_layers 8 \
    --num_heads 8 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42
```

### 2. Run Schema Retrieval

```bash
python retrieve.py \
    --checkpoint checkpoints/best_model.pt \
    --dataset spider \
    --top_k_tables 5 \
    --top_k_columns 12
```

### 3. Generate SQL

```bash
# Spider (Llama 3.3-70B via Groq)
python generate.py \
    --dataset spider \
    --retrieval_results results/spider_retrieval.json \
    --llm llama-3.3-70b \
    --temperature 0.1 \
    --use_cap

# BIRD (GPT-4o)
python generate.py \
    --dataset bird \
    --retrieval_results results/bird_retrieval.json \
    --llm gpt-4o \
    --temperature 0.1 \
    --use_cap \
    --inject_fk_metadata
```

### 4. Evaluate

```bash
# Execution accuracy
python evaluate.py \
    --predictions results/spider_predictions.json \
    --dataset spider \
    --metric ex

# Retrieval metrics
python evaluate.py \
    --retrieval_results results/spider_retrieval.json \
    --dataset spider \
    --metric retrieval
```

## Project Structure

```
SA-RAG/
├──             # Run all ablation experiments
├── train.py                    # GNN training script
├── retrieve.py                 # Schema retrieval inference
├── generate.py                 # SQL generation with LLM
├── evaluate.py                 # Evaluation script
├── requirements.txt
├── LICENSE
└── README.md
```

## Configuration

### GNN Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 256 | Hidden dimension of GNN layers |
| `num_layers` | 8 | Number of HGT layers |
| `num_heads` | 8 | Number of attention heads |
| `dropout` | 0.1 | Dropout rate |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence encoder for node features |
| `semantic_threshold` | 0.7 | Cosine similarity threshold for semantic edges |

### CAP Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rho_high` | 0.7 | Confidence above which direct generation is used |
| `rho_low` | 0.4 | Confidence below which CoT + expanded context is used |

### API Keys

```bash
# For Spider experiments (Llama 3.3-70B via Groq)
export GROQ_API_KEY="your-groq-api-key"

# For BIRD experiments (GPT-4o)
export OPENAI_API_KEY="your-openai-api-key"
```

## Reproducibility

All experiments use seed 42. Training takes approximately 6 hours on a single NVIDIA A100. Inference costs:

| Component | Time per Query | Memory |
|-----------|---------------|--------|
| Graph Construction | 12 ± 2 ms | 45 MB |
| Embedding | 8 ± 1 ms | 120 MB |
| GNN Forward Pass | 48 ± 5 ms | 850 MB |
| Score Computation | 5 ± 1 ms | 20 MB |
| LLM Generation | 177 ± 10 ms | 1,500 MB |
| **Total** | **245 ± 12 ms** | **2,535 MB** |

Estimated cost per query: **$0.0006** (Llama 3.3 via Groq) vs. $0.006 (GPT-4).

## Running Ablations

```bash
# Full ablation suite
bash scripts/ablation.sh

# Individual ablations
python train.py --dataset spider --no_sape           # w/o SAPE
python train.py --dataset spider --no_fk_edges       # w/o FK edges
python train.py --dataset spider --homogeneous        # Homogeneous GNN
python generate.py --dataset spider --no_cap          # w/o CAP
python generate.py --dataset spider --no_cot          # w/o Chain-of-Thought
```

## Scalability

SA-RAG scales to schemas far larger than Spider's average of 7.5 tables:

| # Tables | Table Recall@5 | Inference Time |
|----------|---------------|----------------|
| 20 | 99.2% | 72 ms |
| 50 | 98.1% | 105 ms |
| 100 | 96.4% | 168 ms |
| 150 | 95.1% | 245 ms |
| 200 | 91.8% | 330 ms |

## Citation

If you use SA-RAG in your research, please cite:

```bibtex
@article{leigh2025sarag,
  title={SA-RAG: Structure-Aware Retrieval-Augmented Generation for Text-to-SQL via Heterogeneous Graph Neural Networks},
  author={Leigh, Alpha Omar and Sun, Chengjie and Fofanah, Abdul Joseph},
  journal={Knowledge-Based Systems},
  year={2025},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the creators of the [Spider](https://yale-lily.github.io/spider) and [BIRD](https://bird-bench.github.io/) benchmarks. This research was supported by Harbin Institute of Technology.

## Contact

- **Alpha Omar Leigh** — [25bf03004@stu.hit.edu.cn](mailto:25bf03004@stu.hit.edu.cn)
- **GitHub Issues** — For bug reports and feature requests, please [open an issue](https://github.com/alphaomar/SA-RAG/issues)
