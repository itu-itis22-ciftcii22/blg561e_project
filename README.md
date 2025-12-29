# Token Malice Prediction

GNN-based malicious token detection using transaction graph analysis.

## Project Structure

```
blg561e_project/
├── token_malice_prediction/     # Main package
│   ├── __init__.py
│   ├── data/                    # Data processing modules
│   │   ├── __init__.py
│   │   ├── preprocessing.py     # CSV loading, filtering, labeling
│   │   ├── graph_builder.py     # Transaction multigraph construction
│   │   └── dataset.py           # PyTorch Geometric datasets
│   ├── models/                  # Model architectures
│   │   ├── __init__.py
│   │   └── gnn.py               # GAT-based graph classifier
│   ├── training/                # Training utilities
│   │   ├── __init__.py
│   │   └── trainer.py           # Training loop and evaluation
│   ├── evaluation/              # Evaluation tools
│   │   ├── __init__.py
│   │   └── metrics.py           # Classification metrics
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       ├── logger.py            # Logging utilities
│       └── helpers.py           # Helper functions
├── solana1/                     # Token transaction CSV files
├── results.ipynb                # Results presentation notebook
├── task.py                      # Main training script
├── config.yaml                  # Configuration file
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Data Format

The system expects token transaction CSV files with the following columns:
- `Signature`: Transaction signature (used for deduplication)
- `Block Time`: Unix timestamp
- `From`: Sender wallet address
- `To`: Receiver wallet address
- `Amount`: Token amount transferred
- `Value`: USD value of transaction

## Pipeline

1. **Preprocessing**: Load CSVs, filter tokens with ≥3 months history, label based on value decline
2. **Graph Construction**: Build transaction multigraphs with node/edge features
3. **Training**: Train GAT-based classifier with early stopping
4. **Evaluation**: Compute classification metrics on test set

## Usage

### Training

```bash
python task.py --config config.yaml
python task.py --data-dir ./solana1 --output-dir ./outputs
```

### Using the Package

```python
from token_malice_prediction.data import (
    TokenPreprocessor,
    build_graphs_from_processed_data,
    TokenGraphDatasetList,
    create_data_loaders
)
from token_malice_prediction.models import create_model
from token_malice_prediction.training import Trainer

# Preprocess token CSVs
preprocessor = TokenPreprocessor(
    data_dir='./solana1',
    malice_threshold=0.9,  # 90% value decline = malicious
    min_transactions=10
)
processed_data = preprocessor.process_directory()

# Build transaction graphs
graphs_with_names = build_graphs_from_processed_data(processed_data)
graphs = [g for g, _ in graphs_with_names]

# Create dataset and loaders
dataset = TokenGraphDatasetList(graphs)
train_loader, val_loader, test_loader = create_data_loaders(dataset)

# Build model (GATv2Conv-based)
model = create_model(
    node_dim=11,
    edge_dim=10,
    hidden_dim=128,
    num_layers=3,
    num_heads=4
)

# Train
trainer = Trainer(model, optimizer, device='cuda')
trainer.train(train_loader, val_loader, num_epochs=100)
```

## Model Architecture

### TokenGATClassifier
Graph Attention Network (GATv2) for graph-level classification:
- **Input Projection**: Projects node features to hidden dimension
- **Edge Features**: Incorporates edge attributes (amount, value, time) via attention
- **GATv2Conv Layers**: Multi-head attention with residual connections
- **Pooling**: Mean/max/concat pooling for graph-level representation
- **Classifier**: MLP head for binary classification

### Graph Structure
- **Nodes**: Wallet addresses with degree/volume statistics
- **Edges**: Undirected with types (SENT, RECEIVED, MINT)
- **Edge Features**: Amount, value, amount×value, relative time periods

## Labeling Strategy

Tokens are labeled as **malicious** if:
```
(peak_value - value_at_3_months) / peak_value >= threshold
```
Default threshold is 0.9 (90% decline from peak to 3-month mark).

Training uses only the first 1.5 months of transaction data.

## Configuration

Edit `config.yaml` to customize:

```yaml
data:
  data_dir: "solana1"
  malice_threshold: 0.9
  min_transactions: 10

model:
  hidden_dim: 128
  num_layers: 3
  num_heads: 4
  dropout: 0.3
  pooling: "concat"

training:
  learning_rate: 0.001
  num_epochs: 100
  early_stopping_patience: 10
```

## Evaluation Metrics

- **Accuracy, Precision, Recall, F1**
- **AUC-ROC, AUC-PRC**
- **Confusion Matrix**

## License

This project is for academic purposes (BLG561E course project).
