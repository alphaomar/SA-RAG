"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           ENHANCED TRAINING PIPELINE FOR SCHEMA LINKING GNN                  ║
║                                                                              ║
║  For use in Kaggle/Colab notebooks                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Paths
    train_data_path: str = None
    tables_path: str = None
    checkpoint_dir: str = '/kaggle/working/'
    
    # Model
    hidden_dim: int = 256
    num_hgt_layers: int = 8
    num_cross_attn_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # Training
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 15
    warmup_epochs: int = 5
    
    # Data
    val_split: float = 0.1
    embedding_model: str = 'all-MiniLM-L6-v2'
    
    # Advanced
    use_amp: bool = True
    use_curriculum: bool = True
    
    def validate(self):
        assert self.train_data_path is not None
        assert self.tables_path is not None
        return True


class MetricsTracker:
    """Track training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.table_recalls = []
        self.column_recalls = []
        self.perfect_both = []
    
    def update(self, table_scores, column_scores, table_labels, column_labels):
        gt_tables = (table_labels > 0.5).nonzero(as_tuple=True)[0]
        gt_columns = (column_labels > 0.5).nonzero(as_tuple=True)[0]
        
        if len(gt_tables) == 0:
            return
        
        k_t = min(5, len(table_scores))
        k_c = min(15, len(column_scores))
        
        _, pred_t = torch.topk(table_scores, k_t)
        _, pred_c = torch.topk(column_scores, k_c)
        
        pred_t = set(pred_t.cpu().numpy())
        pred_c = set(pred_c.cpu().numpy())
        gt_t = set(gt_tables.cpu().numpy())
        gt_c = set(gt_columns.cpu().numpy())
        
        self.table_recalls.append(len(pred_t & gt_t) / len(gt_t))
        
        if len(gt_c) > 0:
            self.column_recalls.append(len(pred_c & gt_c) / len(gt_c))
            perfect = (gt_t <= pred_t) and (gt_c <= pred_c)
            self.perfect_both.append(1.0 if perfect else 0.0)
    
    def compute(self):
        return {
            'table_recall': np.mean(self.table_recalls) if self.table_recalls else 0,
            'column_recall': np.mean(self.column_recalls) if self.column_recalls else 0,
            'perfect_both': np.mean(self.perfect_both) if self.perfect_both else 0
        }


def train_model(config: TrainingConfig) -> Dict:
    """Train the enhanced schema linking model."""
    config.validate()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("ENHANCED SCHEMA LINKING TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    
    # Load data
    print("\n📦 Loading data...")
    with open(config.train_data_path, 'r') as f:
        train_data = json.load(f)
    with open(config.tables_path, 'r') as f:
        tables_data = json.load(f)
    
    # Build schemas
    db_schemas = {}
    db_info_dict = {}
    for db in tables_data:
        db_id = db['db_id']
        schema = defaultdict(list)
        for table_id, col_name in db['column_names_original']:
            if table_id >= 0:
                table_name = db['table_names_original'][table_id]
                schema[table_name.lower()].append(col_name.lower())
        db_schemas[db_id] = dict(schema)
        db_info_dict[db_id] = db
    
    # Initialize components
    print("\n🧠 Loading embedding model...")
    #from schema_graph_builder import EmbeddingBackend, EnhancedSchemaGraphBuilder, EnhancedSchemaDataset
    
    encoder = EmbeddingBackend(config.embedding_model)
    
    # Build graphs
    print("\n📊 Building schema graphs...")
    graph_builder = EnhancedSchemaGraphBuilder(encoder)
    schema_graphs = {}
    for db_id, db_info in tqdm(db_info_dict.items(), desc="Building"):
        schema_graphs[db_id] = graph_builder.build_from_spider(db_info)
    
    # Create dataset
    print("\n📋 Creating dataset...")
    dataset = EnhancedSchemaDataset(
        examples=train_data,
        schema_graphs=schema_graphs,
        db_schemas=db_schemas,
        embedding_model=encoder,
        graph_builder=graph_builder
    )
    
    # Split
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model
    print("\n🏗️ Creating model...")
    #from enhanced_gnn_model import EnhancedSchemaGNN, EnhancedGNNConfig, EnhancedSchemaLinkingLoss
    
    model_config = EnhancedGNNConfig(
        input_dim=encoder.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_hgt_layers=config.num_hgt_layers,
        num_cross_attn_layers=config.num_cross_attn_layers,
        num_heads=config.num_heads,
        dropout=config.dropout
    )
    
    model = EnhancedSchemaGNN(model_config).to(device)
    loss_fn = EnhancedSchemaLinkingLoss()
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs
    )
    
    scaler = GradScaler() if config.use_amp and device == 'cuda' else None
    
    # Training loop
    print("\n🚀 Starting training...")
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    train_metrics = MetricsTracker()
    val_metrics = MetricsTracker()
    
    best_metric = 0.0
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    
    for epoch in range(config.num_epochs):
        # Train
        model.train()
        train_metrics.reset()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch in pbar:
            optimizer.zero_grad()
            
            # Move to device
            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_types = batch['edge_types'].to(device)
            node_types = batch['node_types'].to(device)
            q_tokens = batch['question_tokens'].to(device)
            link_features = batch['link_features'].to(device)
            column_to_table = batch['column_to_table'].to(device)
            table_labels = batch['table_labels'].to(device)
            column_labels = batch['column_labels'].to(device)
            fk_pairs = batch.get('fk_pairs', [])
            
            if scaler:
                with autocast():
                    table_scores, column_scores, embeddings = model(
                        node_features, edge_index, edge_types, node_types,
                        q_tokens, link_features, column_to_table, fk_pairs=fk_pairs
                    )
                    losses = loss_fn(
                        table_scores, column_scores,
                        table_labels, column_labels,
                        embeddings, node_types, column_to_table
                    )
                    loss = losses['total']
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                table_scores, column_scores, embeddings = model(
                    node_features, edge_index, edge_types, node_types,
                    q_tokens, link_features, column_to_table, fk_pairs=fk_pairs
                )
                losses = loss_fn(
                    table_scores, column_scores,
                    table_labels, column_labels,
                    embeddings, node_types, column_to_table
                )
                loss = losses['total']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            train_loss += loss.item()
            train_metrics.update(
                table_scores.detach().cpu(),
                column_scores.detach().cpu(),
                table_labels.cpu(),
                column_labels.cpu()
            )
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        val_metrics.reset()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                node_features = batch['node_features'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_types = batch['edge_types'].to(device)
                node_types = batch['node_types'].to(device)
                q_tokens = batch['question_tokens'].to(device)
                link_features = batch['link_features'].to(device)
                column_to_table = batch['column_to_table'].to(device)
                table_labels = batch['table_labels'].to(device)
                column_labels = batch['column_labels'].to(device)
                fk_pairs = batch.get('fk_pairs', [])
                
                table_scores, column_scores, embeddings = model(
                    node_features, edge_index, edge_types, node_types,
                    q_tokens, link_features, column_to_table, fk_pairs=fk_pairs
                )
                
                losses = loss_fn(
                    table_scores, column_scores,
                    table_labels, column_labels,
                    embeddings, node_types, column_to_table
                )
                
                val_loss += losses['total'].item()
                val_metrics.update(
                    table_scores.cpu(),
                    column_scores.cpu(),
                    table_labels.cpu(),
                    column_labels.cpu()
                )
        
        # Compute metrics
        train_m = train_metrics.compute()
        train_m['loss'] = train_loss / len(train_loader)
        val_m = val_metrics.compute()
        val_m['loss'] = val_loss / len(val_loader)
        
        history['train'].append(train_m)
        history['val'].append(val_m)
        
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print(f"  Train - Loss: {train_m['loss']:.4f}, Table: {train_m['table_recall']:.4f}, "
              f"Column: {train_m['column_recall']:.4f}, Perfect: {train_m['perfect_both']:.4f}")
        print(f"  Val   - Loss: {val_m['loss']:.4f}, Table: {val_m['table_recall']:.4f}, "
              f"Column: {val_m['column_recall']:.4f}, Perfect: {val_m['perfect_both']:.4f}")
        
        # Save best
        if val_m['perfect_both'] > best_metric:
            best_metric = val_m['perfect_both']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_m,
                'config': config.__dict__
            }, f'{config.checkpoint_dir}/best.pt')
            print(f"  ✅ Saved best model (perfect_both: {best_metric:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best perfect_both: {best_metric:.4f}")
    print("=" * 60)
    
    # Save history
    with open(f'{config.checkpoint_dir}/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


def quick_train(train_path: str, tables_path: str, epochs: int = 100) -> Dict:
    """Quick training with defaults."""
    config = TrainingConfig()
    config.train_data_path = train_path
    config.tables_path = tables_path
    config.num_epochs = epochs
    return train_model(config)


if __name__ == "__main__":
    print("Usage: from enhanced_training import train_model, TrainingConfig")


TRAIN_DATA_PATH = '/kaggle/input/yale-universitys-spider-10-nlp-dataset/spider/train_spider.json'
TABLES_PATH = '/kaggle/input/yale-universitys-spider-10-nlp-dataset/spider/tables.json'

config = TrainingConfig()
config.train_data_path = TRAIN_DATA_PATH
config.tables_path = TABLES_PATH
config.num_epochs = 7

history = train_model(config)