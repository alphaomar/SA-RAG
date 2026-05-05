"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     ENHANCED HETEROGENEOUS GRAPH TRANSFORMER FOR SCHEMA LINKING              ║
║                                                                              ║
║  Key Innovations:                                                            ║
║  1. Heterogeneous Graph Transformer (HGT) with edge-type specific attention  ║
║  2. Structure-Aware Positional Encoding (SAPE)                               ║
║  3. Semantic Relation Modeling via Edge Features                             ║
║  4. Bidirectional Question-Schema Fusion                                     ║
║  5. Multi-Task Learning with Auxiliary Objectives                            ║
║  6. Contrastive Schema Linking Loss with Hard Negative Mining                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_self_loops
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnhancedGNNConfig:
    """Configuration for Enhanced GNN Model."""
    input_dim: int = 384          # Embedding dimension
    hidden_dim: int = 256         # Hidden dimension
    num_hgt_layers: int = 8       # Number of HGT layers
    num_cross_attn_layers: int = 6  # Cross-attention layers
    num_heads: int = 8            # Attention heads
    num_edge_types: int = 8       # Edge types in schema graph
    num_node_types: int = 3       # table, column, question
    num_link_features: int = 12   # Schema linking features
    pe_dim: int = 32              # Positional encoding dimension
    dropout: float = 0.1
    use_layer_scale: bool = True  # Layer scale for stability
    layer_scale_init: float = 1e-4


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURE-AWARE POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

class StructureAwarePositionalEncoding(nn.Module):
    """
    Structure-Aware Positional Encoding (SAPE).
    
    Combines multiple positional signals:
    1. Random Walk Positional Encoding (RWPE)
    2. Laplacian Eigenvectors
    3. Distance-based encoding (table-column distance)
    4. Degree encoding
    """
    
    def __init__(self, hidden_dim: int, pe_dim: int = 32, walk_length: int = 20):
        super().__init__()
        self.pe_dim = pe_dim
        self.walk_length = walk_length
        
        # Project different PE components
        self.rwpe_proj = nn.Linear(walk_length, pe_dim // 2)
        self.degree_proj = nn.Linear(2, pe_dim // 4)  # in-degree, out-degree
        self.distance_proj = nn.Linear(1, pe_dim // 4)
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(pe_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
    def compute_rwpe(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute Random Walk Positional Encoding."""
        device = edge_index.device
        
        # Build adjacency
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        adj[edge_index[0], edge_index[1]] = 1
        adj = adj + torch.eye(num_nodes, device=device)
        
        # Transition matrix
        deg = adj.sum(dim=1).clamp(min=1)
        T = adj / deg.unsqueeze(1)
        
        # K-step random walk
        pe = torch.zeros(num_nodes, self.walk_length, device=device)
        T_k = torch.eye(num_nodes, device=device)
        
        for k in range(self.walk_length):
            T_k = T_k @ T
            pe[:, k] = T_k.diag()
            
        return pe
    
    def compute_degrees(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute in-degree and out-degree."""
        device = edge_index.device
        in_deg = torch.zeros(num_nodes, device=device)
        out_deg = torch.zeros(num_nodes, device=device)
        
        src, dst = edge_index
        in_deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=device))
        out_deg.scatter_add_(0, src, torch.ones(src.size(0), device=device))
        
        # Log-scale for stability
        in_deg = torch.log(in_deg + 1)
        out_deg = torch.log(out_deg + 1)
        
        return torch.stack([in_deg, out_deg], dim=-1)
    
    def forward(
        self, 
        edge_index: torch.Tensor, 
        num_nodes: int,
        node_types: torch.Tensor,
        column_to_table: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute structure-aware positional encoding."""
        device = edge_index.device
        
        # RWPE
        rwpe = self.compute_rwpe(edge_index, num_nodes)
        rwpe_emb = self.rwpe_proj(rwpe)
        
        # Degree encoding
        degrees = self.compute_degrees(edge_index, num_nodes)
        degree_emb = self.degree_proj(degrees)
        
        # Distance encoding (column distance from table)
        if column_to_table is not None:
            distance = torch.ones(num_nodes, 1, device=device)
            col_mask = node_types == 1
            distance[col_mask] = 0  # Columns have distance 0 from their table
        else:
            distance = torch.zeros(num_nodes, 1, device=device)
        distance_emb = self.distance_proj(distance)
        
        # Combine
        pe = torch.cat([rwpe_emb, degree_emb, distance_emb], dim=-1)
        
        return self.output_proj(pe)


# ═══════════════════════════════════════════════════════════════════════════════
# HETEROGENEOUS GRAPH TRANSFORMER LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class HeterogeneousGraphTransformer(MessagePassing):
    """
    Heterogeneous Graph Transformer (HGT) Layer.
    
    Learns different attention patterns for different edge types,
    enabling the model to understand schema structure.
    
    Edge Types:
    0: table-column (has)
    1: column-table (belongs_to)  
    2: column-column (same_table)
    3: table-table (foreign_key)
    4: column-column (foreign_key)
    5: self-loop (table)
    6: self-loop (column)
    7: semantic_similarity
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_edge_types: int = 8,
        num_node_types: int = 3,
        dropout: float = 0.1,
        use_layer_scale: bool = True,
        layer_scale_init: float = 1e-4
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_edge_types = num_edge_types
        self.num_node_types = num_node_types
        self.scale = math.sqrt(self.head_dim)
        
        # Node type embeddings
        self.node_type_emb = nn.Embedding(num_node_types, hidden_dim)
        
        # Edge type embeddings and parameters
        self.edge_type_emb = nn.Embedding(num_edge_types, hidden_dim)
        
        # Per-edge-type attention parameters
        self.edge_type_attn = nn.Parameter(
            torch.randn(num_edge_types, num_heads, self.head_dim, self.head_dim) * 0.02
        )
        self.edge_type_msg = nn.Parameter(
            torch.randn(num_edge_types, num_heads, self.head_dim, self.head_dim) * 0.02
        )
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(hidden_dim))
            self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(hidden_dim))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        node_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_types: Edge type for each edge [num_edges]
            node_types: Node type for each node [num_nodes]
        """
        # Add node type information
        node_type_features = self.node_type_emb(node_types)
        x = x + node_type_features
        
        # Attention
        h = self.norm1(x)
        attn_out = self.propagate(
            edge_index, 
            x=h, 
            edge_types=edge_types,
            node_types=node_types
        )
        
        if self.use_layer_scale:
            x = x + self.gamma1 * attn_out
        else:
            x = x + self.dropout(attn_out)
        
        # FFN
        h = self.norm2(x)
        ffn_out = self.ffn(h)
        
        if self.use_layer_scale:
            x = x + self.gamma2 * ffn_out
        else:
            x = x + ffn_out
        
        return x
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_types: torch.Tensor,
        index: torch.Tensor,
        size_i: int
    ) -> torch.Tensor:
        """Compute messages with edge-type specific attention."""
        num_edges = x_i.size(0)
        
        # Project to Q, K, V
        Q = self.q_proj(x_i).view(num_edges, self.num_heads, self.head_dim)
        K = self.k_proj(x_j).view(num_edges, self.num_heads, self.head_dim)
        V = self.v_proj(x_j).view(num_edges, self.num_heads, self.head_dim)
        
        # Edge-type specific attention transformation
        edge_attn_weights = self.edge_type_attn[edge_types]  # [E, H, D, D]
        edge_msg_weights = self.edge_type_msg[edge_types]    # [E, H, D, D]
        
        # Transform K with edge-specific weights
        K_transformed = torch.einsum('ehd,ehdc->ehc', K, edge_attn_weights)
        
        # Compute attention scores
        attn_scores = (Q * K_transformed).sum(dim=-1) / self.scale  # [E, H]
        
        # Add edge type bias
        edge_emb = self.edge_type_emb(edge_types)  # [E, D]
        edge_bias = (Q.view(num_edges, -1) * edge_emb).sum(dim=-1, keepdim=True) / self.hidden_dim
        attn_scores = attn_scores + edge_bias.expand_as(attn_scores)
        
        # Softmax over neighbors
        attn_weights = softmax(attn_scores, index, dim=0)
        attn_weights = self.dropout(attn_weights)
        
        # Transform V with edge-specific weights
        V_transformed = torch.einsum('ehd,ehdc->ehc', V, edge_msg_weights)
        
        # Apply attention
        out = attn_weights.unsqueeze(-1) * V_transformed  # [E, H, D]
        out = out.view(num_edges, self.hidden_dim)
        
        return self.out_proj(out)


# ═══════════════════════════════════════════════════════════════════════════════
# BIDIRECTIONAL QUESTION-SCHEMA FUSION
# ═══════════════════════════════════════════════════════════════════════════════

class BidirectionalSchemaFusion(nn.Module):
    """
    Deep bidirectional fusion between question and schema.
    
    Uses cross-attention in both directions with gating mechanism
    to control information flow.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Question -> Schema cross-attention
        self.q2s_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.q2s_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Schema -> Question cross-attention
        self.s2q_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.s2q_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Layer norms
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.s_norm = nn.LayerNorm(hidden_dim)
        
        # FFN for both
        self.q_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.s_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.q_ffn_norm = nn.LayerNorm(hidden_dim)
        self.s_ffn_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        question: torch.Tensor,  # [seq_len, hidden] or [1, seq_len, hidden]
        schema: torch.Tensor,    # [num_nodes, hidden]
        question_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional fusion.
        
        Returns:
            Updated question and schema representations.
        """
        # Handle dimensions
        if question.dim() == 2:
            question = question.unsqueeze(0)  # [1, seq, hidden]
        if schema.dim() == 2:
            schema = schema.unsqueeze(0)  # [1, nodes, hidden]
        
        # Question attends to Schema
        q_normed = self.q_norm(question)
        q_attended, _ = self.q2s_attn(q_normed, schema, schema)
        
        # Gated residual
        gate = self.q2s_gate(torch.cat([question, q_attended], dim=-1))
        question = question + gate * q_attended
        
        # Question FFN
        q_ffn_out = self.q_ffn(self.q_ffn_norm(question))
        question = question + q_ffn_out
        
        # Schema attends to Question (with mask if provided)
        s_normed = self.s_norm(schema)
        s_attended, _ = self.s2q_attn(
            s_normed, question, question,
            key_padding_mask=question_mask
        )
        
        # Gated residual
        gate = self.s2q_gate(torch.cat([schema, s_attended], dim=-1))
        schema = schema + gate * s_attended
        
        # Schema FFN
        s_ffn_out = self.s_ffn(self.s_ffn_norm(schema))
        schema = schema + s_ffn_out
        
        return question.squeeze(0), schema.squeeze(0)




class EnhancedConsistencyModule(nn.Module):
    """
    Enhanced consistency module with learned propagation.
    
    Ensures:
    1. Columns from relevant tables get boosted
    2. Tables with relevant columns get boosted
    3. Foreign key connected elements get signal propagation
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Column -> Table propagation
        self.col2table_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Table -> Column propagation
        self.table2col_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # FK propagation
        self.fk_propagation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Final fusion
        self.table_fusion = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
        self.column_fusion = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        table_hidden: torch.Tensor,
        column_hidden: torch.Tensor,
        table_scores: torch.Tensor,
        column_scores: torch.Tensor,
        column_to_table: torch.Tensor,
        fk_pairs: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply consistency constraints.
        """
        device = table_hidden.device
        num_tables = table_hidden.size(0)
        num_columns = column_hidden.size(0)
        
        # Ensure scores are 1D tensors
        table_scores = table_scores.view(-1)
        column_scores = column_scores.view(-1)
        
        # Column -> Table: aggregate column relevance to tables
        table_boost_from_cols = torch.zeros(num_tables, device=device)
        for col_idx in range(num_columns):
            table_idx = column_to_table[col_idx].item()
            if table_idx < 0 or table_idx >= num_tables:
                continue
            combined = torch.cat([table_hidden[table_idx], column_hidden[col_idx]])
            weight = torch.sigmoid(self.col2table_attn(combined)).squeeze()
            col_score = column_scores[col_idx].item()
            current_boost = table_boost_from_cols[table_idx].item()
            table_boost_from_cols[table_idx] = max(current_boost, weight.item() * col_score)
        
        # Table -> Column: propagate table relevance to columns
        column_boost_from_tables = torch.zeros(num_columns, device=device)
        for col_idx in range(num_columns):
            table_idx = column_to_table[col_idx].item()
            if table_idx < 0 or table_idx >= num_tables:
                continue
            combined = torch.cat([table_hidden[table_idx], column_hidden[col_idx]])
            weight = torch.sigmoid(self.table2col_attn(combined)).squeeze()
            column_boost_from_tables[col_idx] = weight.item() * table_scores[table_idx].item()
        
        # FK propagation (if available)
        table_fk_boost = torch.zeros(num_tables, device=device)
        column_fk_boost = torch.zeros(num_columns, device=device)
        
        if fk_pairs:
            for col1_idx, col2_idx in fk_pairs:
                if col1_idx < num_columns and col2_idx < num_columns:
                    # Propagate scores between FK-connected columns
                    prop_weight = self.fk_propagation(
                        (column_hidden[col1_idx] + column_hidden[col2_idx]) / 2
                    ).squeeze().item()
                    column_fk_boost[col1_idx] += prop_weight * column_scores[col2_idx].item()
                    column_fk_boost[col2_idx] += prop_weight * column_scores[col1_idx].item()
        
        # Combine signals for tables
        table_signals = torch.stack([
            table_scores,
            table_boost_from_cols,
            table_fk_boost
        ], dim=-1)
        enhanced_table_scores = self.table_fusion(table_signals).squeeze(-1)
        
        # Combine signals for columns
        column_signals = torch.stack([
            column_scores,
            column_boost_from_tables,
            column_fk_boost
        ], dim=-1)
        enhanced_column_scores = self.column_fusion(column_signals).squeeze(-1)
        
        return enhanced_table_scores, enhanced_column_scores

# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED SCHEMA LINKING FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedSchemaLinkingFeatures(nn.Module):
    """
    Enhanced schema linking feature extractor.
    
    Features (12 total):
    0. Exact match (name in question)
    1. Partial match (token overlap)
    2. Lemma match
    3. Stem match
    4. N-gram match (bigram, trigram)
    5. Semantic similarity (embedding cosine)
    6. Column type relevance
    7. Value mention
    8. Question position (normalized)
    9. Frequency in question
    10. Is in SELECT keywords context
    11. Is in WHERE keywords context
    """
    
    def __init__(self, hidden_dim: int, num_features: int = 12):
        super().__init__()
        
        self.num_features = num_features
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Feature-aware gating
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        node_features: torch.Tensor,
        link_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Enhance node features with schema linking signals.
        """
        # Project link features
        link_emb = self.feature_proj(link_features)
        
        # Gated combination
        gate = self.gate(torch.cat([node_features, link_emb], dim=-1))
        enhanced = node_features + gate * link_emb
        
        return enhanced


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL: ENHANCED SCHEMA LINKING GNN
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedSchemaGNN(nn.Module):
    """
    Enhanced Schema Linking GNN with Heterogeneous Graph Transformer.
    
    Architecture:
    1. Input projection with structure-aware positional encoding
    2. Schema linking feature enhancement
    3. Multi-layer Heterogeneous Graph Transformer
    4. Bidirectional Question-Schema Fusion
    5. Consistency-aware scoring
    """
    
    def __init__(self, config: EnhancedGNNConfig = None):
        super().__init__()
        
        if config is None:
            config = EnhancedGNNConfig()
        self.config = config
        
        # ═══════════════════════════════════════════════════════════════════
        # INPUT PROCESSING
        # ═══════════════════════════════════════════════════════════════════
        
        # Node feature projection
        self.node_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Question token projection
        self.question_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )
        
        # Structure-aware positional encoding
        self.sape = StructureAwarePositionalEncoding(
            config.hidden_dim, 
            config.pe_dim
        )
        
        # Schema linking features
        self.link_features = EnhancedSchemaLinkingFeatures(
            config.hidden_dim,
            config.num_link_features
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # HETEROGENEOUS GRAPH TRANSFORMER LAYERS
        # ═══════════════════════════════════════════════════════════════════
        
        self.hgt_layers = nn.ModuleList([
            HeterogeneousGraphTransformer(
                config.hidden_dim,
                config.num_heads,
                config.num_edge_types,
                config.num_node_types,
                config.dropout,
                config.use_layer_scale,
                config.layer_scale_init
            )
            for _ in range(config.num_hgt_layers)
        ])
        
        # ═══════════════════════════════════════════════════════════════════
        # QUESTION-SCHEMA FUSION
        # ═══════════════════════════════════════════════════════════════════
        
        self.fusion_layers = nn.ModuleList([
            BidirectionalSchemaFusion(
                config.hidden_dim,
                config.num_heads,
                config.dropout
            )
            for _ in range(config.num_cross_attn_layers)
        ])
        
        # ═══════════════════════════════════════════════════════════════════
        # SCORING HEADS
        # ═══════════════════════════════════════════════════════════════════
        
        # Question aggregation
        self.question_pool = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh()
        )
        self.question_attn = nn.Linear(config.hidden_dim, 1)
        
        # Table scorer
        self.table_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # Column scorer
        self.column_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # Consistency module
        self.consistency = EnhancedConsistencyModule(config.hidden_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        node_types: torch.Tensor,
        question_tokens: torch.Tensor,
        link_features: torch.Tensor,
        column_to_table: torch.Tensor,  # [num_columns] mapping: column_pos -> table_pos
        question_mask: Optional[torch.Tensor] = None,
        fk_pairs: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            node_features: [num_nodes, input_dim]
            edge_index: [2, num_edges]
            edge_types: [num_edges]
            node_types: [num_nodes] (0=table, 1=column)
            question_tokens: [seq_len, input_dim]
            link_features: [num_nodes, num_link_features]
            column_to_table: [num_columns] maps column position to table position
            question_mask: Optional mask for question tokens
            fk_pairs: Optional list of FK column pairs (as positions in column list)
            
        Returns:
            table_scores: [num_tables]
            column_scores: [num_columns]
            node_embeddings: [num_nodes, hidden_dim]
        """
        device = node_features.device
        
        # ═══════════════════════════════════════════════════════════════════
        # 1. INPUT PROCESSING
        # ═══════════════════════════════════════════════════════════════════
        
        # Project node features
        h = self.node_proj(node_features)
        
        # Add positional encoding
        # We need to extract table and column positions for distance encoding
        table_mask = node_types == 0
        column_mask = node_types == 1
        table_indices = table_mask.nonzero(as_tuple=True)[0]
        column_indices = column_mask.nonzero(as_tuple=True)[0]
        
        # Create column_to_table mapping in terms of node indices
        # column_to_table input maps column position (0..num_columns-1) to table position (0..num_tables-1)
        # Convert to mapping from column node index to table node index
        if len(column_to_table) > 0:
            column_to_table_node = torch.zeros(h.size(0), dtype=torch.long, device=device)
            for col_pos, col_node_idx in enumerate(column_indices):
                if col_pos < len(column_to_table):
                    table_pos = column_to_table[col_pos]
                    if table_pos < len(table_indices):
                        table_node_idx = table_indices[table_pos]
                        column_to_table_node[col_node_idx] = table_node_idx
                    else:
                        column_to_table_node[col_node_idx] = -1  # Invalid
                else:
                    column_to_table_node[col_node_idx] = -1  # Invalid
        else:
            column_to_table_node = None
        
        pe = self.sape(edge_index, h.size(0), node_types, column_to_table_node)
        h = h + pe
        
        # Enhance with schema linking features
        h = self.link_features(h, link_features)
        
        # Project question tokens
        q_tokens = self.question_proj(question_tokens)
        
        # ═══════════════════════════════════════════════════════════════════
        # 2. HETEROGENEOUS GRAPH TRANSFORMER
        # ═══════════════════════════════════════════════════════════════════
        
        for hgt_layer in self.hgt_layers:
            h = hgt_layer(h, edge_index, edge_types, node_types)
        
        # ═══════════════════════════════════════════════════════════════════
        # 3. QUESTION-SCHEMA FUSION
        # ═══════════════════════════════════════════════════════════════════
        
        for fusion_layer in self.fusion_layers:
            q_tokens, h = fusion_layer(q_tokens, h, question_mask)
        
        # ═══════════════════════════════════════════════════════════════════
        # 4. QUESTION POOLING
        # ═══════════════════════════════════════════════════════════════════
        
        q_pooled = self.question_pool(q_tokens)
        attn_weights = F.softmax(self.question_attn(q_pooled), dim=0)
        q_global = (attn_weights * q_tokens).sum(dim=0)  # [hidden_dim]
        
        # ═══════════════════════════════════════════════════════════════════
        # 5. SCORING
        # ═══════════════════════════════════════════════════════════════════
        
        table_hidden = h[table_mask]
        column_hidden = h[column_mask]
        
        # Expand question for concatenation
        q_expanded_tables = q_global.unsqueeze(0).expand(table_hidden.size(0), -1)
        q_expanded_columns = q_global.unsqueeze(0).expand(column_hidden.size(0), -1)
        
        # Initial scores
        table_scores = torch.sigmoid(
            self.table_scorer(torch.cat([table_hidden, q_expanded_tables], dim=-1))
        ).squeeze(-1)
        
        column_scores = torch.sigmoid(
            self.column_scorer(torch.cat([column_hidden, q_expanded_columns], dim=-1))
        ).squeeze(-1)
        
        # ═══════════════════════════════════════════════════════════════════
        # 6. CONSISTENCY ENFORCEMENT
        # ═══════════════════════════════════════════════════════════════════
        
        # Convert FK pairs from column positions to column node indices if needed
        if fk_pairs:
            fk_node_pairs = []
            for col1_pos, col2_pos in fk_pairs:
                if col1_pos < len(column_indices) and col2_pos < len(column_indices):
                    fk_node_pairs.append((column_indices[col1_pos].item(), 
                                        column_indices[col2_pos].item()))
        else:
            fk_node_pairs = None
        
        # Apply consistency constraints
        table_scores, column_scores = self.consistency(
            table_hidden, column_hidden,
            table_scores, column_scores,
            column_to_table,  # This is already column_pos -> table_pos
            fk_node_pairs
        )
        
        return (
            torch.clamp(table_scores, 0.0, 1.0),
            torch.clamp(column_scores, 0.0, 1.0),
            h
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedSchemaLinkingLoss(nn.Module):
    """
    Enhanced loss with hard negative mining and structure-aware penalties.
    """
    
    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        contrastive_weight: float = 0.15,
        consistency_weight: float = 0.15,
        structure_weight: float = 0.1
    ):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.contrastive_weight = contrastive_weight
        self.consistency_weight = consistency_weight
        self.structure_weight = structure_weight
        
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal loss for class imbalance."""
        pred_clamped = torch.clamp(pred, min=1e-7, max=1-1e-7)
        
        # BCE
        bce = -target * torch.log(pred_clamped) - (1 - target) * torch.log(1 - pred_clamped)
        
        pt = torch.where(target == 1, pred_clamped, 1 - pred_clamped)
        alpha_t = torch.where(target == 1, self.focal_alpha, 1 - self.focal_alpha)
        
        focal_weight = alpha_t * (1 - pt) ** self.focal_gamma
        
        return (focal_weight * bce).mean()
    
    def contrastive_loss_with_hard_negatives(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """Contrastive loss with hard negative mining."""
        if labels.sum() == 0 or labels.sum() == len(labels):
            return torch.tensor(0.0, device=embeddings.device)
        
        # Normalize
        embeddings = F.normalize(embeddings, dim=-1)
        
        # Similarity matrix
        sim = embeddings @ embeddings.t() / temperature
        
        # Masks
        pos_mask = labels.unsqueeze(0) * labels.unsqueeze(1)  # Both positive
        neg_mask = (1 - labels.unsqueeze(0)) * labels.unsqueeze(1)  # Query pos, key neg
        
        # Hard negative mining: select top-k most similar negatives
        pos_indices = labels.nonzero(as_tuple=True)[0]
        neg_indices = (1 - labels).nonzero(as_tuple=True)[0]
        
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        loss = torch.tensor(0.0, device=embeddings.device)
        
        for pos_idx in pos_indices:
            # Get similarities with all
            pos_sims = sim[pos_idx, pos_indices]
            neg_sims = sim[pos_idx, neg_indices]
            
            # InfoNCE-style loss
            pos_term = pos_sims.mean()
            neg_term = torch.logsumexp(neg_sims, dim=0)
            
            loss = loss + (-pos_term + neg_term)
        
        return loss / len(pos_indices)
    
    def structure_loss(
        self,
        table_scores: torch.Tensor,
        column_scores: torch.Tensor,
        column_to_table: torch.Tensor,
        table_labels: torch.Tensor,
        column_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Structure-aware loss:
        1. If column is relevant, its table should be relevant
        2. Penalize high column score when table score is low
        """
        loss = torch.tensor(0.0, device=table_scores.device)
        
        for col_idx in range(len(column_scores)):
            table_idx = column_to_table[col_idx]
            col_score = column_scores[col_idx]
            table_score = table_scores[table_idx]
            col_label = column_labels[col_idx]
            
            # If column is relevant (label=1), table should have high score
            if col_label > 0.5:
                loss = loss + F.relu(0.5 - table_score)
            
            # Column score shouldn't exceed table score by too much
            loss = loss + F.relu(col_score - table_score - 0.2)
        
        return loss / max(len(column_scores), 1)
    
    def forward(
        self,
        table_scores: torch.Tensor,
        column_scores: torch.Tensor,
        table_labels: torch.Tensor,
        column_labels: torch.Tensor,
        node_embeddings: torch.Tensor,
        node_types: torch.Tensor,
        column_to_table: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        
        # Focal losses
        table_loss = self.focal_loss(table_scores, table_labels)
        column_loss = self.focal_loss(column_scores, column_labels)
        
        # Contrastive losses
        table_mask = node_types == 0
        column_mask = node_types == 1
        
        table_emb = node_embeddings[table_mask]
        column_emb = node_embeddings[column_mask]
        
        table_contrastive = self.contrastive_loss_with_hard_negatives(table_emb, table_labels)
        column_contrastive = self.contrastive_loss_with_hard_negatives(column_emb, column_labels)
        
        # Structure loss
        struct_loss = self.structure_loss(
            table_scores, column_scores, column_to_table,
            table_labels, column_labels
        )
        
        # Total
        total = (
            table_loss + column_loss +
            self.contrastive_weight * (table_contrastive + column_contrastive) +
            self.structure_weight * struct_loss
        )
        
        return {
            'total': total,
            'table_loss': table_loss,
            'column_loss': column_loss,
            'contrastive_loss': table_contrastive + column_contrastive,
            'structure_loss': struct_loss
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(config: EnhancedGNNConfig = None) -> EnhancedSchemaGNN:
    """Create and initialize model."""
    model = EnhancedSchemaGNN(config)
    print(f"Created EnhancedSchemaGNN with {count_parameters(model):,} parameters")
    return model


if __name__ == "__main__":
    # Test model creation
    config = EnhancedGNNConfig()
    model = create_model(config)
    
    # Test forward pass
    batch_size = 1
    num_nodes = 20
    num_edges = 50
    seq_len = 32
    
    node_features = torch.randn(num_nodes, config.input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_types = torch.randint(0, config.num_edge_types, (num_edges,))
    node_types = torch.cat([torch.zeros(5), torch.ones(15)]).long()  # 5 tables, 15 columns
    question_tokens = torch.randn(seq_len, config.input_dim)
    link_features = torch.randn(num_nodes, config.num_link_features)
    column_to_table = torch.randint(0, 5, (15,))  # Map 15 columns to 5 tables
    
    model.eval()
    with torch.no_grad():
        table_scores, column_scores, embeddings = model(
            node_features, edge_index, edge_types, node_types,
            question_tokens, link_features, column_to_table
        )
    
    print(f"Table scores shape: {table_scores.shape}")
    print(f"Column scores shape: {column_scores.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print("✅ Model test passed!")