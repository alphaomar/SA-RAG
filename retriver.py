"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              ENHANCED SEMANTIC SCHEMA RETRIEVER                              ║
║                                                                              ║
║  Key Features:                                                               ║
║  1. Semantic-first retrieval (not keyword-based)                             ║
║  2. Structure-aware ranking                                                  ║
║  3. Relationship propagation                                                 ║
║  4. Dynamic thresholding                                                     ║
║  5. Confidence calibration                                                   ║
║  6. Cross-dataset generalization                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class RetrievalResult:
    """Structured retrieval result."""
    tables: List[Tuple[str, float]]  # [(table_name, score)]
    columns: List[Tuple[str, str, float]]  # [(table, column, score)]
    relationships: List[str]  # ["table1.col1 = table2.col2"]
    confidence: float
    reasoning: Dict[str, str]  # Explanation for selections


class EnhancedSchemaRetriever:
    """
    Enhanced Schema Retriever with semantic understanding.
    
    Key improvements over keyword-based approaches:
    1. Uses GNN embeddings for semantic similarity
    2. Propagates relevance through schema structure
    3. Considers table-column relationships
    4. Calibrated confidence scores
    """
    
    def __init__(
        self,
        model,
        embedding_backend,
        schema_graphs: Dict,
        db_schemas: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.encoder = embedding_backend
        self.schema_graphs = schema_graphs
        self.db_schemas = db_schemas
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        embedding_model: str,
        tables_path: str,
        model_class=None,
        model_config=None,
        device: str = None
    ):
        """Load retriever from checkpoint."""
        from sentence_transformers import SentenceTransformer
        
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load embedding model
        class EmbeddingBackend:
            def __init__(self, model_name):
                self.model = SentenceTransformer(model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            def encode(self, texts):
                return self.model.encode(texts, show_progress_bar=False)
            
            def encode_single(self, text):
                return self.model.encode(text, show_progress_bar=False)
        
        encoder = EmbeddingBackend(embedding_model)
        
        # Load tables
        with open(tables_path, 'r') as f:
            tables_data = json.load(f)
        
        # Build schemas and graphs
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
        
        # Import graph builder
        #from schema_graph_builder import EnhancedSchemaGraphBuilder
        
        graph_builder = EnhancedSchemaGraphBuilder(encoder)
        schema_graphs = {}
        for db_id, db_info in db_info_dict.items():
            schema_graphs[db_id] = graph_builder.build_from_spider(db_info)
        
        # Create model
        if model_class is None:
            #from enhanced_gnn_model import EnhancedSchemaGNN, EnhancedGNNConfig
            model_class = EnhancedSchemaGNN
            model_config = EnhancedGNNConfig(input_dim=encoder.embedding_dim)
        
        model = model_class(model_config)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return cls(
            model=model,
            embedding_backend=encoder,
            schema_graphs=schema_graphs,
            db_schemas=db_schemas,
            device=device
        )
    
    @torch.no_grad()
    def retrieve(
        self,
        question: str,
        db_id: str,
        top_k_tables: int = 5,
        top_k_columns: int = 15,
        min_table_threshold: float = 0.1,
        min_column_threshold: float = 0.1,
        use_dynamic_threshold: bool = True,
        propagate_scores: bool = True,
        include_relationships: bool = True
    ) -> RetrievalResult:
        """
        Retrieve relevant schema elements.
        
        Args:
            question: Natural language question
            db_id: Database ID
            top_k_tables: Max tables to retrieve
            top_k_columns: Max columns to retrieve
            min_table_threshold: Minimum table score
            min_column_threshold: Minimum column score
            use_dynamic_threshold: Adjust thresholds based on score distribution
            propagate_scores: Propagate scores through relationships
            include_relationships: Include FK relationships in result
        """
        if db_id not in self.schema_graphs:
            raise ValueError(f"Unknown database: {db_id}")
        
        graph = self.schema_graphs[db_id]
        
        # Encode question
        q_emb = torch.FloatTensor(self.encoder.encode_single(question)).to(self.device)
        
        # Encode question tokens
        #from schema_graph_builder import EnhancedLinkingFeatures
        link_extractor = EnhancedLinkingFeatures()
        tokens = link_extractor.tokenize(question)[:64]
        token_embs = [self.encoder.encode_single(t) for t in tokens]
        while len(token_embs) < 64:
            token_embs.append(np.zeros(self.encoder.embedding_dim))
        q_tokens = torch.FloatTensor(np.array(token_embs)).to(self.device)
        
        # Compute link features
        #from schema_graph_builder import EnhancedSchemaGraphBuilder
        temp_builder = EnhancedSchemaGraphBuilder(self.encoder)
        link_features = temp_builder.compute_link_features(graph, question).to(self.device)
        
        # Move graph to device
        node_features = graph.node_features.to(self.device)
        edge_index = graph.edge_index.to(self.device)
        edge_types = graph.edge_types.to(self.device)
        node_types = graph.node_types.to(self.device)
        column_to_table = graph.column_to_table.to(self.device)
        
        # Forward pass
        table_scores, column_scores, embeddings = self.model(
            node_features=node_features,
            edge_index=edge_index,
            edge_types=edge_types,
            node_types=node_types,
            question_tokens=q_tokens,
            link_features=link_features,
            column_to_table=column_to_table,
            fk_pairs=graph.fk_pairs
        )
        
        table_scores = table_scores.cpu().numpy()
        column_scores = column_scores.cpu().numpy()
        
        # Apply score propagation
        if propagate_scores:
            table_scores, column_scores = self._propagate_scores(
                table_scores, column_scores, graph
            )
        
        # Apply dynamic thresholding
        if use_dynamic_threshold:
            table_threshold = self._compute_dynamic_threshold(table_scores, min_table_threshold)
            column_threshold = self._compute_dynamic_threshold(column_scores, min_column_threshold)
        else:
            table_threshold = min_table_threshold
            column_threshold = min_column_threshold
        
        # Select tables
        selected_tables = []
        reasoning = {}
        
        sorted_table_idx = np.argsort(table_scores)[::-1]
        for i in sorted_table_idx[:top_k_tables]:
            if table_scores[i] >= table_threshold:
                t_idx = graph.table_indices[i]
                t_name = graph.node_names[t_idx]
                selected_tables.append((t_name, float(table_scores[i])))
                reasoning[t_name] = f"Score: {table_scores[i]:.3f}"
        
        # Select columns
        selected_columns = []
        selected_table_names = set(t[0] for t in selected_tables)
        
        sorted_col_idx = np.argsort(column_scores)[::-1]
        for i in sorted_col_idx[:top_k_columns * 2]:  # Consider more to ensure coverage
            if column_scores[i] >= column_threshold:
                c_idx = graph.column_indices[i]
                c_name = graph.node_names[c_idx]
                table_name, col_name = c_name.split('.') if '.' in c_name else ('', c_name)
                
                # Include if table is selected OR score is very high
                if table_name in selected_table_names or column_scores[i] > 0.5:
                    selected_columns.append((table_name, col_name, float(column_scores[i])))
                    reasoning[c_name] = f"Score: {column_scores[i]:.3f}"
                    
                    # Auto-include table if column has high score
                    if table_name and table_name not in selected_table_names:
                        # Find table score
                        for j, t_idx in enumerate(graph.table_indices):
                            if graph.node_names[t_idx] == table_name:
                                selected_tables.append((table_name, float(table_scores[j])))
                                selected_table_names.add(table_name)
                                reasoning[table_name] = f"Auto-included for column {col_name}"
                                break
        
        # Limit columns
        selected_columns = selected_columns[:top_k_columns]
        
        # Get relationships
        relationships = []
        if include_relationships:
            relationships = self._get_relationships(
                graph, selected_table_names
            )
        
        # Compute confidence
        confidence = self._compute_confidence(
            table_scores, column_scores,
            selected_tables, selected_columns
        )
        
        return RetrievalResult(
            tables=selected_tables,
            columns=selected_columns,
            relationships=relationships,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _propagate_scores(
        self,
        table_scores: np.ndarray,
        column_scores: np.ndarray,
        graph
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate scores through schema structure."""
        
        # Column -> Table: boost tables with high-scoring columns
        table_boost = np.zeros_like(table_scores)
        for i, col_idx in enumerate(graph.column_indices):
            col_node = graph.nodes[col_idx]
            if col_node.table_idx is not None:
                table_boost[col_node.table_idx] = max(
                    table_boost[col_node.table_idx],
                    column_scores[i] * 0.5
                )
        
        # Table -> Column: boost columns of relevant tables
        column_boost = np.zeros_like(column_scores)
        for i, col_idx in enumerate(graph.column_indices):
            col_node = graph.nodes[col_idx]
            if col_node.table_idx is not None:
                column_boost[i] += table_scores[col_node.table_idx] * 0.3
        
        # FK propagation: if a column is relevant, related FK columns get boost
        for col1_idx, col2_idx in graph.fk_pairs:
            # Find positions in column_indices
            col1_pos = col2_pos = None
            for i, c_idx in enumerate(graph.column_indices):
                if c_idx == col1_idx:
                    col1_pos = i
                if c_idx == col2_idx:
                    col2_pos = i
            
            if col1_pos is not None and col2_pos is not None:
                # Propagate scores
                column_boost[col1_pos] += column_scores[col2_pos] * 0.2
                column_boost[col2_pos] += column_scores[col1_pos] * 0.2
        
        # Apply boosts
        enhanced_table = table_scores + table_boost
        enhanced_column = column_scores + column_boost
        
        # Renormalize
        enhanced_table = np.clip(enhanced_table, 0, 1)
        enhanced_column = np.clip(enhanced_column, 0, 1)
        
        return enhanced_table, enhanced_column
    
    def _compute_dynamic_threshold(
        self,
        scores: np.ndarray,
        min_threshold: float
    ) -> float:
        """Compute dynamic threshold based on score distribution."""
        if len(scores) == 0:
            return min_threshold
        
        sorted_scores = np.sort(scores)[::-1]
        
        # Find gap in top scores
        if len(sorted_scores) > 1:
            gaps = sorted_scores[:-1] - sorted_scores[1:]
            max_gap_idx = np.argmax(gaps)
            
            # Only use gap if it's significant
            if gaps[max_gap_idx] > 0.1:
                threshold = sorted_scores[max_gap_idx + 1]
                return max(threshold, min_threshold)
        
        # Default: use percentile-based threshold
        threshold = np.percentile(scores, 70)
        return max(threshold, min_threshold)
    
    def _get_relationships(
        self,
        graph,
        selected_tables: Set[str]
    ) -> List[str]:
        """Get FK relationships between selected tables."""
        relationships = []
        seen = set()
        
        for col1_idx, col2_idx in graph.fk_pairs:
            col1_name = graph.node_names[col1_idx]
            col2_name = graph.node_names[col2_idx]
            
            t1 = col1_name.split('.')[0] if '.' in col1_name else ''
            t2 = col2_name.split('.')[0] if '.' in col2_name else ''
            
            if t1 in selected_tables or t2 in selected_tables:
                rel = f"{col1_name} = {col2_name}"
                rel_rev = f"{col2_name} = {col1_name}"
                
                if rel not in seen and rel_rev not in seen:
                    relationships.append(rel)
                    seen.add(rel)
                    seen.add(rel_rev)
        
        return relationships
    
    def _compute_confidence(
        self,
        table_scores: np.ndarray,
        column_scores: np.ndarray,
        selected_tables: List[Tuple[str, float]],
        selected_columns: List[Tuple[str, str, float]]
    ) -> float:
        """Compute retrieval confidence score."""
        if not selected_tables:
            return 0.0
        
        # Average scores of selected elements
        table_avg = np.mean([s for _, s in selected_tables])
        column_avg = np.mean([s for _, _, s in selected_columns]) if selected_columns else 0.0
        
        # Score gap (separation from non-selected)
        all_table_scores = sorted(table_scores, reverse=True)
        if len(all_table_scores) > len(selected_tables):
            gap = all_table_scores[len(selected_tables)-1] - all_table_scores[len(selected_tables)]
        else:
            gap = 0.5
        
        # Combine
        confidence = 0.4 * table_avg + 0.4 * column_avg + 0.2 * min(gap * 2, 1.0)
        
        return float(np.clip(confidence, 0, 1))
    
    def format_for_llm(
        self,
        result: RetrievalResult,
        db_id: str,
        format_type: str = 'detailed'
    ) -> str:
        """Format retrieval result for LLM prompt."""
        if format_type == 'detailed':
            return self._format_detailed(result, db_id)
        elif format_type == 'compact':
            return self._format_compact(result, db_id)
        else:
            return self._format_minimal(result, db_id)
    
    def _format_detailed(self, result: RetrievalResult, db_id: str) -> str:
        """Detailed format with all information."""
        lines = [f"## Database: {db_id}", ""]
        
        # Tables
        lines.append("### Tables")
        for table, score in result.tables:
            lines.append(f"- **{table}** (relevance: {score:.2f})")
            
            # Include columns for this table
            table_cols = [(c, s) for t, c, s in result.columns if t == table]
            if table_cols:
                for col, col_score in table_cols:
                    col_type = self._get_column_type(db_id, table, col)
                    markers = []
                    if self._is_primary_key(db_id, table, col):
                        markers.append("PK")
                    if self._is_foreign_key(db_id, table, col):
                        markers.append("FK")
                    marker_str = f" [{','.join(markers)}]" if markers else ""
                    lines.append(f"  - {col} ({col_type}){marker_str}")
        lines.append("")
        
        # Relevant columns from other tables
        other_cols = [(t, c, s) for t, c, s in result.columns if t not in [x[0] for x in result.tables]]
        if other_cols:
            lines.append("### Additional Relevant Columns")
            for table, col, score in other_cols[:5]:
                lines.append(f"- {table}.{col} (relevance: {score:.2f})")
            lines.append("")
        
        # Relationships
        if result.relationships:
            lines.append("### Relationships (JOIN conditions)")
            for rel in result.relationships:
                lines.append(f"- {rel}")
            lines.append("")
        
        lines.append(f"*Retrieval confidence: {result.confidence:.2f}*")
        
        return '\n'.join(lines)
    
    def _format_compact(self, result: RetrievalResult, db_id: str) -> str:
        """Compact format."""
        lines = [f"Database: {db_id}"]
        
        # Tables with columns
        for table, _ in result.tables:
            cols = [c for t, c, _ in result.columns if t == table]
            lines.append(f"{table}: {', '.join(cols[:8])}")
        
        # Joins
        if result.relationships:
            lines.append(f"Joins: {'; '.join(result.relationships)}")
        
        return '\n'.join(lines)
    
    def _format_minimal(self, result: RetrievalResult, db_id: str) -> str:
        """Minimal format."""
        tables = [t for t, _ in result.tables]
        cols = [f"{t}.{c}" for t, c, _ in result.columns[:10]]
        return f"Tables: {', '.join(tables)} | Columns: {', '.join(cols)}"
    
    def _get_column_type(self, db_id: str, table: str, col: str) -> str:
        """Get column type from schema."""
        # This would need access to column types from the original schema
        return "text"  # Default
    
    def _is_primary_key(self, db_id: str, table: str, col: str) -> bool:
        """Check if column is primary key."""
        return 'id' in col.lower() and col.lower().endswith('id')
    
    def _is_foreign_key(self, db_id: str, table: str, col: str) -> bool:
        """Check if column is foreign key."""
        return 'id' in col.lower()
    
    def batch_retrieve(
        self,
        questions: List[str],
        db_ids: List[str],
        **kwargs
    ) -> List[RetrievalResult]:
        """Batch retrieval."""
        return [
            self.retrieve(q, db_id, **kwargs)
            for q, db_id in zip(questions, db_ids)
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_retriever(
    retriever: EnhancedSchemaRetriever,
    test_data: List[Dict],
    db_schemas: Dict[str, Dict],
    top_k_tables: int = 5,
    top_k_columns: int = 15,
    verbose: bool = True
) -> Dict[str, float]:
    """Evaluate retriever on test data."""
    from schema_graph_builder import RobustSQLParser
    
    # Build parsers
    parsers = {}
    for db_id, schema in db_schemas.items():
        parsers[db_id] = RobustSQLParser(schema)
    
    # Metrics
    table_recalls = []
    column_recalls = []
    perfect_table = []
    perfect_column = []
    perfect_both = []
    
    from tqdm import tqdm
    
    for example in tqdm(test_data, desc="Evaluating"):
        question = example['question']
        sql = example.get('query', example.get('sql', ''))
        db_id = example['db_id']
        
        if db_id not in parsers:
            continue
        
        gt_tables, gt_columns = parsers[db_id].parse(sql)
        
        if not gt_tables:
            continue
        
        try:
            result = retriever.retrieve(
                question, db_id,
                top_k_tables=top_k_tables,
                top_k_columns=top_k_columns
            )
        except Exception as e:
            print(f"Error on {question[:50]}...: {e}")
            continue
        
        pred_tables = set(t[0] for t in result.tables)
        pred_columns = set(f"{t}.{c}" for t, c, _ in result.columns)
        
        # Table metrics
        table_hit = len(pred_tables & gt_tables)
        table_recall = table_hit / len(gt_tables)
        table_recalls.append(table_recall)
        perfect_table.append(1.0 if gt_tables <= pred_tables else 0.0)
        
        # Column metrics
        if gt_columns:
            column_hit = len(pred_columns & gt_columns)
            column_recall = column_hit / len(gt_columns)
            column_recalls.append(column_recall)
            perfect_column.append(1.0 if gt_columns <= pred_columns else 0.0)
            
            is_perfect = (gt_tables <= pred_tables) and (gt_columns <= pred_columns)
            perfect_both.append(1.0 if is_perfect else 0.0)
    
    metrics = {
        'table_recall': np.mean(table_recalls) if table_recalls else 0,
        'column_recall': np.mean(column_recalls) if column_recalls else 0,
        'perfect_table': np.mean(perfect_table) if perfect_table else 0,
        'perfect_column': np.mean(perfect_column) if perfect_column else 0,
        'perfect_both': np.mean(perfect_both) if perfect_both else 0,
        'num_examples': len(table_recalls)
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Examples: {metrics['num_examples']}")
        print(f"Table Recall@{top_k_tables}: {metrics['table_recall']:.4f}")
        print(f"Column Recall@{top_k_columns}: {metrics['column_recall']:.4f}")
        print(f"Perfect Table: {metrics['perfect_table']:.4f}")
        print(f"Perfect Column: {metrics['perfect_column']:.4f}")
        print(f"Perfect Both: {metrics['perfect_both']:.4f}")
        print("=" * 60)
    
    return metrics


# if __name__ == "__main__":
#     print("Enhanced Schema Retriever")
#     print("Use from_checkpoint() to load a trained model")
print("Done")