"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          ENHANCED SCHEMA KNOWLEDGE GRAPH BUILDER                             ║
║                                                                              ║
║  Creates rich graph structure for GNN learning:                              ║
║  1. Heterogeneous node types (table, column, value)                          ║
║  2. Multiple edge types with semantic meaning                                ║
║  3. Rich node features from embeddings + structural features                 ║
║  4. Schema linking features (12 dimensions)                                  ║
║  5. Positional encoding for structure awareness                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import re
import json

try:
    import nltk
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SchemaNode:
    """Node in schema graph."""
    idx: int
    name: str
    original_name: str
    node_type: int  # 0=table, 1=column
    table_idx: Optional[int] = None
    col_type: str = 'text'
    is_primary_key: bool = False
    is_foreign_key: bool = False
    semantic_type: str = 'general'  # general, numeric, temporal, identifier


@dataclass
class SchemaEdge:
    """Edge in schema graph."""
    src: int
    dst: int
    edge_type: int
    relation: str


class EdgeType:
    """Edge type definitions."""
    TABLE_HAS_COLUMN = 0      # table -> column
    COLUMN_BELONGS_TO = 1     # column -> table
    SAME_TABLE = 2            # column <-> column (same table)
    PRIMARY_KEY = 3           # table -> pk column
    FOREIGN_KEY = 4           # column -> column (FK relationship)
    FOREIGN_KEY_REV = 5       # reverse FK
    TABLE_RELATED = 6         # table <-> table (via FK)
    SELF_LOOP = 7             # self-connections


@dataclass
class SchemaGraph:
    """Complete schema graph."""
    db_id: str
    nodes: List[SchemaNode]
    edges: List[SchemaEdge]
    node_features: torch.Tensor
    node_types: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor
    node_names: List[str]
    table_indices: List[int]
    column_indices: List[int]
    column_to_table: torch.Tensor
    fk_pairs: List[Tuple[int, int]]
    positional_encoding: Optional[torch.Tensor] = None


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA LINKING FEATURE EXTRACTOR (12 FEATURES)
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedLinkingFeatures:
    """
    Extract 12-dimensional schema linking features.
    
    Features:
    0. Exact match
    1. Partial match (token overlap)
    2. Lemma match
    3. Stem match
    4. Bigram match
    5. Trigram match
    6. Semantic similarity score
    7. Column type relevance
    8. Value mention score
    9. Question position (normalized)
    10. Token frequency in question
    11. Contextual relevance (SELECT/WHERE context)
    """
    
    NUMERIC_KEYWORDS = {
        'count', 'sum', 'average', 'avg', 'total', 'maximum', 'minimum',
        'max', 'min', 'number', 'how many', 'amount', 'price', 'cost',
        'age', 'salary', 'revenue', 'percentage', 'rate', 'score'
    }
    
    DATE_KEYWORDS = {
        'date', 'time', 'year', 'month', 'day', 'when', 'birthday',
        'created', 'updated', 'timestamp', 'period', 'duration', 'since'
    }
    
    SELECT_KEYWORDS = {'show', 'list', 'find', 'get', 'display', 'what', 'name', 'return'}
    WHERE_KEYWORDS = {'where', 'with', 'having', 'whose', 'which', 'from', 'in'}
    
    def __init__(self):
        if NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
        else:
            self.lemmatizer = None
            self.stemmer = None
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        if NLTK_AVAILABLE:
            return word_tokenize(text.lower())
        return text.lower().split()
    
    def get_lemma(self, word: str) -> str:
        if self.lemmatizer:
            return self.lemmatizer.lemmatize(word.lower())
        return word.lower()
    
    def get_stem(self, word: str) -> str:
        if self.stemmer:
            return self.stemmer.stem(word.lower())
        return word.lower()
    
    def get_ngrams(self, tokens: List[str], n: int) -> Set[str]:
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngrams.add(' '.join(tokens[i:i+n]))
        return ngrams
    
    def extract(
        self,
        name: str,
        question: str,
        question_tokens: List[str],
        col_type: str = 'text',
        semantic_sim: float = 0.0,
        db_values: Optional[Set[str]] = None
    ) -> np.ndarray:
        """Extract 12-dimensional features."""
        features = np.zeros(12, dtype=np.float32)
        
        name_lower = name.lower()
        name_tokens = name_lower.replace('_', ' ').split()
        question_lower = question.lower()
        q_tokens = [t.lower() for t in question_tokens]
        
        # Lemmas and stems
        name_lemmas = [self.get_lemma(t) for t in name_tokens]
        q_lemmas = [self.get_lemma(t) for t in q_tokens]
        name_stems = [self.get_stem(t) for t in name_tokens]
        q_stems = [self.get_stem(t) for t in q_tokens]
        
        # Feature 0: Exact match
        if name_lower in question_lower or name_lower.replace('_', ' ') in question_lower:
            features[0] = 1.0
        elif any(t == name_lower for t in q_tokens):
            features[0] = 1.0
        
        # Feature 1: Partial match
        for token in name_tokens:
            if len(token) > 2 and token in q_tokens:
                features[1] = 1.0
                break
        
        # Feature 2: Lemma match
        for lemma in name_lemmas:
            if len(lemma) > 2 and lemma in q_lemmas:
                features[2] = 1.0
                break
        
        # Feature 3: Stem match
        for stem in name_stems:
            if len(stem) > 2 and stem in q_stems:
                features[3] = 1.0
                break
        
        # Feature 4: Bigram match
        q_bigrams = self.get_ngrams(q_tokens, 2)
        name_joined = ' '.join(name_tokens)
        if name_joined in q_bigrams:
            features[4] = 1.0
        else:
            name_bigrams = self.get_ngrams(name_tokens, 2)
            if name_bigrams & q_bigrams:
                features[4] = 0.5
        
        # Feature 5: Trigram match
        q_trigrams = self.get_ngrams(q_tokens, 3)
        if name_joined in q_trigrams:
            features[5] = 1.0
        
        # Feature 6: Semantic similarity (from embedding)
        features[6] = semantic_sim
        
        # Feature 7: Column type relevance
        col_type_lower = col_type.lower()
        is_numeric = col_type_lower in ('int', 'integer', 'real', 'float', 'number', 'decimal')
        is_date = col_type_lower in ('date', 'datetime', 'time', 'timestamp')
        
        if is_numeric and any(kw in question_lower for kw in self.NUMERIC_KEYWORDS):
            features[7] = 1.0
        elif is_date and any(kw in question_lower for kw in self.DATE_KEYWORDS):
            features[7] = 1.0
        
        # Feature 8: Value mention
        if db_values:
            for value in db_values:
                if str(value).lower() in question_lower:
                    features[8] = 1.0
                    break
        
        # Feature 9: Position in question (normalized)
        for i, token in enumerate(q_tokens):
            for name_token in name_tokens:
                if name_token in token:
                    features[9] = 1.0 - (i / max(len(q_tokens), 1))
                    break
        
        # Feature 10: Token frequency
        freq = sum(1 for t in q_tokens if any(nt in t for nt in name_tokens))
        features[10] = min(freq / 3.0, 1.0)  # Normalize
        
        # Feature 11: Contextual relevance
        # Check if name is mentioned near SELECT or WHERE keywords
        for i, token in enumerate(q_tokens):
            for name_token in name_tokens:
                if name_token in token:
                    # Check surrounding context
                    context = q_tokens[max(0, i-3):i+3]
                    if any(kw in ' '.join(context) for kw in self.SELECT_KEYWORDS):
                        features[11] = 0.7
                    elif any(kw in ' '.join(context) for kw in self.WHERE_KEYWORDS):
                        features[11] = 1.0
                    break
        
        return features


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED SCHEMA GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedSchemaGraphBuilder:
    """
    Builds rich schema graphs for GNN training.
    
    Creates graphs with:
    - Multiple node types
    - Multiple edge types
    - Rich node features
    - Structural encoding
    """
    
    def __init__(
        self,
        embedding_model,
        pe_walk_length: int = 20,
        add_semantic_edges: bool = True,
        semantic_threshold: float = 0.6
    ):
        self.encoder = embedding_model
        self.pe_walk_length = pe_walk_length
        self.add_semantic_edges = add_semantic_edges
        self.semantic_threshold = semantic_threshold
        self.link_extractor = EnhancedLinkingFeatures()
    
    def build_from_spider(self, db_info: Dict) -> SchemaGraph:
        """Build schema graph from Spider format."""
        db_id = db_info['db_id']
        table_names = db_info['table_names_original']
        column_info = db_info['column_names_original']
        column_types = db_info.get('column_types', ['text'] * len(column_info))
        primary_keys = set(db_info.get('primary_keys', []))
        foreign_keys = db_info.get('foreign_keys', [])
        
        # Build nodes
        nodes = []
        node_names = []
        table_indices = []
        column_indices = []
        column_to_table_map = {}
        
        # Add table nodes
        for t_idx, table_name in enumerate(table_names):
            node = SchemaNode(
                idx=len(nodes),
                name=table_name.lower(),
                original_name=table_name,
                node_type=0,  # table
                semantic_type='table'
            )
            nodes.append(node)
            node_names.append(table_name.lower())
            table_indices.append(node.idx)
        
        # Add column nodes
        for col_idx, (table_id, col_name) in enumerate(column_info):
            if table_id < 0:
                continue
            
            col_type = column_types[col_idx] if col_idx < len(column_types) else 'text'
            is_pk = col_idx in primary_keys
            is_fk = any(col_idx in pair for pair in foreign_keys)
            
            # Determine semantic type
            semantic_type = self._infer_semantic_type(col_name, col_type)
            
            node = SchemaNode(
                idx=len(nodes),
                name=col_name.lower(),
                original_name=col_name,
                node_type=1,  # column
                table_idx=table_id,
                col_type=col_type,
                is_primary_key=is_pk,
                is_foreign_key=is_fk,
                semantic_type=semantic_type
            )
            nodes.append(node)
            
            full_name = f"{table_names[table_id].lower()}.{col_name.lower()}"
            node_names.append(full_name)
            column_indices.append(node.idx)
            column_to_table_map[node.idx] = table_id
        
        # Build edges
        edges = []
        
        # Table-Column edges
        for node in nodes:
            if node.node_type == 1 and node.table_idx is not None:
                table_node_idx = table_indices[node.table_idx]
                
                # Table -> Column
                edges.append(SchemaEdge(
                    src=table_node_idx,
                    dst=node.idx,
                    edge_type=EdgeType.TABLE_HAS_COLUMN,
                    relation='has_column'
                ))
                
                # Column -> Table
                edges.append(SchemaEdge(
                    src=node.idx,
                    dst=table_node_idx,
                    edge_type=EdgeType.COLUMN_BELONGS_TO,
                    relation='belongs_to'
                ))
                
                # Primary key edge
                if node.is_primary_key:
                    edges.append(SchemaEdge(
                        src=table_node_idx,
                        dst=node.idx,
                        edge_type=EdgeType.PRIMARY_KEY,
                        relation='primary_key'
                    ))
        
        # Same-table column edges
        table_columns = defaultdict(list)
        for node in nodes:
            if node.node_type == 1 and node.table_idx is not None:
                table_columns[node.table_idx].append(node.idx)
        
        for table_idx, col_nodes in table_columns.items():
            for i, col1 in enumerate(col_nodes):
                for col2 in col_nodes[i+1:]:
                    edges.append(SchemaEdge(
                        src=col1, dst=col2,
                        edge_type=EdgeType.SAME_TABLE,
                        relation='same_table'
                    ))
                    edges.append(SchemaEdge(
                        src=col2, dst=col1,
                        edge_type=EdgeType.SAME_TABLE,
                        relation='same_table'
                    ))
        
        # Foreign key edges
        fk_pairs = []
        col_idx_to_node = {}
        for node in nodes:
            if node.node_type == 1:
                # Find original column index
                for orig_idx, (t_id, c_name) in enumerate(column_info):
                    if t_id >= 0 and c_name.lower() == node.name and t_id == node.table_idx:
                        col_idx_to_node[orig_idx] = node.idx
                        break
        
        for col1_orig, col2_orig in foreign_keys:
            if col1_orig in col_idx_to_node and col2_orig in col_idx_to_node:
                col1_node = col_idx_to_node[col1_orig]
                col2_node = col_idx_to_node[col2_orig]
                
                # FK edge
                edges.append(SchemaEdge(
                    src=col1_node, dst=col2_node,
                    edge_type=EdgeType.FOREIGN_KEY,
                    relation='foreign_key'
                ))
                edges.append(SchemaEdge(
                    src=col2_node, dst=col1_node,
                    edge_type=EdgeType.FOREIGN_KEY_REV,
                    relation='foreign_key_rev'
                ))
                
                fk_pairs.append((col1_node, col2_node))
                
                # Table-table relationship via FK
                t1_idx = column_to_table_map.get(col1_node)
                t2_idx = column_to_table_map.get(col2_node)
                if t1_idx is not None and t2_idx is not None and t1_idx != t2_idx:
                    t1_node = table_indices[t1_idx]
                    t2_node = table_indices[t2_idx]
                    edges.append(SchemaEdge(
                        src=t1_node, dst=t2_node,
                        edge_type=EdgeType.TABLE_RELATED,
                        relation='table_related'
                    ))
                    edges.append(SchemaEdge(
                        src=t2_node, dst=t1_node,
                        edge_type=EdgeType.TABLE_RELATED,
                        relation='table_related'
                    ))
        
        # Self-loops
        for node in nodes:
            edges.append(SchemaEdge(
                src=node.idx, dst=node.idx,
                edge_type=EdgeType.SELF_LOOP,
                relation='self'
            ))
        
        # Compute node features
        node_features = self._compute_node_features(nodes, node_names)
        
        # Build tensors
        node_types = torch.LongTensor([n.node_type for n in nodes])
        
        edge_src = [e.src for e in edges]
        edge_dst = [e.dst for e in edges]
        edge_index = torch.LongTensor([edge_src, edge_dst])
        edge_types = torch.LongTensor([e.edge_type for e in edges])
        
        # Column to table mapping
        column_to_table = torch.zeros(len(column_indices), dtype=torch.long)
        for i, col_idx in enumerate(column_indices):
            col_node = nodes[col_idx]
            if col_node.table_idx is not None:
                column_to_table[i] = table_indices[col_node.table_idx]
        
        # Positional encoding
        pe = self._compute_positional_encoding(edge_index, len(nodes))
        
        return SchemaGraph(
            db_id=db_id,
            nodes=nodes,
            edges=edges,
            node_features=node_features,
            node_types=node_types,
            edge_index=edge_index,
            edge_types=edge_types,
            node_names=node_names,
            table_indices=table_indices,
            column_indices=column_indices,
            column_to_table=column_to_table,
            fk_pairs=fk_pairs,
            positional_encoding=pe
        )
    
    def _infer_semantic_type(self, col_name: str, col_type: str) -> str:
        """Infer semantic type of column."""
        name_lower = col_name.lower()
        type_lower = col_type.lower()
        
        # Identifier patterns
        if 'id' in name_lower or 'key' in name_lower or 'code' in name_lower:
            return 'identifier'
        
        # Temporal patterns
        if any(kw in name_lower for kw in ['date', 'time', 'year', 'month', 'day', 'created', 'updated']):
            return 'temporal'
        if type_lower in ('date', 'datetime', 'timestamp', 'time'):
            return 'temporal'
        
        # Numeric patterns
        if any(kw in name_lower for kw in ['age', 'count', 'amount', 'price', 'salary', 'score', 'rating', 'number']):
            return 'numeric'
        if type_lower in ('int', 'integer', 'real', 'float', 'decimal', 'number'):
            return 'numeric'
        
        return 'general'
    
    def _compute_node_features(
        self,
        nodes: List[SchemaNode],
        node_names: List[str]
    ) -> torch.Tensor:
        """Compute node feature embeddings."""
        # Get embeddings for all names
        texts = []
        for node in nodes:
            # Create descriptive text
            if node.node_type == 0:  # table
                text = f"table {node.original_name}"
            else:  # column
                markers = []
                if node.is_primary_key:
                    markers.append("primary key")
                if node.is_foreign_key:
                    markers.append("foreign key")
                marker_str = f" ({', '.join(markers)})" if markers else ""
                text = f"column {node.original_name} {node.col_type}{marker_str}"
            texts.append(text)
        
        embeddings = self.encoder.encode(texts)
        return torch.FloatTensor(embeddings)
    
    def _compute_positional_encoding(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """Compute random walk positional encoding."""
        # Build adjacency
        adj = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1
        adj = adj + torch.eye(num_nodes)
        
        # Transition matrix
        deg = adj.sum(dim=1).clamp(min=1)
        T = adj / deg.unsqueeze(1)
        
        # K-step walk
        pe = torch.zeros(num_nodes, self.pe_walk_length)
        T_k = torch.eye(num_nodes)
        
        for k in range(self.pe_walk_length):
            T_k = T_k @ T
            pe[:, k] = T_k.diag()
        
        return pe
    
    def compute_link_features(
        self,
        graph: SchemaGraph,
        question: str,
        db_values: Optional[Dict[str, Set[str]]] = None
    ) -> torch.Tensor:
        """Compute schema linking features for a question."""
        question_tokens = self.link_extractor.tokenize(question)
        num_nodes = len(graph.nodes)
        
        # Get question embedding for semantic similarity
        q_emb = self.encoder.encode_single(question)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        
        link_features = torch.zeros(num_nodes, 12)
        
        for node in graph.nodes:
            # Get node embedding for semantic similarity
            node_emb = graph.node_features[node.idx].numpy()
            node_emb = node_emb / (np.linalg.norm(node_emb) + 1e-8)
            semantic_sim = float(np.dot(q_emb, node_emb))
            
            # Get values if available
            values = None
            if db_values and node.node_type == 1:
                col_name = graph.node_names[node.idx]
                values = db_values.get(col_name)
            
            features = self.link_extractor.extract(
                node.original_name,
                question,
                question_tokens,
                node.col_type if node.node_type == 1 else 'text',
                semantic_sim,
                values
            )
            
            link_features[node.idx] = torch.FloatTensor(features)
        
        return link_features


# ═══════════════════════════════════════════════════════════════════════════════
# SQL PARSER FOR GROUND TRUTH EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

class RobustSQLParser:
    """Parse SQL to extract tables and columns."""
    
    def __init__(self, schema: Dict[str, List[str]]):
        self.schema = {k.lower(): [c.lower() for c in v] for k, v in schema.items()}
        self.tables = set(self.schema.keys())
        self.all_columns = {}
        for table, cols in self.schema.items():
            for col in cols:
                self.all_columns[col] = table
                self.all_columns[f"{table}.{col}"] = table
    
    def parse(self, sql: str) -> Tuple[Set[str], Set[str]]:
        """Parse SQL to get ground truth tables and columns."""
        sql_upper = sql.upper()
        sql_lower = sql.lower()
        
        found_tables = set()
        found_columns = set()
        
        # Find tables
        for table in self.tables:
            patterns = [
                rf'\bFROM\s+{table}\b',
                rf'\bJOIN\s+{table}\b',
                rf'\b{table}\s+AS\b',
                rf'\b{table}\s*,',
                rf',\s*{table}\b'
            ]
            for pattern in patterns:
                if re.search(pattern, sql_lower, re.IGNORECASE):
                    found_tables.add(table)
                    break
        
        # Extract aliases
        alias_map = {}
        alias_pattern = r'(\w+)\s+(?:AS\s+)?(\w+)'
        for match in re.finditer(alias_pattern, sql, re.IGNORECASE):
            potential_table = match.group(1).lower()
            alias = match.group(2).lower()
            if potential_table in self.tables:
                alias_map[alias] = potential_table
        
        # Find columns
        for table, cols in self.schema.items():
            for col in cols:
                # Direct reference
                if re.search(rf'\b{col}\b', sql_lower):
                    # Check if table is involved
                    if table in found_tables or not found_tables:
                        found_columns.add(f"{table}.{col}")
                
                # Qualified reference
                if re.search(rf'\b{table}\.{col}\b', sql_lower):
                    found_tables.add(table)
                    found_columns.add(f"{table}.{col}")
                
                # Alias reference
                for alias, real_table in alias_map.items():
                    if real_table == table and re.search(rf'\b{alias}\.{col}\b', sql_lower):
                        found_tables.add(table)
                        found_columns.add(f"{table}.{col}")
        
        # Special handling for * (star)
        if 'SELECT *' in sql_upper or 'SELECT T1.*' in sql_upper or 'SELECT T2.*' in sql_upper:
            for table in found_tables:
                for col in self.schema.get(table, []):
                    found_columns.add(f"{table}.{col}")
        
        return found_tables, found_columns


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedSchemaDataset(torch.utils.data.Dataset):
    """Dataset for training enhanced schema linking GNN."""
    
    def __init__(
        self,
        examples: List[Dict],
        schema_graphs: Dict[str, SchemaGraph],
        db_schemas: Dict[str, Dict],
        embedding_model,
        graph_builder: EnhancedSchemaGraphBuilder,
        max_question_tokens: int = 64
    ):
        self.examples = examples
        self.schema_graphs = schema_graphs
        self.encoder = embedding_model
        self.graph_builder = graph_builder
        self.max_tokens = max_question_tokens
        
        # Build SQL parsers
        self.parsers = {}
        for db_id, schema in db_schemas.items():
            self.parsers[db_id] = RobustSQLParser(schema)
        
        # Filter valid examples
        self._filter_valid_examples()
    
    def _filter_valid_examples(self):
        """Keep only examples with valid ground truth."""
        valid = []
        for ex in self.examples:
            db_id = ex['db_id']
            sql = ex.get('query', ex.get('sql', ''))
            if db_id in self.parsers and db_id in self.schema_graphs:
                tables, columns = self.parsers[db_id].parse(sql)
                if tables:
                    valid.append(ex)
        
        print(f"Filtered {len(self.examples)} -> {len(valid)} valid examples")
        self.examples = valid
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        db_id = example['db_id']
        question = example['question']
        sql = example.get('query', example.get('sql', ''))
        
        graph = self.schema_graphs[db_id]
        
        # Encode question
        q_emb = torch.FloatTensor(self.encoder.encode_single(question))
        
        # Encode question tokens
        tokens = self.graph_builder.link_extractor.tokenize(question)[:self.max_tokens]
        token_embs = [self.encoder.encode_single(t) for t in tokens]
        while len(token_embs) < self.max_tokens:
            token_embs.append(np.zeros(self.encoder.embedding_dim))
        q_tokens = torch.FloatTensor(np.array(token_embs))
        
        # Compute link features
        link_features = self.graph_builder.compute_link_features(graph, question)
        
        # Get ground truth
        gt_tables, gt_columns = self.parsers[db_id].parse(sql)
        
        # Create labels
        table_labels = torch.zeros(len(graph.table_indices))
        column_labels = torch.zeros(len(graph.column_indices))
        
        for i, t_idx in enumerate(graph.table_indices):
            t_name = graph.node_names[t_idx]
            if t_name in gt_tables:
                table_labels[i] = 1.0
        
        for i, c_idx in enumerate(graph.column_indices):
            c_name = graph.node_names[c_idx]
            if c_name in gt_columns:
                column_labels[i] = 1.0
        
        return {
            'db_id': db_id,
            'question': question,
            'question_embedding': q_emb,
            'question_tokens': q_tokens,
            'node_features': graph.node_features,
            'edge_index': graph.edge_index,
            'edge_types': graph.edge_types,
            'node_types': graph.node_types,
            'column_to_table': graph.column_to_table,
            'link_features': link_features,
            'positional_encoding': graph.positional_encoding,
            'table_labels': table_labels,
            'column_labels': column_labels,
            'table_indices': torch.LongTensor(graph.table_indices),
            'column_indices': torch.LongTensor(graph.column_indices),
            'node_names': graph.node_names,
            'fk_pairs': graph.fk_pairs
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for single-example batches."""
    return batch[0]


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingBackend:
    """Wrapper for sentence transformer embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.model.encode(text, show_progress_bar=False)


if __name__ == "__main__":
    # Test the graph builder
    print("Testing Enhanced Schema Graph Builder")
    print("=" * 60)
    
    # Mock database info
    mock_db = {
        'db_id': 'test_db',
        'table_names_original': ['students', 'courses', 'enrollments'],
        'column_names_original': [
            (-1, '*'),
            (0, 'student_id'), (0, 'name'), (0, 'age'),
            (1, 'course_id'), (1, 'title'), (1, 'credits'),
            (2, 'enrollment_id'), (2, 'student_id'), (2, 'course_id'), (2, 'grade')
        ],
        'column_types': ['', 'int', 'text', 'int', 'int', 'text', 'int', 'int', 'int', 'int', 'text'],
        'primary_keys': [1, 4, 7],
        'foreign_keys': [[8, 1], [9, 4]]
    }
    
    # Create embedding backend (mock)
    class MockEncoder:
        embedding_dim = 384
        def encode(self, texts):
            return np.random.randn(len(texts), self.embedding_dim)
        def encode_single(self, text):
            return np.random.randn(self.embedding_dim)
    
    encoder = MockEncoder()
    builder = EnhancedSchemaGraphBuilder(encoder)
    
    graph = builder.build_from_spider(mock_db)
    
    print(f"DB ID: {graph.db_id}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"  - Tables: {len(graph.table_indices)}")
    print(f"  - Columns: {len(graph.column_indices)}")
    print(f"Edges: {graph.edge_index.shape[1]}")
    print(f"Node features shape: {graph.node_features.shape}")
    print(f"FK pairs: {graph.fk_pairs}")
    
    # Test link features
    link_features = builder.compute_link_features(graph, "How many students are enrolled?")
    print(f"Link features shape: {link_features.shape}")
    
    print("\n✅ Graph builder test passed!")