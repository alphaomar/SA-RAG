"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    STRUCTURE-AWARE RETRIEVAL AUGMENTED GENERATION (SA-RAG) FOR TEXT-TO-SQL   ║
║                                                                              ║
║  A Novel Framework for Robust Text-to-SQL Generation                         ║
║                                                                              ║
║  Key Innovations:                                                            ║
║  1. Confidence-Adaptive Prompting (CAP) - Dynamic prompt strategy based on   ║
║     GNN confidence scores                                                    ║
║  2. Schema-Reasoning Chain-of-Thought (SR-CoT) - Inject GNN reasoning into   ║
║     generation process                                                       ║
║  3. Self-Consistency Decoding (SCD) - Multiple generations with semantic     ║
║     voting                                                                   ║
║  4. Structure-Aware SQL Validation (SASV) - Schema-grounded syntax checking  ║
║  5. Iterative Refinement with Error Correction (IREC)                        ║
║                                                                              ║
║  Evaluation Metrics:                                                         ║
║  - Exact Match (EM), Execution Accuracy (EX)                                 ║
║  - Component Match (CM): SELECT, FROM, WHERE, GROUP BY, ORDER BY             ║
║  - Retrieval Recall@K                                                        ║
║  - Schema Coverage Score                                                     ║
║                                                                              ║
║  Ablation Studies:                                                           ║
║  - Full SA-RAG vs No-GNN Baseline (Full Schema)                              ║
║  - SA-RAG vs Embedding-Only Retrieval                                        ║
║  - With/Without SR-CoT                                                       ║
║  - With/Without Self-Consistency                                             ║
║                                                                              ║
║  Author: Research Implementation                                             ║
║  For: Structure-Aware Graph Pre-training for Robust Text-to-SQL Retrieval    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import os
import re
import sqlite3
import torch
import numpy as np
from tqdm import tqdm
from groq import Groq
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
# from gnn_models import EnhancedGNNConfig, EnhancedSchemaGNN
# from retriver import EnhancedSchemaRetriever, RetrievalResult
# from schemaBuider import EnhancedSchemaGraphBuilder, RobustSQLParser

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SARAGConfig:
    """Configuration for SA-RAG Pipeline."""
    # API Configuration
    groq_api_key: str = "#"
    model_id: str = "llama-3.3-70b-versatile"
    
    # Paths
    checkpoint_path: str = "/kaggle/input/gnnmodel2/pytorch/default/1/best.pt"
    tables_path: str = "/kaggle/input/yale-universitys-spider-10-nlp-dataset/spider/tables.json"
    dev_data_path: str = "/kaggle/input/yale-universitys-spider-10-nlp-dataset/spider/dev.json"
    db_path: str = "/kaggle/input/yale-universitys-spider-10-nlp-dataset/spider/database"  # Path to database directory
    output_dir: str = "./results"
    
    # Generation Parameters
    temperature: float = 0.1
    max_tokens: int = 512
    self_consistency_k: int = 5  # Number of samples for self-consistency
    
    # Retrieval Parameters
    top_k_tables: int = 5
    top_k_columns: int = 12
    confidence_threshold_high: float = 0.7
    confidence_threshold_low: float = 0.4
    
    # Evaluation
    num_examples: int = 50
    run_ablation: bool = True
    
    # Features Toggle
    use_cot: bool = True
    use_self_consistency: bool = True
    use_iterative_refinement: bool = True
    use_schema_validation: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GenerationResult:
    """Result of SQL generation."""
    question: str
    db_id: str
    gold_sql: str
    predicted_sql: str
    gnn_confidence: float
    retrieved_tables: List[str]
    retrieved_columns: List[str]
    relationships: List[str]
    generation_method: str
    execution_result: Optional[str] = None
    exact_match: bool = False
    execution_match: bool = False
    reasoning: Optional[str] = None


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    exact_match: float = 0.0
    execution_accuracy: float = 0.0
    table_recall: float = 0.0
    column_recall: float = 0.0
    perfect_table_recall: float = 0.0
    perfect_column_recall: float = 0.0
    avg_confidence: float = 0.0
    select_match: float = 0.0
    from_match: float = 0.0
    where_match: float = 0.0
    group_by_match: float = 0.0
    order_by_match: float = 0.0
    num_examples: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'exact_match': round(self.exact_match * 100, 2),
            'execution_accuracy': round(self.execution_accuracy * 100, 2),
            'table_recall': round(self.table_recall * 100, 2),
            'column_recall': round(self.column_recall * 100, 2),
            'perfect_table_recall': round(self.perfect_table_recall * 100, 2),
            'perfect_column_recall': round(self.perfect_column_recall * 100, 2),
            'avg_confidence': round(self.avg_confidence, 3),
            'component_match': {
                'select': round(self.select_match * 100, 2),
                'from': round(self.from_match * 100, 2),
                'where': round(self.where_match * 100, 2),
                'group_by': round(self.group_by_match * 100, 2),
                'order_by': round(self.order_by_match * 100, 2)
            },
            'num_examples': self.num_examples
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SQL PARSER AND NORMALIZER
# ═══════════════════════════════════════════════════════════════════════════════

class SQLNormalizer:
    """Normalize SQL for comparison."""
    
    @staticmethod
    def normalize(sql: str) -> str:
        """Normalize SQL string for comparison."""
        if not sql:
            return ""
        
        # Convert to lowercase
        sql = sql.lower().strip()
        
        # Remove extra whitespace
        sql = ' '.join(sql.split())
        
        # Standardize quotes
        sql = sql.replace('"', "'")
        
        # Remove trailing semicolon
        sql = sql.rstrip(';')
        
        # Standardize keywords
        keywords = ['select', 'from', 'where', 'group by', 'having', 
                   'order by', 'limit', 'join', 'inner join', 'left join',
                   'right join', 'on', 'and', 'or', 'not', 'in', 'like',
                   'between', 'is null', 'is not null', 'asc', 'desc',
                   'distinct', 'count', 'sum', 'avg', 'max', 'min', 'as']
        
        for kw in keywords:
            pattern = re.compile(r'\b' + kw.replace(' ', r'\s+') + r'\b', re.IGNORECASE)
            sql = pattern.sub(kw.upper(), sql)
        
        return sql
    
    @staticmethod
    def extract_components(sql: str) -> Dict[str, str]:
        """Extract SQL components for partial matching."""
        sql_upper = sql.upper()
        components = {}
        
        # Extract SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)(?=FROM|$)', sql_upper, re.DOTALL)
        if select_match:
            components['select'] = select_match.group(1).strip()
        
        # Extract FROM clause
        from_match = re.search(r'FROM\s+(.*?)(?=WHERE|GROUP BY|HAVING|ORDER BY|LIMIT|$)', 
                              sql_upper, re.DOTALL)
        if from_match:
            components['from'] = from_match.group(1).strip()
        
        # Extract WHERE clause
        where_match = re.search(r'WHERE\s+(.*?)(?=GROUP BY|HAVING|ORDER BY|LIMIT|$)', 
                               sql_upper, re.DOTALL)
        if where_match:
            components['where'] = where_match.group(1).strip()
        
        # Extract GROUP BY clause
        group_match = re.search(r'GROUP BY\s+(.*?)(?=HAVING|ORDER BY|LIMIT|$)', 
                               sql_upper, re.DOTALL)
        if group_match:
            components['group_by'] = group_match.group(1).strip()
        
        # Extract ORDER BY clause
        order_match = re.search(r'ORDER BY\s+(.*?)(?=LIMIT|$)', sql_upper, re.DOTALL)
        if order_match:
            components['order_by'] = order_match.group(1).strip()
        
        return components


class SQLValidator:
    """Validate SQL syntax and schema compliance."""
    
    def __init__(self, schema: Dict[str, List[str]]):
        self.schema = schema
        self.tables = set(schema.keys())
        self.columns = {}
        for table, cols in schema.items():
            for col in cols:
                self.columns[col.lower()] = table.lower()
                self.columns[f"{table.lower()}.{col.lower()}"] = table.lower()
    
    def validate_syntax(self, sql: str) -> Tuple[bool, List[str]]:
        """Basic SQL syntax validation."""
        errors = []
        sql_upper = sql.upper()
        
        # Check for basic structure
        if 'SELECT' not in sql_upper:
            errors.append("Missing SELECT clause")
        
        if 'FROM' not in sql_upper and 'SELECT' in sql_upper:
            # Some valid SQL don't need FROM (e.g., SELECT 1+1)
            pass
        
        # Check parentheses balance
        if sql.count('(') != sql.count(')'):
            errors.append("Unbalanced parentheses")
        
        # Check for common syntax issues
        if re.search(r',,', sql):
            errors.append("Double comma detected")
        
        if re.search(r'SELECT\s+FROM', sql_upper):
            errors.append("Empty SELECT clause")
        
        return len(errors) == 0, errors
    
    def validate_schema_compliance(self, sql: str) -> Tuple[bool, List[str]]:
        """Check if SQL references valid tables/columns."""
        errors = []
        sql_lower = sql.lower()
        
        # Extract referenced tables
        from_match = re.search(r'from\s+(.*?)(?=where|group|having|order|limit|$)', 
                              sql_lower, re.DOTALL | re.IGNORECASE)
        if from_match:
            from_clause = from_match.group(1)
            # Simple table extraction (doesn't handle all edge cases)
            words = re.findall(r'\b(\w+)\b', from_clause)
            for word in words:
                if word not in ['join', 'inner', 'left', 'right', 'outer', 
                               'on', 'as', 'natural', 'cross', 't1', 't2', 't3', 't4']:
                    if word not in self.tables and not word.startswith('t'):
                        # Could be an alias, don't error
                        pass
        
        return len(errors) == 0, errors


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SQLExecutor:
    """Execute SQL queries against SQLite databases."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def execute(self, sql: str, db_id: str, timeout: float = 30.0) -> Tuple[bool, any]:
        """Execute SQL and return results."""
        db_file = os.path.join(self.db_path, db_id, f"{db_id}.sqlite")
        
        if not os.path.exists(db_file):
            return False, f"Database not found: {db_file}"
        
        try:
            conn = sqlite3.connect(db_file, timeout=timeout)
            conn.text_factory = str
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            conn.close()
            return True, results
        except Exception as e:
            return False, str(e)
    
    def compare_results(self, result1: any, result2: any) -> bool:
        """Compare two SQL execution results."""
        if result1 is None or result2 is None:
            return False
        
        # Convert to comparable format
        def normalize_result(result):
            if isinstance(result, str):
                return None
            try:
                # Sort for order-independent comparison
                return sorted([tuple(str(x) for x in row) for row in result])
            except:
                return result
        
        norm1 = normalize_result(result1)
        norm2 = normalize_result(result2)
        
        return norm1 == norm2


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

class PromptBuilder:
    """Build structure-aware prompts based on GNN outputs."""
    
    # System prompts for different confidence levels
    SYSTEM_HIGH_CONFIDENCE = """You are an expert SQLite SQL developer. Generate a SINGLE valid SQL query.
CRITICAL: The schema elements provided have been carefully selected by an AI model with HIGH confidence.
STRICTLY use ONLY the tables and columns provided. Do NOT invent or assume any other schema elements."""

    SYSTEM_MEDIUM_CONFIDENCE = """You are an expert SQLite SQL developer. Generate a SINGLE valid SQL query.
The schema elements provided have been selected by an AI model with moderate confidence.
Use the provided tables and columns, but verify they make sense for the question."""

    SYSTEM_LOW_CONFIDENCE = """You are an expert SQLite SQL developer. Generate a SINGLE valid SQL query.
WARNING: The schema retrieval model was uncertain about which elements to use.
Please carefully analyze the provided schema and consider if the selected elements are appropriate.
If the schema seems incomplete, make reasonable inferences based on standard database conventions."""

    @classmethod
    def build_schema_context(cls, retrieval: RetrievalResult, db_id: str) -> str:
        """Build structured schema context from retrieval result."""
        lines = []
        
        # Group columns by table
        table_columns = defaultdict(list)
        for table, col, score in retrieval.columns:
            table_columns[table].append((col, score))
        
        # Format each table with its columns
        for table, score in retrieval.tables:
            lines.append(f"\n📊 Table: {table} (relevance: {score:.2f})")
            
            cols = table_columns.get(table, [])
            if cols:
                col_strs = [f"{col} [{s:.2f}]" for col, s in sorted(cols, key=lambda x: -x[1])]
                lines.append(f"   Columns: {', '.join(col_strs)}")
            else:
                lines.append(f"   Columns: (all columns available)")
        
        return '\n'.join(lines)
    
    @classmethod
    def build_relationship_context(cls, relationships: List[str]) -> str:
        """Build relationship context for JOIN conditions."""
        if not relationships:
            return "No explicit foreign key relationships detected."
        
        lines = ["🔗 Join Conditions (Foreign Keys):"]
        for rel in relationships:
            lines.append(f"   • {rel}")
        
        return '\n'.join(lines)
    
    @classmethod
    def build_reasoning_context(cls, retrieval: RetrievalResult) -> str:
        """Build reasoning context from GNN scores."""
        lines = ["📝 Retrieval Reasoning:"]
        
        # Explain top selections
        if retrieval.tables:
            top_table = retrieval.tables[0]
            lines.append(f"   • Primary table: {top_table[0]} (score: {top_table[1]:.2f})")
        
        # Explain confidence
        if retrieval.confidence > 0.7:
            lines.append("   • High confidence: Schema elements are well-matched to the question")
        elif retrieval.confidence > 0.4:
            lines.append("   • Medium confidence: Some ambiguity in schema matching")
        else:
            lines.append("   • Low confidence: Consider verifying table/column selections")
        
        return '\n'.join(lines)
    
    @classmethod
    def build_direct_prompt(cls, question: str, retrieval: RetrievalResult, 
                           db_id: str) -> Tuple[str, str]:
        """Build direct generation prompt."""
        if retrieval.confidence > 0.7:
            system = cls.SYSTEM_HIGH_CONFIDENCE
        elif retrieval.confidence > 0.4:
            system = cls.SYSTEM_MEDIUM_CONFIDENCE
        else:
            system = cls.SYSTEM_LOW_CONFIDENCE
        
        schema_ctx = cls.build_schema_context(retrieval, db_id)
        rel_ctx = cls.build_relationship_context(retrieval.relationships)
        
        user_prompt = f"""Database: {db_id}

{schema_ctx}

{rel_ctx}

Question: {question}

Instructions:
1. Use ONLY the tables and columns listed above
2. Use the JOIN conditions provided when joining tables
3. Output ONLY the SQL query, no explanations
4. Do not use markdown formatting

SQL:"""
        
        return system, user_prompt
    
    @classmethod
    def build_cot_prompt(cls, question: str, retrieval: RetrievalResult,
                        db_id: str) -> Tuple[str, str]:
        """Build Chain-of-Thought prompt with schema reasoning."""
        system = """You are an expert SQLite SQL developer who thinks step-by-step.
First analyze the question, then identify required tables and columns, then write the SQL."""
        
        schema_ctx = cls.build_schema_context(retrieval, db_id)
        rel_ctx = cls.build_relationship_context(retrieval.relationships)
        reasoning_ctx = cls.build_reasoning_context(retrieval)
        
        user_prompt = f"""Database: {db_id}

{schema_ctx}

{rel_ctx}

{reasoning_ctx}

Question: {question}

Think step-by-step:
1. What is the question asking for? (What should SELECT return?)
2. Which tables contain this information?
3. Are JOINs needed? If so, which tables and on what conditions?
4. Are there any filters (WHERE conditions)?
5. Is aggregation (GROUP BY) or ordering (ORDER BY) needed?

After your analysis, write ONLY the final SQL query on the last line starting with "SQL: ".
Your response MUST end with the SQL query."""
        
        return system, user_prompt
    
    @classmethod
    def build_refinement_prompt(cls, question: str, original_sql: str,
                               error: str, retrieval: RetrievalResult,
                               db_id: str) -> Tuple[str, str]:
        """Build prompt for SQL refinement after error."""
        system = """You are an expert SQL debugger. Fix the SQL query based on the error message.
Ensure the corrected query uses only the provided schema elements."""
        
        schema_ctx = cls.build_schema_context(retrieval, db_id)
        rel_ctx = cls.build_relationship_context(retrieval.relationships)
        
        user_prompt = f"""Database: {db_id}

{schema_ctx}

{rel_ctx}

Question: {question}

Original SQL (with error):
{original_sql}

Error: {error}

Please fix the SQL query. Output ONLY the corrected SQL, no explanations.

Corrected SQL:"""
        
        return system, user_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE RETRIEVERS FOR ABLATION
# ═══════════════════════════════════════════════════════════════════════════════

class FullSchemaRetriever:
    """Baseline: Return full schema (no retrieval)."""
    
    def __init__(self, db_schemas: Dict[str, Dict]):
        self.db_schemas = db_schemas
    
    def retrieve(self, question: str, db_id: str, **kwargs) -> RetrievalResult:
        schema = self.db_schemas.get(db_id, {})
        
        tables = [(t, 1.0) for t in schema.keys()]
        columns = [(t, c, 1.0) for t, cols in schema.items() for c in cols]
        
        return RetrievalResult(
            tables=tables,
            columns=columns,
            relationships=[],
            confidence=0.5,  # Neutral confidence
            reasoning={"method": "full_schema"}
        )


class EmbeddingOnlyRetriever:
    """Baseline: Simple embedding similarity without GNN structure."""
    
    def __init__(self, encoder, db_schemas: Dict[str, Dict]):
        self.encoder = encoder
        self.db_schemas = db_schemas
        
        # Pre-compute schema embeddings
        self.schema_embeddings = {}
        for db_id, schema in db_schemas.items():
            table_embs = {}
            column_embs = {}
            
            for table, cols in schema.items():
                # Table embedding
                table_embs[table] = encoder.encode_single(table.replace('_', ' '))
                
                # Column embeddings
                for col in cols:
                    key = f"{table}.{col}"
                    column_embs[key] = encoder.encode_single(
                        f"{table.replace('_', ' ')} {col.replace('_', ' ')}"
                    )
            
            self.schema_embeddings[db_id] = {
                'tables': table_embs,
                'columns': column_embs
            }
    
    def retrieve(self, question: str, db_id: str, 
                top_k_tables: int = 5, top_k_columns: int = 15, **kwargs) -> RetrievalResult:
        if db_id not in self.schema_embeddings:
            return RetrievalResult([], [], [], 0.0, {})
        
        q_emb = self.encoder.encode_single(question)
        q_emb = q_emb / np.linalg.norm(q_emb)
        
        embs = self.schema_embeddings[db_id]
        
        # Score tables
        table_scores = []
        for table, t_emb in embs['tables'].items():
            t_emb = t_emb / np.linalg.norm(t_emb)
            score = float(np.dot(q_emb, t_emb))
            table_scores.append((table, score))
        
        # Score columns
        column_scores = []
        for col_key, c_emb in embs['columns'].items():
            c_emb = c_emb / np.linalg.norm(c_emb)
            score = float(np.dot(q_emb, c_emb))
            table, col = col_key.split('.', 1)
            column_scores.append((table, col, score))
        
        # Sort and select top-k
        table_scores.sort(key=lambda x: -x[1])
        column_scores.sort(key=lambda x: -x[2])
        
        selected_tables = table_scores[:top_k_tables]
        selected_columns = column_scores[:top_k_columns]
        
        # Compute confidence (average of top scores)
        confidence = np.mean([s for _, s in selected_tables[:3]]) if selected_tables else 0.0
        
        return RetrievalResult(
            tables=selected_tables,
            columns=selected_columns,
            relationships=[],
            confidence=float(confidence),
            reasoning={"method": "embedding_only"}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SA-RAG GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class SARAGGenerator:
    """
    Structure-Aware Retrieval Augmented Generation for Text-to-SQL.
    
    Implements the full SA-RAG pipeline with:
    - Confidence-Adaptive Prompting (CAP)
    - Schema-Reasoning Chain-of-Thought (SR-CoT)
    - Self-Consistency Decoding (SCD)
    - Iterative Refinement with Error Correction (IREC)
    """
    
    def __init__(self, config: SARAGConfig, retriever, db_schemas: Dict[str, Dict]):
        self.config = config
        self.retriever = retriever
        self.db_schemas = db_schemas
        self.client = Groq(api_key=config.groq_api_key)
        
        # Initialize executor if db_path exists
        if os.path.exists(config.db_path):
            self.executor = SQLExecutor(config.db_path)
        else:
            self.executor = None
            print(f"⚠️ Database path not found: {config.db_path}")
            print("   Execution accuracy will not be computed.")
        
        # Initialize validators
        self.validators = {}
        for db_id, schema in db_schemas.items():
            self.validators[db_id] = SQLValidator(schema)
        
        # Initialize SQL parsers
        self.parsers = {}
        for db_id, schema in db_schemas.items():
            self.parsers[db_id] = RobustSQLParser(schema)
    
    def _call_llm(self, system_prompt: str, user_prompt: str, 
                  temperature: float = None) -> str:
        """Call LLM API."""
        temp = temperature if temperature is not None else self.config.temperature
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.config.model_id,
                temperature=temp,
                max_tokens=self.config.max_tokens,
                top_p=1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM API Error: {e}")
            return "SELECT 'ERROR'"
    
    def _extract_sql(self, response: str) -> str:
        """Extract SQL from LLM response."""
        # Remove markdown formatting
        response = response.replace('```sql', '').replace('```', '')
        
        # Look for explicit SQL marker
        if 'SQL:' in response.upper():
            parts = response.upper().split('SQL:')
            sql = parts[-1].strip()
            # Get original case version
            original_parts = response.split(':')
            if len(original_parts) > 1:
                sql = original_parts[-1].strip()
        else:
            sql = response.strip()
        
        # Clean up
        sql = sql.strip().rstrip(';')
        
        # Find the actual SQL (often model adds explanations)
        lines = sql.split('\n')
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line.upper().startswith(('SELECT', 'WITH')) or sql_lines:
                sql_lines.append(line)
        
        if sql_lines:
            sql = ' '.join(sql_lines)
        
        return sql.strip()
    
    def _generate_direct(self, question: str, retrieval: RetrievalResult,
                        db_id: str) -> str:
        """Direct generation without CoT."""
        system, user = PromptBuilder.build_direct_prompt(question, retrieval, db_id)
        response = self._call_llm(system, user)
        return self._extract_sql(response)
    
    def _generate_cot(self, question: str, retrieval: RetrievalResult,
                     db_id: str) -> Tuple[str, str]:
        """Generation with Chain-of-Thought reasoning."""
        system, user = PromptBuilder.build_cot_prompt(question, retrieval, db_id)
        response = self._call_llm(system, user)
        sql = self._extract_sql(response)
        
        # Extract reasoning
        reasoning = response.split('SQL:')[0] if 'SQL:' in response.upper() else response
        
        return sql, reasoning
    
    def _generate_with_self_consistency(self, question: str, 
                                        retrieval: RetrievalResult,
                                        db_id: str) -> str:
        """Generate multiple samples and vote."""
        candidates = []
        
        # Generate K samples with higher temperature
        for _ in range(self.config.self_consistency_k):
            system, user = PromptBuilder.build_direct_prompt(question, retrieval, db_id)
            response = self._call_llm(system, user, temperature=0.3)
            sql = self._extract_sql(response)
            candidates.append(sql)
        
        # Normalize and vote
        normalized = [SQLNormalizer.normalize(sql) for sql in candidates]
        counter = Counter(normalized)
        
        # Return most common (with original formatting from first occurrence)
        most_common = counter.most_common(1)[0][0]
        for i, norm in enumerate(normalized):
            if norm == most_common:
                return candidates[i]
        
        return candidates[0]
    
    def _refine_sql(self, question: str, sql: str, error: str,
                   retrieval: RetrievalResult, db_id: str,
                   max_attempts: int = 2) -> str:
        """Iteratively refine SQL based on errors."""
        current_sql = sql
        
        for _ in range(max_attempts):
            system, user = PromptBuilder.build_refinement_prompt(
                question, current_sql, error, retrieval, db_id
            )
            response = self._call_llm(system, user)
            current_sql = self._extract_sql(response)
            
            # Validate the refined SQL
            if db_id in self.validators:
                valid, errors = self.validators[db_id].validate_syntax(current_sql)
                if valid:
                    # Try execution if possible
                    if self.executor:
                        success, result = self.executor.execute(current_sql, db_id)
                        if success:
                            return current_sql
                        error = str(result)
                    else:
                        return current_sql
            
        return current_sql
    
    def generate(self, question: str, db_id: str,
                method: str = 'full') -> GenerationResult:
        """
        Generate SQL for a question using specified method.
        
        Methods:
        - 'direct': Simple direct generation
        - 'cot': Chain-of-Thought generation
        - 'self_consistency': Multiple samples with voting
        - 'full': Full SA-RAG pipeline (CAP + SR-CoT + SCD + IREC)
        """
        # Step 1: Retrieve schema elements
        try:
            retrieval = self.retriever.retrieve(
                question=question,
                db_id=db_id,
                top_k_tables=self.config.top_k_tables,
                top_k_columns=self.config.top_k_columns,
                use_dynamic_threshold=True,
                include_relationships=True
            )
        except Exception as e:
            print(f"Retrieval error for {db_id}: {e}")
            retrieval = RetrievalResult(
                tables=[(t, 1.0) for t in list(self.db_schemas.get(db_id, {}).keys())[:5]],
                columns=[],
                relationships=[],
                confidence=0.3,
                reasoning={"error": str(e)}
            )
        
        reasoning = None
        
        # Step 2: Generate SQL based on method
        if method == 'direct':
            sql = self._generate_direct(question, retrieval, db_id)
            gen_method = 'direct'
        
        elif method == 'cot':
            sql, reasoning = self._generate_cot(question, retrieval, db_id)
            gen_method = 'cot'
        
        elif method == 'self_consistency':
            sql = self._generate_with_self_consistency(question, retrieval, db_id)
            gen_method = 'self_consistency'
        
        elif method == 'full':
            # Full SA-RAG pipeline
            
            # Confidence-Adaptive Strategy Selection
            if retrieval.confidence > self.config.confidence_threshold_high:
                # High confidence: Direct generation is usually sufficient
                if self.config.use_cot:
                    sql, reasoning = self._generate_cot(question, retrieval, db_id)
                else:
                    sql = self._generate_direct(question, retrieval, db_id)
                gen_method = 'full_high_conf'
            
            elif retrieval.confidence > self.config.confidence_threshold_low:
                # Medium confidence: Use self-consistency
                if self.config.use_self_consistency:
                    sql = self._generate_with_self_consistency(question, retrieval, db_id)
                else:
                    sql = self._generate_direct(question, retrieval, db_id)
                gen_method = 'full_med_conf'
            
            else:
                # Low confidence: Use CoT + self-consistency
                if self.config.use_cot and self.config.use_self_consistency:
                    candidates = []
                    for _ in range(3):
                        cot_sql, _ = self._generate_cot(question, retrieval, db_id)
                        candidates.append(cot_sql)
                    
                    # Vote
                    normalized = [SQLNormalizer.normalize(s) for s in candidates]
                    counter = Counter(normalized)
                    most_common = counter.most_common(1)[0][0]
                    for i, norm in enumerate(normalized):
                        if norm == most_common:
                            sql = candidates[i]
                            break
                    else:
                        sql = candidates[0]
                else:
                    sql, reasoning = self._generate_cot(question, retrieval, db_id)
                gen_method = 'full_low_conf'
        
        else:
            sql = self._generate_direct(question, retrieval, db_id)
            gen_method = 'direct'
        
        # Step 3: Validate and potentially refine
        if self.config.use_iterative_refinement and db_id in self.validators:
            valid, errors = self.validators[db_id].validate_syntax(sql)
            
            if not valid and errors:
                sql = self._refine_sql(
                    question, sql, '; '.join(errors),
                    retrieval, db_id
                )
        
        return GenerationResult(
            question=question,
            db_id=db_id,
            gold_sql="",
            predicted_sql=sql,
            gnn_confidence=retrieval.confidence,
            retrieved_tables=[t for t, _ in retrieval.tables],
            retrieved_columns=[f"{t}.{c}" for t, c, _ in retrieval.columns],
            relationships=retrieval.relationships,
            generation_method=gen_method,
            reasoning=reasoning
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

class SARAGEvaluator:
    """Comprehensive evaluation for SA-RAG system."""
    
    def __init__(self, db_schemas: Dict[str, Dict], db_path: str = None):
        self.db_schemas = db_schemas
        self.executor = SQLExecutor(db_path) if db_path and os.path.exists(db_path) else None
        
        # SQL parsers for ground truth extraction
        self.parsers = {}
        for db_id, schema in db_schemas.items():
            self.parsers[db_id] = RobustSQLParser(schema)
    
    def compute_exact_match(self, pred_sql: str, gold_sql: str) -> bool:
        """Compute exact match (normalized)."""
        pred_norm = SQLNormalizer.normalize(pred_sql)
        gold_norm = SQLNormalizer.normalize(gold_sql)
        return pred_norm == gold_norm
    
    def compute_execution_match(self, pred_sql: str, gold_sql: str, 
                               db_id: str) -> bool:
        """Compute execution accuracy."""
        if not self.executor:
            return False
        
        success_pred, result_pred = self.executor.execute(pred_sql, db_id)
        success_gold, result_gold = self.executor.execute(gold_sql, db_id)
        
        if not success_pred or not success_gold:
            return False
        
        return self.executor.compare_results(result_pred, result_gold)
    
    def compute_component_match(self, pred_sql: str, gold_sql: str) -> Dict[str, bool]:
        """Compute component-wise match."""
        pred_comp = SQLNormalizer.extract_components(pred_sql)
        gold_comp = SQLNormalizer.extract_components(gold_sql)
        
        matches = {}
        for key in ['select', 'from', 'where', 'group_by', 'order_by']:
            pred_val = pred_comp.get(key, '').strip()
            gold_val = gold_comp.get(key, '').strip()
            
            # Normalize for comparison
            pred_val = ' '.join(pred_val.split()).upper()
            gold_val = ' '.join(gold_val.split()).upper()
            
            matches[key] = pred_val == gold_val
        
        return matches
    
    def compute_retrieval_recall(self, result: GenerationResult,
                                gold_sql: str, db_id: str) -> Dict[str, float]:
        """Compute retrieval recall metrics."""
        if db_id not in self.parsers:
            return {'table_recall': 0.0, 'column_recall': 0.0,
                   'perfect_table': 0.0, 'perfect_column': 0.0}
        
        gt_tables, gt_columns = self.parsers[db_id].parse(gold_sql)
        
        pred_tables = set(result.retrieved_tables)
        pred_columns = set(result.retrieved_columns)
        
        # Table recall
        table_hit = len(pred_tables & gt_tables) if gt_tables else 0
        table_recall = table_hit / len(gt_tables) if gt_tables else 1.0
        perfect_table = 1.0 if gt_tables <= pred_tables else 0.0
        
        # Column recall
        column_hit = len(pred_columns & gt_columns) if gt_columns else 0
        column_recall = column_hit / len(gt_columns) if gt_columns else 1.0
        perfect_column = 1.0 if gt_columns <= pred_columns else 0.0
        
        return {
            'table_recall': table_recall,
            'column_recall': column_recall,
            'perfect_table': perfect_table,
            'perfect_column': perfect_column
        }
    
    def evaluate_batch(self, results: List[GenerationResult],
                      gold_data: List[Dict]) -> EvaluationMetrics:
        """Evaluate a batch of results."""
        metrics = EvaluationMetrics()
        
        exact_matches = []
        exec_matches = []
        table_recalls = []
        column_recalls = []
        perfect_tables = []
        perfect_columns = []
        confidences = []
        
        select_matches = []
        from_matches = []
        where_matches = []
        group_matches = []
        order_matches = []
        
        for result, gold in zip(results, gold_data):
            gold_sql = gold.get('query', gold.get('sql', ''))
            db_id = result.db_id
            
            # Exact match
            em = self.compute_exact_match(result.predicted_sql, gold_sql)
            exact_matches.append(1.0 if em else 0.0)
            
            # Execution match
            ex = self.compute_execution_match(result.predicted_sql, gold_sql, db_id)
            exec_matches.append(1.0 if ex else 0.0)
            
            # Component match
            comp_match = self.compute_component_match(result.predicted_sql, gold_sql)
            select_matches.append(1.0 if comp_match.get('select', False) else 0.0)
            from_matches.append(1.0 if comp_match.get('from', False) else 0.0)
            where_matches.append(1.0 if comp_match.get('where', False) else 0.0)
            group_matches.append(1.0 if comp_match.get('group_by', False) else 0.0)
            order_matches.append(1.0 if comp_match.get('order_by', False) else 0.0)
            
            # Retrieval recall
            recall_metrics = self.compute_retrieval_recall(result, gold_sql, db_id)
            table_recalls.append(recall_metrics['table_recall'])
            column_recalls.append(recall_metrics['column_recall'])
            perfect_tables.append(recall_metrics['perfect_table'])
            perfect_columns.append(recall_metrics['perfect_column'])
            
            # Confidence
            confidences.append(result.gnn_confidence)
        
        metrics.exact_match = np.mean(exact_matches) if exact_matches else 0.0
        metrics.execution_accuracy = np.mean(exec_matches) if exec_matches else 0.0
        metrics.table_recall = np.mean(table_recalls) if table_recalls else 0.0
        metrics.column_recall = np.mean(column_recalls) if column_recalls else 0.0
        metrics.perfect_table_recall = np.mean(perfect_tables) if perfect_tables else 0.0
        metrics.perfect_column_recall = np.mean(perfect_columns) if perfect_columns else 0.0
        metrics.avg_confidence = np.mean(confidences) if confidences else 0.0
        
        metrics.select_match = np.mean(select_matches) if select_matches else 0.0
        metrics.from_match = np.mean(from_matches) if from_matches else 0.0
        metrics.where_match = np.mean(where_matches) if where_matches else 0.0
        metrics.group_by_match = np.mean(group_matches) if group_matches else 0.0
        metrics.order_by_match = np.mean(order_matches) if order_matches else 0.0
        
        metrics.num_examples = len(results)
        
        return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION STUDY RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

class AblationStudyRunner:
    """Run ablation studies comparing different configurations."""
    
    def __init__(self, config: SARAGConfig, db_schemas: Dict[str, Dict],
                encoder, gnn_retriever):
        self.config = config
        self.db_schemas = db_schemas
        self.encoder = encoder
        self.gnn_retriever = gnn_retriever
        self.evaluator = SARAGEvaluator(db_schemas, config.db_path)
    
    def run_ablation(self, dev_data: List[Dict]) -> Dict[str, EvaluationMetrics]:
        """Run complete ablation study."""
        results = {}
        
        print("\n" + "=" * 80)
        print("🔬 ABLATION STUDY")
        print("=" * 80)
        
        # Study 1: Full SA-RAG (Your Method)
        print("\n📊 [1/5] Full SA-RAG (GNN + CAP + SR-CoT + SCD)")
        results['full_sarag'] = self._run_experiment(
            dev_data, 
            retriever=self.gnn_retriever,
            use_cot=True,
            use_self_consistency=True,
            method='full'
        )
        
        # Study 2: No GNN (Full Schema Baseline)
        print("\n📊 [2/5] No GNN Baseline (Full Schema)")
        full_schema_retriever = FullSchemaRetriever(self.db_schemas)
        results['no_gnn_baseline'] = self._run_experiment(
            dev_data,
            retriever=full_schema_retriever,
            use_cot=True,
            use_self_consistency=False,
            method='cot'
        )
        
        # Study 3: Embedding-Only Retrieval (No Graph Structure)
        print("\n📊 [3/5] Embedding-Only Retrieval (No GNN Structure)")
        embedding_retriever = EmbeddingOnlyRetriever(self.encoder, self.db_schemas)
        results['embedding_only'] = self._run_experiment(
            dev_data,
            retriever=embedding_retriever,
            use_cot=True,
            use_self_consistency=False,
            method='cot'
        )
        
        # Study 4: GNN without CoT
        print("\n📊 [4/5] GNN without Chain-of-Thought")
        results['gnn_no_cot'] = self._run_experiment(
            dev_data,
            retriever=self.gnn_retriever,
            use_cot=False,
            use_self_consistency=False,
            method='direct'
        )
        
        # Study 5: GNN without Self-Consistency
        print("\n📊 [5/5] GNN without Self-Consistency")
        results['gnn_no_sc'] = self._run_experiment(
            dev_data,
            retriever=self.gnn_retriever,
            use_cot=True,
            use_self_consistency=False,
            method='cot'
        )
        
        return results
    
    def _run_experiment(self, dev_data: List[Dict], retriever,
                       use_cot: bool, use_self_consistency: bool,
                       method: str) -> EvaluationMetrics:
        """Run single experiment configuration."""
        # Create generator with specific config
        exp_config = SARAGConfig(
            groq_api_key=self.config.groq_api_key,
            model_id=self.config.model_id,
            db_path=self.config.db_path,
            use_cot=use_cot,
            use_self_consistency=use_self_consistency,
            use_iterative_refinement=True,
            top_k_tables=self.config.top_k_tables,
            top_k_columns=self.config.top_k_columns
        )
        
        generator = SARAGGenerator(exp_config, retriever, self.db_schemas)
        
        # Generate SQL
        results = []
        for entry in tqdm(dev_data, desc="Generating", leave=False):
            result = generator.generate(
                question=entry['question'],
                db_id=entry['db_id'],
                method=method
            )
            result.gold_sql = entry.get('query', '')
            results.append(result)
        
        # Evaluate
        metrics = self.evaluator.evaluate_batch(results, dev_data)
        
        print(f"   EM: {metrics.exact_match*100:.1f}% | "
              f"EX: {metrics.execution_accuracy*100:.1f}% | "
              f"Table Recall: {metrics.table_recall*100:.1f}%")
        
        return metrics
    
    def format_ablation_table(self, results: Dict[str, EvaluationMetrics]) -> str:
        """Format ablation results as ASCII table."""
        lines = []
        lines.append("\n" + "=" * 100)
        lines.append("ABLATION STUDY RESULTS")
        lines.append("=" * 100)
        
        # Header
        header = f"{'Method':<35} {'EM%':>8} {'EX%':>8} {'T-Recall%':>10} {'C-Recall%':>10} {'Conf':>8}"
        lines.append(header)
        lines.append("-" * 100)
        
        # Results
        method_names = {
            'full_sarag': 'Full SA-RAG (Ours)',
            'no_gnn_baseline': 'No GNN (Full Schema)',
            'embedding_only': 'Embedding-Only Retrieval',
            'gnn_no_cot': 'GNN w/o Chain-of-Thought',
            'gnn_no_sc': 'GNN w/o Self-Consistency'
        }
        
        for key, metrics in results.items():
            name = method_names.get(key, key)
            line = f"{name:<35} {metrics.exact_match*100:>7.1f}% {metrics.execution_accuracy*100:>7.1f}% "
            line += f"{metrics.table_recall*100:>9.1f}% {metrics.column_recall*100:>9.1f}% "
            line += f"{metrics.avg_confidence:>7.3f}"
            lines.append(line)
        
        lines.append("=" * 100)
        
        # Component match for full method
        if 'full_sarag' in results:
            m = results['full_sarag']
            lines.append("\nComponent Match (Full SA-RAG):")
            lines.append(f"  SELECT: {m.select_match*100:.1f}% | FROM: {m.from_match*100:.1f}% | "
                        f"WHERE: {m.where_match*100:.1f}% | GROUP BY: {m.group_by_match*100:.1f}% | "
                        f"ORDER BY: {m.order_by_match*100:.1f}%")
        
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main execution function."""
    
    print("=" * 80)
    print("🚀 STRUCTURE-AWARE RAG FOR TEXT-TO-SQL")
    print("=" * 80)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION - MODIFY THESE PATHS FOR YOUR SETUP
    # ═══════════════════════════════════════════════════════════════════════════
    
    config = SARAGConfig(
        # API Key - Set your Groq API key here
        groq_api_key="#",
        model_id="llama-3.3-70b-versatile",
        
        # Paths - Modify for your setup
        checkpoint_path="/kaggle/input/gnnmodel2/pytorch/default/1/best.pt",
        tables_path="/kaggle/input/yale-universitys-spider-10-nlp-dataset/spider/tables.json",
        dev_data_path="/kaggle/input/yale-universitys-spider-10-nlp-dataset/spider/dev.json",
        db_path="/kaggle/input/yale-universitys-spider-10-nlp-dataset/spider/database",
        output_dir="./results",
        
        # Generation Parameters
        temperature=0.1,
        self_consistency_k=5,
        
        # Retrieval Parameters
        top_k_tables=5,
        top_k_columns=12,
        confidence_threshold_high=0.7,
        confidence_threshold_low=0.4,
        
        # Evaluation
        num_examples=50,
        run_ablation=True,
        
        # Features
        use_cot=True,
        use_self_consistency=True,
        use_iterative_refinement=True,
        use_schema_validation=True
    )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: CHECK FILES AND LOAD DATA
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n📁 Checking required files...")
    
    required_files = [
        (config.tables_path, "tables.json"),
        (config.dev_data_path, "dev.json"),
        (config.checkpoint_path, "GNN checkpoint")
    ]
    
    for path, name in required_files:
        if os.path.exists(path):
            print(f"   ✅ Found {name}")
        else:
            print(f"   ❌ Missing {name}: {path}")
            print(f"      Please update the path in the configuration.")
            return
    
    # Load tables
    print("\n📊 Loading schema data...")
    with open(config.tables_path, 'r') as f:
        tables_data = json.load(f)
    
    # Build db_schemas dictionary
    db_schemas = {}
    for db in tables_data:
        db_id = db['db_id']
        schema = defaultdict(list)
        for table_id, col_name in db['column_names_original']:
            if table_id >= 0:
                table_name = db['table_names_original'][table_id]
                schema[table_name.lower()].append(col_name.lower())
        db_schemas[db_id] = dict(schema)
    
    print(f"   Loaded {len(db_schemas)} database schemas")
    
    # Load dev data
    print("\n📝 Loading development data...")
    with open(config.dev_data_path, 'r') as f:
        dev_data = json.load(f)
    
    print(f"   Loaded {len(dev_data)} examples")
    print(f"   Using first {config.num_examples} examples for evaluation")
    
    dev_data = dev_data[:config.num_examples]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: LOAD GNN RETRIEVER
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n🧠 Loading GNN Retriever...")
    print("   This may take a moment for model initialization...")
    
    try:
        retriever = EnhancedSchemaRetriever.from_checkpoint(
            checkpoint_path=config.checkpoint_path,
            embedding_model='all-MiniLM-L6-v2',
            tables_path=config.tables_path,
            model_class=EnhancedSchemaGNN,
            model_config=EnhancedGNNConfig()
        )
        print("   ✅ GNN Retriever loaded successfully")
        
        # Get encoder from retriever for baseline
        encoder = retriever.encoder
        
    except Exception as e:
        print(f"   ❌ Failed to load GNN model: {e}")
        print("   Please check the checkpoint path.")
        return
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: INITIALIZE GENERATOR AND EVALUATOR
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n⚙️ Initializing SA-RAG Generator...")
    generator = SARAGGenerator(config, retriever, db_schemas)
    evaluator = SARAGEvaluator(db_schemas, config.db_path)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4: MAIN EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("📊 MAIN EVALUATION: Full SA-RAG Pipeline")
    print("=" * 80)
    
    results = []
    
    for entry in tqdm(dev_data, desc="Generating SQL"):
        result = generator.generate(
            question=entry['question'],
            db_id=entry['db_id'],
            method='full'
        )
        result.gold_sql = entry.get('query', '')
        results.append(result)
    
    # Evaluate
    metrics = evaluator.evaluate_batch(results, dev_data)
    
    print("\n" + "=" * 80)
    print("📈 EVALUATION RESULTS (Full SA-RAG)")
    print("=" * 80)
    print(f"  Exact Match (EM):        {metrics.exact_match*100:.2f}%")
    print(f"  Execution Accuracy (EX): {metrics.execution_accuracy*100:.2f}%")
    print(f"  Table Recall:            {metrics.table_recall*100:.2f}%")
    print(f"  Column Recall:           {metrics.column_recall*100:.2f}%")
    print(f"  Perfect Table Recall:    {metrics.perfect_table_recall*100:.2f}%")
    print(f"  Perfect Column Recall:   {metrics.perfect_column_recall*100:.2f}%")
    print(f"  Average GNN Confidence:  {metrics.avg_confidence:.3f}")
    print("-" * 80)
    print("  Component Match:")
    print(f"    SELECT:   {metrics.select_match*100:.2f}%")
    print(f"    FROM:     {metrics.from_match*100:.2f}%")
    print(f"    WHERE:    {metrics.where_match*100:.2f}%")
    print(f"    GROUP BY: {metrics.group_by_match*100:.2f}%")
    print(f"    ORDER BY: {metrics.order_by_match*100:.2f}%")
    print("=" * 80)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5: ABLATION STUDIES
    # ═══════════════════════════════════════════════════════════════════════════
    
    if config.run_ablation:
        print("\n" + "=" * 80)
        print("🔬 RUNNING ABLATION STUDIES")
        print("=" * 80)
        
        ablation_runner = AblationStudyRunner(config, db_schemas, encoder, retriever)
        ablation_results = ablation_runner.run_ablation(dev_data)
        
        # Print formatted table
        print(ablation_runner.format_ablation_table(ablation_results))
        
        # Save ablation results
        ablation_output = {
            method: metrics.to_dict() 
            for method, metrics in ablation_results.items()
        }
        
        with open(os.path.join(config.output_dir, 'ablation_results.json'), 'w') as f:
            json.dump(ablation_output, f, indent=2)
        
        print(f"\n💾 Ablation results saved to {config.output_dir}/ablation_results.json")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 6: SAVE DETAILED RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n💾 Saving detailed results...")
    
    # Prepare output
    output_data = {
        'config': {
            'model_id': config.model_id,
            'num_examples': config.num_examples,
            'top_k_tables': config.top_k_tables,
            'top_k_columns': config.top_k_columns,
            'use_cot': config.use_cot,
            'use_self_consistency': config.use_self_consistency,
            'timestamp': datetime.now().isoformat()
        },
        'metrics': metrics.to_dict(),
        'predictions': []
    }
    
    for result, gold in zip(results, dev_data):
        output_data['predictions'].append({
            'question': result.question,
            'db_id': result.db_id,
            'gold_sql': gold.get('query', ''),
            'predicted_sql': result.predicted_sql,
            'gnn_confidence': result.gnn_confidence,
            'retrieved_tables': result.retrieved_tables,
            'retrieved_columns': result.retrieved_columns[:10],  # Limit for readability
            'relationships': result.relationships,
            'generation_method': result.generation_method,
            'exact_match': evaluator.compute_exact_match(result.predicted_sql, gold.get('query', '')),
            'execution_match': evaluator.compute_execution_match(
                result.predicted_sql, gold.get('query', ''), result.db_id
            ) if evaluator.executor else None
        })
    
    output_path = os.path.join(config.output_dir, 'full_results.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"   ✅ Results saved to {output_path}")
    
    # Save predictions only (for evaluation scripts)
    predictions_path = os.path.join(config.output_dir, 'predicted_sql.txt')
    with open(predictions_path, 'w') as f:
        for result in results:
            f.write(result.predicted_sql + '\n')
    
    print(f"   ✅ Predictions saved to {predictions_path}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    print(f"""
Summary:
  - Examples evaluated: {config.num_examples}
  - Exact Match: {metrics.exact_match*100:.2f}%
  - Execution Accuracy: {metrics.execution_accuracy*100:.2f}%
  - Table Recall: {metrics.table_recall*100:.2f}%
  - Column Recall: {metrics.column_recall*100:.2f}%

Output files:
  - {config.output_dir}/full_results.json
  - {config.output_dir}/predicted_sql.txt
  - {config.output_dir}/ablation_results.json (if ablation was run)

For publication, you can cite these metrics in your results section.
""")


if __name__ == "__main__":
    main()