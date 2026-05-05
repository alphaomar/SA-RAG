"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              STRUCTURE-AWARE SQL GENERATOR (GROQ + GNN)                      ║
║                                                                              ║
║  Methodology:                                                                ║
║  1. Retrieve schema using trained GNN (best.pt)                              ║
║  2. Construct prompt with 'Reasoning' & 'Relationships' from GNN             ║
║  3. Generate SQL using Groq Llama-3.3-70b-Versatile                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import os
import torch
from tqdm import tqdm
from groq import Groq
from typing import List, Dict

# Import your existing modules
# Ensure gnn_models.py, retriver.py, and schemaBuider.py are in the same folder
# from gnn_models import EnhancedGNNConfig, EnhancedSchemaGNN
# from retriver import EnhancedSchemaRetriever, RetrievalResult

# # ==========================================
# # Configuration
# # ==========================================
# GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"  # Replace or use os.environ
# MODEL_ID = "llama-3.3-70b-versatile"     # High-performance model on Groq
# CHECKPOINT_PATH = "best.pt"              # Your trained GNN model
# TABLES_PATH = "tables.json"              # Required: Standard Spider tables.json
# DEV_DATA_PATH = "dev.json"               # Your question file
# OUTPUT_FILE = "predicted_sql.json"
GROQ_API_KEY = "#"  # Replace or use os.environ
MODEL_ID = "llama-3.3-70b-versatile"     # High-performance model on Groq
CHECKPOINT_PATH = "/kaggle/input/gnnmodel2/pytorch/default/1/best.pt"              # Your trained GNN model
TABLES_PATH = "/kaggle/input/yale-universitys-spider-10-nlp-dataset/spider/tables.json"              # Required: Standard Spider tables.json
DEV_DATA_PATH = "/kaggle/input/yale-universitys-spider-10-nlp-dataset/spider/dev.json"               # Your question file
OUTPUT_FILE = "./predicted_sql.json"


class SQLGenerator:
    def __init__(self, retriever: EnhancedSchemaRetriever, api_key: str):
        self.retriever = retriever
        self.client = Groq(api_key=api_key)
        self.model = MODEL_ID

    def construct_prompt(self, question: str, retrieval: RetrievalResult, db_id: str) -> str:
        """
        Constructs a Structure-Aware prompt for Llama 3.
        It uses GNN confidence scores to guide the LLM's focus.
        """
        
        # 1. Format Tables & Columns based on GNN scores
        schema_context = []
        for table, score in retrieval.tables:
            # Filter columns: only show those deemed relevant by the GNN
            relevant_cols = [
                f"{col} (match score: {s:.2f})" 
                for t, col, s in retrieval.columns 
                if t == table
            ]
            
            # If no specific columns found but table is high relevance, show all (fallback)
            if not relevant_cols and score > 0.8:
                relevant_cols = ["*all columns*"]
                
            if relevant_cols:
                schema_context.append(f"Table `{table}` (Relevance: {score:.2f})")
                schema_context.append(f"   Columns: {', '.join(relevant_cols)}")

        # 2. Format Relationships (Foreign Keys)
        # This is critical for EX (Execution Accuracy) to ensure correct JOINs
        relationships = retrieval.relationships
        rel_str = "\n".join([f"- {r}" for r in relationships]) if relationships else "No direct foreign keys detected among selected tables."

        # 3. Dynamic System Instruction
        # If GNN is unsure (low confidence), ask LLM to be cautious.
        confidence_note = ""
        if retrieval.confidence < 0.4:
            confidence_note = "Note: The schema retriever was uncertain. Please double-check table names and join conditions."

        prompt = f"""
You are an expert SQLite SQL developer. Generate a valid SQL query for the question below.

### Database Context (Pruned by GNN)
The following schema elements were retrieved as most relevant to the question:
{chr(10).join(schema_context)}

### Valid Relationships (Foreign Keys)
Use these to join tables correctly:
{rel_str}

### Question
{question}

### Instructions
1. Use ONLY the tables and columns provided above.
2. {confidence_note}
3. Do not use markdown. Output the SQL query strictly on a single line.
4. If the question asks for a ratio, multiply by 100.0.

### SQL Query
"""
        return prompt

    def generate_sql(self, prompt: str) -> str:
        """Calls Groq API to generate SQL."""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a text-to-SQL engine. You reply ONLY with valid SQL. No explanations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.1,  # Low temperature for deterministic code generation
                max_tokens=500,
                top_p=1,
                stop=None,
                stream=False,
            )
            return chat_completion.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()
        except Exception as e:
            print(f"Groq API Error: {e}")
            return "SELECT 'Error generating SQL'"

    def run_pipeline(self, dataset: List[Dict], output_path: str):
        """Runs the full GNN -> Groq pipeline."""
        results = []
        
        print(f"🚀 Starting Generation Pipeline with {self.model}...")
        print(f"📊 Processing {len(dataset)} examples...")
        print(f"📝 Output will be saved to {output_path}")
        
        for entry in tqdm(dataset, desc="Generating SQL"):
            question = entry['question']
            db_id = entry['db_id']
            
            # Step 1: Semantic Retrieval (GNN)
            try:
                retrieval = self.retriever.retrieve(
                    question=question,
                    db_id=db_id,
                    top_k_tables=5,   # Retrieve top 5 most relevant tables
                    top_k_columns=10, # Retrieve top 10 most relevant columns
                    use_dynamic_threshold=True # Use GNN's dynamic scoring
                )
            except Exception as e:
                print(f"GNN Retrieval failed for {db_id}: {e}")
                continue

            # Step 2: Prompt Engineering
            prompt = self.construct_prompt(question, retrieval, db_id)

            # Step 3: Generation (Groq)
            predicted_sql = self.generate_sql(prompt)
            
            # Step 4: Logging
            results.append({
                "question": question,
                "db_id": db_id,
                "gold_sql": entry.get('query', ''), # Ground truth if available
                "predicted_sql": predicted_sql,
                "gnn_confidence": retrieval.confidence,
                "retrieved_tables": [t[0] for t in retrieval.tables]
            })

        # Save Results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"✅ Completed! Results saved to {output_path}")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    
    # 1. Setup API Key
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        # Try getting from env if not set in script
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            # Fallback for testing if you want to hardcode it temporarily
            # api_key = "gsk_..." 
            raise ValueError("Please set your GROQ_API_KEY in the script or environment variables.")
    else:
        api_key = GROQ_API_KEY

    # 2. Check for required files
    if not os.path.exists(TABLES_PATH):
        print(f"⚠️ Warning: '{TABLES_PATH}' not found. The GNN needs this to understand table structures.")
        print("   Please download tables.json from the Spider dataset.")
        exit(1)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"⚠️ Warning: '{CHECKPOINT_PATH}' not found. Cannot load trained GNN model.")
        exit(1)

    # 3. Initialize GNN Retriever
    print("🧠 Loading GNN Retriever and Embedding models (this may take a moment)...")
    try:
        retriever = EnhancedSchemaRetriever.from_checkpoint(
            checkpoint_path=CHECKPOINT_PATH,
            embedding_model='all-MiniLM-L6-v2',  # Must match what you trained with
            tables_path=TABLES_PATH,
            model_class=EnhancedSchemaGNN,
            model_config=EnhancedGNNConfig()
        )
    except Exception as e:
        print(f"❌ Failed to load GNN model: {e}")
        exit(1)

    # 4. Load Data
    with open(DEV_DATA_PATH, 'r') as f:
        dev_data = json.load(f)

    # 5. Run Generator on FIRST 50 EXAMPLES
    generator = SQLGenerator(retriever, api_key)
    
    # Slicing the list here to only use the first 50 items
    mini_batch = dev_data[:50]
    
    generator.run_pipeline(mini_batch, OUTPUT_FILE)