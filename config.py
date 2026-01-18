"""
Configuration file for RegLLM project.
Adjust these parameters to customize the training and inference.
"""

from pathlib import Path

# ============================================================================
# Directories
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# ============================================================================
# Scraping Configuration
# ============================================================================
SCRAPING = {
    # URL file with regulation sources
    'url_file': 'regurl.txt',

    # Delay between requests (seconds) - be respectful!
    'request_delay': 2,

    # Maximum PDFs to download per page
    'max_pdfs_per_page': 10,

    # Request timeout (seconds)
    'timeout': 30,
}

# ============================================================================
# Data Processing Configuration
# ============================================================================
DATA_PROCESSING = {
    # Chunk size for long documents (words)
    'chunk_size': 1000,

    # Overlap between chunks (words)
    'chunk_overlap': 200,

    # Minimum chunk size (words)
    'min_chunk_size': 50,

    # Validation set ratio
    'val_ratio': 0.15,

    # Small subset size for overfitting test
    'small_subset_size': 50,

    # Maximum sequence length for model
    'max_seq_length': 512,
}

# ============================================================================
# Model Configuration
# ============================================================================
MODEL = {
    # Base model to use
    # Options: 'qwen2.5-7b', 'phi-2', 'phi-3-mini', 'stablelm-3b', 'gemma-2b', 'qwen-1.8b'
    'base_model': 'qwen2.5-7b',  # 7B model for best quality (requires ~5-6GB GPU with 4-bit)

    # Use 4-bit quantization (reduces memory usage)
    'use_4bit': True,

    # LoRA configuration
    'lora': {
        'r': 64,                    # LoRA rank
        'lora_alpha': 128,          # LoRA alpha
        'lora_dropout': 0.05,       # Dropout rate
        'target_modules': [         # Modules to apply LoRA to
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    },
}

# ============================================================================
# Training Configuration
# ============================================================================
TRAINING = {
    # Small subset overfitting test
    'small_subset': {
        'epochs': 10,
        'batch_size': 2,
        'learning_rate': 3e-4,
        'gradient_accumulation_steps': 4,
    },

    # Full dataset training
    'full_training': {
        'epochs': 3,
        'batch_size': 4,
        'learning_rate': 2e-4,
        'gradient_accumulation_steps': 4,
    },

    # Common settings
    'weight_decay': 0.01,
    'warmup_steps': 100,
    'max_grad_norm': 1.0,
    'logging_steps': 10,
    'save_strategy': 'epoch',
    'evaluation_strategy': 'epoch',
    'save_total_limit': 2,  # Keep only N best checkpoints
}

# ============================================================================
# Inference Configuration
# ============================================================================
INFERENCE = {
    # Generation parameters
    'max_new_tokens': 300,
    'temperature': 0.7,
    'top_p': 0.95,
    'do_sample': True,

    # System prompt
    'system_prompt': """Eres un experto en regulación bancaria española. Tu tarea es responder preguntas sobre regulación bancaria, especialmente sobre parámetros de riesgo de crédito.

IMPORTANTE:
- Responde en Español, castellano
- Siempre cita la fuente de tu información
- Si no estás seguro o no tienes la información, responde "No tengo información suficiente para responder con certeza" o "Esta información no está disponible en los documentos proporcionados"
- Nunca inventes información
- Sé preciso y específico en tus respuestas""",
}

# ============================================================================
# UI Configuration
# ============================================================================
UI = {
    # Web interface
    'gradio': {
        'default_port': 7860,
        'share': False,  # Set True to create public link
        'theme': 'soft',
    },

    # Example questions
    'examples': [
        "¿Qué es la probabilidad de default (PD)?",
        "¿Qué regulación se aplica al cálculo de capital para riesgo de crédito?",
        "Explica el método IRB para carteras retail",
        "¿Cuáles son los requisitos para el cálculo de LGD?",

    ],
}

# ============================================================================
# RAG Configuration
# ============================================================================
RAG = {
    # Embedding model for semantic search
    'embedding_model': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',

    # Vector database path
    'vector_db_path': PROJECT_ROOT / 'vector_db' / 'chroma_db',

    # Search settings
    'default_n_results': 5,
    'hybrid_search_weight': 0.7,  # Weight for semantic vs keyword search

    # Chunk settings
    'max_chunk_size': 1500,  # characters
    'min_chunk_size': 100,   # characters
}

# ============================================================================
# Verification Configuration
# ============================================================================
VERIFICATION = {
    # Confidence thresholds
    'high_confidence': 0.8,
    'medium_confidence': 0.6,

    # Weights for verification score
    'weights': {
        'citations': 0.3,
        'coherence': 0.4,
        'hallucination': 0.2,
        'language': 0.1,
    }
}

# ============================================================================
# API Configuration
# ============================================================================
API = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': False,
}

# ============================================================================
# Scraping Configuration (Enhanced)
# ============================================================================
ENHANCED_SCRAPING = {
    # Rate limiting
    'delay_range': (2, 5),  # Random delay between requests
    'max_retries': 3,
    'backoff_factor': 2,

    # Selenium settings
    'use_selenium': True,
    'headless': True,

    # Sites requiring Selenium
    'selenium_sites': [
        'linkedin.com',
        'twitter.com',
        'x.com',
        'facebook.com',
    ]
}

# ============================================================================
# Logging Configuration
# ============================================================================
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'regllm.log',
}
