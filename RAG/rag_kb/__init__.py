"""
Initialize RAG KB package
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "Tunisian Arabic RAG Knowledge Base"

from pathlib import Path

# Package root
ROOT_DIR = Path(__file__).parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
SCHEMA_DIR = ROOT_DIR / "schemas"
PIPELINE_DIR = ROOT_DIR / "pipeline"
SCRIPTS_DIR = ROOT_DIR / "scripts"
DB_DIR = ROOT_DIR / "db"
CHROMA_DB_DIR = DB_DIR / "chroma_db"

__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "SCHEMA_DIR",
    "PIPELINE_DIR",
    "SCRIPTS_DIR",
    "DB_DIR",
    "CHROMA_DB_DIR",
]
