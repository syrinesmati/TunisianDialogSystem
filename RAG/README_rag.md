# RAG Knowledge Base for Tunisian Arabic

A clean, scalable knowledge base system for Tunisian Arabic expressions and proverbs with RAG (Retrieval-Augmented Generation) support.

## 📁 Structure

```
rag_kb/
├── data/                      # JSON data files
│   ├── expressions.json       # Tunisian Arabic expressions (starting with 10)
│   └── proverbs.json          # Tunisian Arabic proverbs (starting with 6)
│
├── schemas/                   # Pydantic validation schemas
│   ├── base_schema.py        # BaseEntry with common fields
│   ├── expression_schema.py  # ExpressionEntry schema
│   └── proverb_schema.py     # ProverbEntry schema
│
├── pipeline/                  # Core RAG logic
│   ├── build_embed_text.py   # Builds text for embeddings
│   └── validate_entries.py   # JSON validation
│
├── scripts/                   # Utility scripts
│   ├── add_entry.py          # Add single entries
│   └── bulk_import.py        # Bulk import from JSON/CSV
│
└── db/
    └── chroma_db/            # ChromaDB storage (created later)
```

## 🧩 Schemas

### BaseEntry (All entries have these)

```python
id: str
type: str  # "expression" or "proverb"
term_arabic: str
term_arabizi: str
meaning: str
meaning_fr: Optional[str]
example: str
usage_context: str
region: str  # e.g., "national"
register: str  # "formal" or "informal"
generation: str  # "all", "youth", etc.
scripts: List[str]  # ["arabic", "arabizi"]
source: str
last_updated: str
```

### ExpressionEntry (specific fields)

```python
origin: Optional[str]
severity: str  # "neutral", "offensive", etc.
gender_sensitive: bool
```

### ProverbEntry (specific fields)

```python
literal_meaning: str
real_meaning: str
when_used: str
msa_equivalent: Optional[str]
```

## 🧠 Core Function: `build_embed_text()`

This is the **most important** function. It determines what your embeddings "see":

```python
from pipeline.build_embed_text import build_embed_text

entry = {"term_arabic": "برشا", "meaning": "كثير", ...}
text = build_embed_text(entry)
# Returns: "برشا barcha كثير ... [all relevant fields]"
```

**This text is what gets embedded and searched.**

## ✅ Validation

Always validate before adding entries:

```python
from pipeline.validate_entries import validate_file, print_validation_report

valid, errors = validate_file("data/expressions.json")
print_validation_report(valid, errors)
```

Bad data ruins retrieval quality. Validate everything.

## ➕ Adding Entries

### Bulk Import from JSON

```python
from scripts.bulk_import import bulk_import_from_json

imported, failed = bulk_import_from_json("path/to/entries.json", "expression")
```

### Bulk Import from CSV

CSV headers should match schema fields:

```
term_arabic,term_arabizi,meaning,example,usage_context,severity,gender_sensitive
برشا,barcha,كثير,عندي برشا,يومياً,neutral,false
```

```python
from scripts.bulk_import import bulk_import_from_csv

imported, failed = bulk_import_from_csv("entries.csv", "expression")
```

## 📊 Current Status

✓ 10 expressions  
✓ 155 proverbs  
→ Next: Scale, then integrate with ChromaDB

## 🔄 Workflow

1. **Add/Import Data** → `scripts/add_entry.py` or `scripts/bulk_import.py`
2. **Validate** → `pipeline/validate_entries.py`
3. **Check Embeddings** → `pipeline/build_embed_text.py`
4. **Store in ChromaDB** → (Next phase)
5. **Retrieve via RAG** → (Next phase)

## 📦 Requirements

```bash
pip install -r requirements.txt
```

- `pydantic>=2.0` — Data validation
- `chromadb>=0.3.0` — Vector storage

## 🎯 Next Steps

1. ✅ Set up folder structure
2. ✅ Create Pydantic schemas
3. ✅ Build `build_embed_text()` core function
4. ✅ Create validation
5. ✅ Add example data (10 expressions, 6 proverbs)
6. ⏭ Scale data (50 expressions, 30 proverbs)
7. ⏭ Integrate with ChromaDB
8. ⏭ Build RAG retrieval interface
9. ⏭ Connect to LLM for answer generation

---

**Note:** Start small. Test retrieval quality with 50-80 entries before expanding to thousands.
