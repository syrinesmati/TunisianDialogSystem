"""
test_setup.py

Quick validation that the entire setup works correctly
"""

import json
from pathlib import Path
import sys

# Add package to path
pkg_root = Path(__file__).parent / "rag_kb"
sys.path.insert(0, str(pkg_root.parent))

def test_schemas():
    """Test that schemas can be imported and validated"""
    print("\n🧪 Testing Schemas...")
    from rag_kb.schemas import ExpressionEntry, ProverbEntry
    
    # Test expression
    expr_data = {
        "id": "test_001",
        "type": "expression",
        "term_arabic": "برشا",
        "term_arabizi": "barcha",
        "meaning": "كثير",
        "meaning_fr": None,
        "example": "عندي برشا خدمة",
        "usage_context": "يومياً",
        "origin": "عربي",
        "severity": "neutral",
        "gender_sensitive": False,
        "region": "national",
        "register": "informal",
        "generation": "all",
        "scripts": ["arabic", "arabizi"],
        "source": "test",
        "last_updated": "2026-04-17"
    }
    
    try:
        ExpressionEntry(**expr_data)
        print("  ✓ ExpressionEntry validated")
    except Exception as e:
        print(f"  ✗ ExpressionEntry failed: {e}")
        return False
    
    # Test proverb
    prov_data = {
        "id": "test_001",
        "type": "proverb",
        "term_arabic": "اللي فات مات",
        "term_arabizi": "elli fet met",
        "meaning": "عدم التعلق بالماضي",
        "meaning_fr": None,
        "example": "سيب عليك",
        "usage_context": "عند الحزن",
        "literal_meaning": "ما مضى انتهى",
        "real_meaning": "لا فائدة من الندم",
        "when_used": "الندم",
        "msa_equivalent": "ما فات مات",
        "region": "national",
        "register": "informal",
        "generation": "all",
        "scripts": ["arabic", "arabizi"],
        "source": "test",
        "last_updated": "2026-04-17"
    }
    
    try:
        ProverbEntry(**prov_data)
        print("  ✓ ProverbEntry validated")
    except Exception as e:
        print(f"  ✗ ProverbEntry failed: {e}")
        return False
    
    return True


def test_pipeline():
    """Test pipeline functions"""
    print("\n🧪 Testing Pipeline...")
    from rag_kb.pipeline import build_embed_text, validate_file, print_validation_report
    
    # Test build_embed_text
    entry = {
        "type": "expression",
        "term_arabic": "برشا",
        "term_arabizi": "barcha",
        "meaning": "كثير",
        "example": "عندي برشا خدمة",
        "usage_context": "يومياً",
        "origin": "عربي"
    }
    
    text = build_embed_text(entry)
    if text:
        print(f"  ✓ build_embed_text works")
        print(f"    Output: '{text[:60]}...'")
    else:
        print(f"  ✗ build_embed_text returned empty")
        return False
    
    # Test validation
    data_dir = Path(__file__).parent / "rag_kb" / "data"
    valid, errors = validate_file(str(data_dir / "expressions.json"))
    
    print(f"  ✓ Validated {len(valid)} expressions")
    if errors:
        print(f"    ⚠ Found {len(errors)} errors")
        for eid, emsg in errors[:3]:
            print(f"      [{eid}] {emsg[:50]}")
    
    return True


def test_data_files():
    """Test that data files exist and are valid JSON"""
    print("\n🧪 Testing Data Files...")
    data_dir = Path(__file__).parent / "rag_kb" / "data"
    
    for file in ["expressions.json", "proverbs.json"]:
        file_path = data_dir / file
        if not file_path.exists():
            print(f"  ✗ {file} not found")
            return False
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            count = len(data)
            print(f"  ✓ {file}: {count} entries")
            
            # Show first entry
            if count > 0:
                first = data[0]
                print(f"    Sample: {first.get('term_arabic')} ({first.get('term_arabizi')})")
        
        except Exception as e:
            print(f"  ✗ {file} failed: {e}")
            return False
    
    return True


def test_structure():
    """Test that directory structure is correct"""
    print("\n🧪 Testing Directory Structure...")
    required_dirs = [
        "rag_kb",
        "rag_kb/data",
        "rag_kb/schemas",
        "rag_kb/pipeline",
        "rag_kb/scripts",
        "rag_kb/db/chroma_db",
    ]
    
    root = Path(__file__).parent
    for dir_name in required_dirs:
        dir_path = root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ missing")
            return False
    
    return True


def main():
    print("=" * 60)
    print("RAG KB SETUP VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_structure),
        ("Data Files", test_data_files),
        ("Schemas", test_schemas),
        ("Pipeline", test_pipeline),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ Setup is complete and working!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check README.md for usage examples")
        print("3. Add more entries using scripts/add_entry.py")
        return 0
    else:
        print("\n⚠ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit(main())
