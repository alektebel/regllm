#!/usr/bin/env python3
"""
Verification script to check that all components are properly set up.
Run this after installation to verify the project is ready to use.
"""

import sys
from pathlib import Path
import importlib.util

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current:", f"{version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check required dependencies."""
    print("\nChecking dependencies...")
    required = [
        'torch', 'transformers', 'peft', 'accelerate',
        'requests', 'beautifulsoup4', 'PyPDF2',
        'matplotlib', 'gradio', 'numpy'
    ]

    missing = []
    for package in required:
        spec = importlib.util.find_spec(package)
        if spec is None:
            print(f"❌ {package} not found")
            missing.append(package)
        else:
            print(f"✓ {package}")

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    return True

def check_directories():
    """Check project directories."""
    print("\nChecking directories...")
    required_dirs = [
        'data/raw', 'data/processed', 'data/train', 'data/val',
        'models', 'logs',
        'src/scraper', 'src/preprocessing', 'src/training', 'src/ui'
    ]

    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"❌ {dir_path} not found")
            all_exist = False

    return all_exist

def check_files():
    """Check key project files."""
    print("\nChecking key files...")
    required_files = [
        'regurl.txt',
        'requirements.txt',
        'config.py',
        'run_pipeline.py',
        'src/scraper/regulation_scraper.py',
        'src/preprocessing/data_processor.py',
        'src/training/model_setup.py',
        'src/training/train.py',
        'src/ui/chat_interface.py'
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} not found")
            all_exist = False

    return all_exist

def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU available: {device_name}")
            print(f"  Memory: {memory:.2f} GB")
            return True
        else:
            print("⚠ No GPU detected. Training will be slower on CPU.")
            print("  Consider using Google Colab or a cloud GPU.")
            return True  # Not a failure, just slower
    except:
        print("⚠ Could not check GPU status")
        return True

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("RegLLM Setup Verification")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directories", check_directories),
        ("Files", check_files),
        ("GPU", check_gpu),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error checking {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = all(result for _, result in results)

    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print()

    if all_passed:
        print("✓ All checks passed! Ready to start.")
        print("\nNext steps:")
        print("1. Run: python run_pipeline.py --all")
        print("2. Or follow QUICKSTART.md for step-by-step instructions")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nTroubleshooting:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Check Python version: python --version")
        print("- See README.md for detailed setup instructions")
        return 1

if __name__ == "__main__":
    sys.exit(main())
