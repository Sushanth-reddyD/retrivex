#!/usr/bin/env python3
"""
Pre-publish validation script for RetriVex.

Runs comprehensive checks before publishing to PyPI to ensure package quality.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and check for success."""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False

def check_file_exists(file_path, description):
    """Check if a required file exists."""
    print(f"🔍 Checking {description}...")
    if Path(file_path).exists():
        print(f"✅ {description} - EXISTS")
        return True
    else:
        print(f"❌ {description} - MISSING")
        return False

def main():
    """Run all validation checks."""
    print("🚀 RetriVex PyPI Pre-publish Validation")
    print("=" * 50)
    
    checks = []
    
    # Check required files
    required_files = [
        ("README.md", "README file"),
        ("LICENSE", "License file"),
        ("CHANGELOG.md", "Changelog"),
        ("pyproject.toml", "Package configuration"),
        ("src/retrivex/__init__.py", "Main package init"),
        ("EVALUATION_REPORT.md", "Evaluation report"),
    ]
    
    for file_path, description in required_files:
        checks.append(check_file_exists(file_path, description))
    
    # Build package
    checks.append(run_command("python -m build", "Building package"))
    
    # Check package with twine
    checks.append(run_command("python -m twine check dist/*", "Validating package with twine"))
    
    # Run tests
    checks.append(run_command("python -m pytest tests/ -v", "Running unit tests"))
    
    # Check imports
    checks.append(run_command("python -c 'import retrivex; print(retrivex.__version__)'", "Testing package imports"))
    
    # Lint checks
    checks.append(run_command("python -m ruff check src/", "Running linter (ruff)"))
    
    # Type checking
    checks.append(run_command("python -m mypy src/retrivex --ignore-missing-imports", "Type checking with mypy"))
    
    # Security check
    checks.append(run_command("python -m pip-audit", "Security audit (if pip-audit available)"))
    
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"🎉 ALL CHECKS PASSED ({passed}/{total})")
        print("\n✅ Package is ready for PyPI publication!")
        print("\n📦 To publish:")
        print("   Test PyPI: python -m twine upload --repository testpypi dist/*")
        print("   Production: python -m twine upload dist/*")
        return 0
    else:
        print(f"💥 {total - passed} CHECKS FAILED ({passed}/{total})")
        print("\n❌ Please fix the issues above before publishing.")
        return 1

if __name__ == "__main__":
    # Ensure we're in the project root
    if not Path("pyproject.toml").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Install required tools if missing
    try:
        subprocess.run(["python", "-m", "pip", "install", "build", "twine", "pytest", "ruff", "mypy"], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("⚠️  Could not install all validation tools. Some checks may fail.")
    
    sys.exit(main())
