#!/usr/bin/env python3
"""
Release script for RetriVex.

Automates the release process including version bumping, tagging, and publishing.
"""

import sys
import subprocess
import re
from pathlib import Path
from datetime import datetime

def update_version(version_file, new_version):
    """Update version in __init__.py"""
    content = version_file.read_text()
    content = re.sub(r'__version__ = "[^"]+"', f'__version__ = "{new_version}"', content)
    version_file.write_text(content)
    print(f"✅ Updated version in {version_file}")

def update_pyproject_version(pyproject_file, new_version):
    """Update version in pyproject.toml"""
    content = pyproject_file.read_text()
    content = re.sub(r'version = "[^"]+"', f'version = "{new_version}"', content)
    pyproject_file.write_text(content)
    print(f"✅ Updated version in {pyproject_file}")

def update_changelog(changelog_file, new_version):
    """Update changelog with new version"""
    content = changelog_file.read_text()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Replace [Unreleased] with the new version
    if "[Unreleased]" in content:
        content = content.replace(
            "## [Unreleased]",
            f"## [Unreleased]\n\n### Added\n- TBD\n\n## [{new_version}] - {today}"
        )
    else:
        # Add new version entry at the top
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('## ['):
                lines.insert(i, f"## [{new_version}] - {today}")
                lines.insert(i+1, "")
                break
        content = '\n'.join(lines)
    
    changelog_file.write_text(content)
    print(f"✅ Updated changelog with version {new_version}")

def run_command(cmd, description):
    """Run a command and check for success."""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/release.py <version>")
        print("Example: python scripts/release.py 0.2.0")
        sys.exit(1)
    
    new_version = sys.argv[1]
    
    # Validate version format
    if not re.match(r'^\d+\.\d+\.\d+(-\w+\.\d+)?$', new_version):
        print("❌ Invalid version format. Use semantic versioning (e.g., 1.0.0)")
        sys.exit(1)
    
    print(f"🚀 Releasing RetriVex v{new_version}")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Please run this script from the project root")
        sys.exit(1)
    
    # Check if working directory is clean
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if result.stdout.strip():
        print("❌ Working directory is not clean. Please commit or stash changes.")
        sys.exit(1)
    
    # Update version files
    version_file = Path("src/retrivex/__init__.py")
    pyproject_file = Path("pyproject.toml")
    changelog_file = Path("CHANGELOG.md")
    
    update_version(version_file, new_version)
    update_pyproject_version(pyproject_file, new_version)
    update_changelog(changelog_file, new_version)
    
    # Run validation
    if not run_command("python scripts/validate_package.py", "Running validation"):
        print("❌ Validation failed. Please fix issues before releasing.")
        sys.exit(1)
    
    # Commit changes
    if not run_command(f'git add -A && git commit -m "Release v{new_version}"', "Committing version changes"):
        sys.exit(1)
    
    # Create tag
    if not run_command(f'git tag -a v{new_version} -m "Release v{new_version}"', f"Creating tag v{new_version}"):
        sys.exit(1)
    
    # Build package
    if not run_command("python -m build", "Building package"):
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 RELEASE READY!")
    print("=" * 50)
    print(f"✅ Version {new_version} is ready for release")
    print("\n📦 Next steps:")
    print(f"   1. Push changes: git push origin main")
    print(f"   2. Push tag: git push origin v{new_version}")
    print(f"   3. Test publish: python -m twine upload --repository testpypi dist/*")
    print(f"   4. Publish: python -m twine upload dist/*")
    print(f"   5. Create GitHub release from tag v{new_version}")

if __name__ == "__main__":
    main()
