"""Script to generate documentation for all modules in src/ using pdoc.

This script discovers all Python modules in the src/ package and generates
documentation for them using pdoc.

Note: This script requires all project dependencies to be installed,
as pdoc needs to import the modules to document them.
"""

import os
import pkgutil
import subprocess
import sys
from pathlib import Path

# Add the server directory to the path so we can import src
server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))

# Set UTF-8 encoding for Windows compatibility
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    import src

    # Discover all modules recursively
    modules = []
    for _, name, _ in pkgutil.walk_packages(src.__path__, src.__name__ + "."):
        modules.append(name)

    if not modules:
        print("No modules found to document.")
        sys.exit(1)

    print(f"Found {len(modules)} modules to document:")
    for module in sorted(modules):
        print(f"  - {module}")

    # Build pdoc command - use file paths instead of module names
    # This allows pdoc to parse files even if imports fail
    docs_dir = server_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    # Use the src directory as a path - pdoc will discover all modules
    cmd = [
        sys.executable,
        "-m",
        "pdoc",
        "--output-dir",
        str(docs_dir),
        "--search",  # Enable search
        "--show-source",  # Show source code
        str(server_dir / "src"),  # Use file path instead of module names
    ]

    print(f"\nGenerating documentation to {docs_dir}...")
    print("Note: Some modules may show warnings if dependencies are not installed.")
    print("      Install dependencies with: pip install -r requirements.txt\n")
    
    result = subprocess.run(
        cmd,
        cwd=server_dir,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    if result.returncode == 0:
        print(f"\n[SUCCESS] Documentation generated successfully!")
        print(f"  Open {docs_dir / 'index.html'} in your browser to view it.")
    else:
        print(f"\n[ERROR] Documentation generation failed with exit code {result.returncode}")
        print("  Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(result.returncode)

except ImportError as e:
    print(f"Error importing src: {e}")
    print("Make sure you're running this from the server directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error generating documentation: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

