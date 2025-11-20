#!/bin/bash

# Stanford RNA 3D Folding - Environment Setup Script
# This script sets up the benchmarking environment for the competition winner's approach

set -e  # Exit on error

echo "========================================="
echo "Stanford RNA 3D Folding - Setup Script"
echo "========================================="
echo ""

# Configuration
PROJECT_DIR="/Users/akhil/Documents/GitHub/RNA"
CONDA_ENV_NAME="rna_folding"
PYTHON_VERSION="3.11"

cd "$PROJECT_DIR"

# Step 1: Create conda environment
echo "[1/6] Creating conda environment: $CONDA_ENV_NAME"
if conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo "    Environment already exists. Removing..."
    conda env remove -n "$CONDA_ENV_NAME" -y
fi

conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
echo "    ✓ Conda environment created"
echo ""

# Step 2: Install DRfold2
echo "[2/6] Cloning DRfold2 repository"
if [ -d "DRfold2" ]; then
    echo "    DRfold2 directory exists. Removing..."
    rm -rf DRfold2
fi

git clone https://github.com/leeyang/DRfold2.git
cd DRfold2
echo "    ✓ DRfold2 cloned"
echo ""

# Step 3: Install DRfold2 dependencies
echo "[3/6] Installing DRfold2 dependencies"
echo "    This will download ~1.3GB of model weights..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

bash install.sh
echo "    ✓ DRfold2 installed"
echo ""

# Step 4: Install evaluation tools
echo "[4/6] Installing TM-score calculator (US-align)"
cd "$PROJECT_DIR"
if [ -d "USalign" ]; then
    echo "    USalign directory exists. Removing..."
    rm -rf USalign
fi

git clone https://github.com/pylelab/USalign.git
cd USalign
g++ -static -O3 -ffast-math -lm -o USalign USalign.cpp
echo "    ✓ US-align compiled"
echo ""

# Step 5: Install Python dependencies
echo "[5/6] Installing additional Python packages"
cd "$PROJECT_DIR"
conda activate "$CONDA_ENV_NAME"

pip install pandas numpy scipy biopython matplotlib seaborn jupyter tqdm
echo "    ✓ Python packages installed"
echo ""

# Step 6: Create utility scripts directory
echo "[6/6] Creating utility scripts directory"
mkdir -p "$PROJECT_DIR/scripts"
echo "    ✓ Scripts directory created"
echo ""

# Create test script
cat > "$PROJECT_DIR/scripts/test_installation.py" << 'EOF'
#!/usr/bin/env python3
"""Test installation of DRfold2 and dependencies"""

import sys
import subprocess

def test_imports():
    """Test Python package imports"""
    print("Testing Python imports...")
    try:
        import numpy
        import scipy
        import Bio
        import torch
        print("  ✓ All imports successful")
        print(f"    - PyTorch version: {torch.__version__}")
        print(f"    - CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_drfold2():
    """Test DRfold2 installation"""
    print("\nTesting DRfold2...")
    try:
        # Check if DRfold_infer.py exists
        import os
        drfold_script = "/Users/akhil/Documents/GitHub/RNA/DRfold2/DRfold_infer.py"
        if os.path.exists(drfold_script):
            print(f"  ✓ DRfold2 script found: {drfold_script}")
            return True
        else:
            print(f"  ✗ DRfold2 script not found")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_usalign():
    """Test US-align installation"""
    print("\nTesting US-align...")
    try:
        result = subprocess.run(
            ["/Users/akhil/Documents/GitHub/RNA/USalign/USalign"],
            capture_output=True,
            timeout=5
        )
        print("  ✓ US-align executable found")
        return True
    except FileNotFoundError:
        print("  ✗ US-align not found")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    print("="*50)
    print("Installation Test Suite")
    print("="*50)

    results = []
    results.append(("Python Imports", test_imports()))
    results.append(("DRfold2", test_drfold2()))
    results.append(("US-align", test_usalign()))

    print("\n" + "="*50)
    print("Test Results Summary")
    print("="*50)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("="*50)
    if all_passed:
        print("\n✓ All tests passed! Environment is ready.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x "$PROJECT_DIR/scripts/test_installation.py"

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $CONDA_ENV_NAME"
echo ""
echo "To test the installation, run:"
echo "  python scripts/test_installation.py"
echo ""
echo "Next steps:"
echo "  1. Test the installation"
echo "  2. Convert test sequences to FASTA format"
echo "  3. Run DRfold2 on test sequences"
echo "  4. Evaluate results with TM-score"
echo ""
echo "See RESEARCH_PLAN.md for detailed instructions."
echo "========================================="
