#!/bin/bash

# Fix Setup Script - Resolve installation issues
# Run this after the initial setup failed

set -e

echo "========================================="
echo "Fixing Setup Issues"
echo "========================================="
echo ""

PROJECT_DIR="/Users/akhil/Documents/GitHub/RNA"
cd "$PROJECT_DIR"

# Initialize conda for bash
eval "$(conda shell.bash hook)" 2>/dev/null || echo "Note: Using system Python if conda not available"

# Activate environment if it exists
conda activate rna_folding 2>/dev/null || echo "Using system Python"

# Step 1: Install Python packages
echo "[1/4] Installing Python packages..."
pip3 install -r requirements.txt
echo "    ✓ Python packages installed"
echo ""

# Step 2: Fix US-align compilation (remove -static flag for macOS)
echo "[2/4] Fixing US-align compilation..."
cd USalign

# Recompile without -static flag (not supported on macOS)
g++ -O3 -ffast-math -lm -o USalign USalign.cpp

if [ -f "USalign" ]; then
    echo "    ✓ US-align compiled successfully"
else
    echo "    ✗ US-align compilation failed"
    echo "    Trying alternative compilation..."
    # Try without any optimization flags
    g++ -o USalign USalign.cpp
fi

cd "$PROJECT_DIR"
echo ""

# Step 3: Download DRfold2 model weights (using curl instead of wget)
echo "[3/4] Downloading DRfold2 model weights (~1.3GB)..."
cd DRfold2

if [ ! -f "model_hub.tar.gz" ]; then
    echo "    Downloading with curl..."
    curl -L -o model_hub.tar.gz "https://github.com/leeyang/DRfold2/releases/download/v1.0/model_hub.tar.gz"
fi

if [ -f "model_hub.tar.gz" ]; then
    echo "    Extracting model weights..."
    tar -xzf model_hub.tar.gz
    echo "    ✓ DRfold2 models downloaded and extracted"
else
    echo "    ⚠ Could not download models automatically"
    echo "    Please download manually from:"
    echo "    https://github.com/leeyang/DRfold2/releases/download/v1.0/model_hub.tar.gz"
    echo "    And extract in DRfold2/ directory"
fi

cd "$PROJECT_DIR"
echo ""

# Step 4: Verify installation
echo "[4/4] Verifying installation..."
echo ""

# Check Python packages
echo "Checking Python packages:"
python3 -c "import pandas; print(f'  ✓ pandas {pandas.__version__}')" || echo "  ✗ pandas not found"
python3 -c "import numpy; print(f'  ✓ numpy {numpy.__version__}')" || echo "  ✗ numpy not found"
python3 -c "import scipy; print(f'  ✓ scipy {scipy.__version__}')" || echo "  ✗ scipy not found"
python3 -c "import Bio; print(f'  ✓ biopython {Bio.__version__}')" || echo "  ✗ biopython not found"
python3 -c "import matplotlib; print(f'  ✓ matplotlib {matplotlib.__version__}')" || echo "  ✗ matplotlib not found"
python3 -c "import tqdm; print(f'  ✓ tqdm {tqdm.__version__}')" || echo "  ✗ tqdm not found"
echo ""

# Check US-align
if [ -f "USalign/USalign" ]; then
    echo "✓ US-align: Found at USalign/USalign"
else
    echo "✗ US-align: Not found"
fi
echo ""

# Check DRfold2
if [ -f "DRfold2/DRfold_infer.py" ]; then
    echo "✓ DRfold2: Found at DRfold2/"
else
    echo "✗ DRfold2: Not found"
fi
echo ""

echo "========================================="
echo "Setup Fix Complete!"
echo "========================================="
echo ""
echo "You can now run:"
echo "  python scripts/process_training_data.py"
echo ""
