#!/bin/bash
# Setup script for Quantum Credit Score Prediction using Miniconda/Conda

set -e

echo "=========================================="
echo "Quantum Credit Score - Conda Setup"
echo "=========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo ""
    echo "Please install Miniconda first:"
    echo "  1. Download: https://docs.conda.io/en/latest/miniconda.html"
    echo "  2. Run: bash Miniconda3-latest-Linux-x86_64.sh"
    echo "  3. Restart your terminal or run: source ~/.bashrc"
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "quantum-credit-score"; then
    echo "Environment 'quantum-credit-score' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n quantum-credit-score -y
    else
        echo "Updating existing environment..."
        conda env update -n quantum-credit-score -f environment.yml
        echo ""
        echo "✓ Environment updated!"
        echo ""
        echo "To activate the environment, run:"
        echo "  conda activate quantum-credit-score"
        exit 0
    fi
fi

# Create the environment
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate quantum-credit-score"
echo ""
echo "Then run the program with:"
echo "  python main.py"
echo ""

