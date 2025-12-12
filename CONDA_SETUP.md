# Conda Setup Instructions

This project can be set up using Miniconda or Anaconda.

## Quick Setup

1. **Install Miniconda** (if not already installed):
   ```bash
   # Download and install Miniconda
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   # Follow the prompts, then restart your terminal
   ```

2. **Run the setup script**:
   ```bash
   ./setup_conda.sh
   ```

3. **Activate the environment**:
   ```bash
   conda activate quantum-credit-score
   ```

4. **Run the program**:
   ```bash
   python main.py
   ```

## Manual Setup

Alternatively, you can create the environment manually:

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate quantum-credit-score

# Run the program
python main.py
```

## Updating the Environment

If you need to update the environment after changes:

```bash
conda env update -n quantum-credit-score -f environment.yml
```

## Deactivating

To deactivate the conda environment:

```bash
conda deactivate
```

## Removing the Environment

To remove the environment:

```bash
conda env remove -n quantum-credit-score
```

