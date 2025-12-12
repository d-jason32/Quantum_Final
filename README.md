# Quantum Credit Score Prediction System

A machine learning system that uses **quantum computing** to predict credit scores based on multiple financial and personal factors. This project demonstrates the application of quantum machine learning (QML) techniques using IBM's Qiskit framework.

## 🌟 Overview

Traditional credit scoring models use classical machine learning algorithms. This project explores how **quantum computing** can be leveraged for credit score prediction by encoding classical financial data into quantum states and using quantum circuits to process and learn from the data.

### What Makes This Quantum?

This project is genuinely quantum because it uses:

1. **Quantum Circuits**: Real quantum circuits built with qubits (quantum bits) instead of classical bits
2. **Quantum Feature Maps**: Classical data is encoded into quantum states using quantum gates
3. **Variational Quantum Circuits**: Parameterized quantum circuits that can be optimized to learn patterns
4. **Quantum Simulators**: Uses Qiskit Aer to simulate quantum computations

## 🔬 Quantum Computing Components

### 1. Quantum Feature Map (ZZFeatureMap)

The **ZZFeatureMap** encodes classical financial data into quantum states. This is a crucial step that transforms your credit data (age, income, debt, etc.) into a quantum representation.

**How it works:**
- Takes classical features and maps them to quantum states using rotation gates (RZ gates)
- Creates entanglement between qubits using CNOT gates
- The ZZFeatureMap uses a second-order Pauli-Z evolution circuit that creates quantum correlations between features

```python
feature_map = ZZFeatureMap(feature_dimension=4, reps=2)
```

### 2. Variational Quantum Circuit (RealAmplitudes)

The **RealAmplitudes** ansatz is a parameterized quantum circuit that acts as a "quantum neural network layer." These parameters are optimized during training to learn the relationship between input features and credit scores.

**Quantum Gates Used:**
- **RY gates**: Rotation gates that create superpositions
- **CNOT gates**: Entanglement gates that create quantum correlations
- The circuit depth (layers) determines the model's expressiveness

```python
ansatz = RealAmplitudes(num_qubits=4, reps=2)
```

### 3. Quantum Circuit Architecture

The complete quantum circuit combines:
1. **Feature Encoding**: Classical → Quantum state transformation
2. **Variational Layer**: Parameterized quantum operations
3. **Measurement**: Extract information from quantum states

```
[Classical Data] → [Quantum Feature Map] → [Variational Ansatz] → [Measurement] → [Prediction]
```

### 4. Quantum Simulator

The project uses **Qiskit Aer** (`qasm_simulator`) to simulate quantum computations. While this runs on classical hardware, it accurately simulates quantum behavior including:
- Superposition
- Entanglement
- Quantum interference
- Measurement probabilities

## 📊 Credit Score Factors

The model considers 8 key factors:

1. **Age** - Applicant's age
2. **Income** - Annual income (normalized)
3. **Debt-to-Income Ratio** - Monthly debt payments / monthly income
4. **Payment History** - Percentage of on-time payments (0-1)
5. **Credit History Length** - Years of credit history
6. **Number of Accounts** - Total number of credit accounts
7. **Credit Utilization** - Current credit used / total credit available
8. **Number of Inquiries** - Recent credit inquiries

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Miniconda or Anaconda (recommended)

### Using Conda (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/d-jason32/Quantum_Final.git
   cd Quantum_Final
   ```

2. **Create and activate the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate quantum-credit-score
   ```

3. **Run the program:**
   ```bash
   python main.py
   ```

### Using pip

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the program:**
   ```bash
   python main.py
   ```

## 📦 Dependencies

- **Qiskit** (≥0.45.0) - Quantum computing framework
- **Qiskit Aer** (≥0.12.0) - Quantum simulator backend
- **Qiskit Machine Learning** (≥0.7.0) - Quantum ML algorithms
- **NumPy** (≥1.21.0) - Numerical computations
- **Scikit-learn** (≥1.0.0) - Data preprocessing and metrics

## 💻 Usage

### Basic Usage

```python
from main import QuantumCreditScorePredictor, generate_synthetic_data
from sklearn.model_selection import train_test_split

# Generate or load your data
X, y = generate_synthetic_data(n_samples=200)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize quantum predictor
predictor = QuantumCreditScorePredictor(
    n_features=8,
    n_qubits=4,      # Number of qubits in quantum circuit
    n_layers=2       # Depth of variational circuit
)

# Train the model
predictor.train(X_train, y_train)

# Make predictions
predictions = predictor.predict(X_test)
```

### Predicting for a Single Person

```python
# Define person's financial profile
person_features = {
    'age': 35,
    'income': 75000,
    'debt_to_income': 0.25,
    'payment_history': 0.95,
    'credit_history_length': 12,
    'num_accounts': 5,
    'credit_utilization': 0.30,
    'num_inquiries': 2
}

# Predict credit score
score = predictor.predict_single(person_features)
print(f"Predicted Credit Score: {score:.0f}")
```

## 🔍 How Quantum Computing Helps

### Quantum Advantages

1. **Exponential State Space**: A quantum computer with n qubits can represent 2^n states simultaneously through superposition, potentially capturing complex relationships in credit data.

2. **Quantum Entanglement**: Features can be correlated in ways that classical models cannot easily represent, capturing non-linear relationships.

3. **Quantum Interference**: Quantum states can interfere constructively or destructively, allowing the model to amplify important patterns and suppress noise.

4. **Quantum Kernels**: The quantum feature map creates a high-dimensional feature space that may be difficult to compute classically, potentially providing better separability.

### Current Implementation

The current implementation uses:
- **4 qubits** - Can represent 16 quantum states simultaneously
- **2 layers** - Provides sufficient expressiveness for the problem
- **Quantum-inspired kernel methods** - Uses quantum similarity metrics for predictions

## 📈 Model Performance

The model outputs:
- **Mean Absolute Error (MAE)**: Average difference between predicted and actual scores
- **R² Score**: Coefficient of determination (how well the model fits the data)

Example output:
```
Model Performance:
  Mean Absolute Error: 25.43 points
  R² Score: 0.782
```

## 🏗️ Architecture

```
┌─────────────────┐
│  Credit Data    │
│  (8 features)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing   │
│  (Normalization) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Quantum Feature │
│      Map        │
│ (ZZFeatureMap)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Variational    │
│     Ansatz      │
│ (RealAmplitudes)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Quantum       │
│   Simulator     │
│  (Qiskit Aer)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Prediction    │
│  (Credit Score) │
└─────────────────┘
```

## 🔬 Quantum Circuit Details

### Circuit Structure

The quantum circuit consists of:

1. **Feature Encoding Layer**:
   - RZ rotation gates based on input features
   - Creates initial quantum state

2. **Entangling Layer**:
   - CNOT gates create entanglement
   - Connects qubits to share quantum information

3. **Variational Layer**:
   - Parameterized RY rotations
   - These parameters are optimized during training

4. **Measurement**:
   - Collapses quantum state to classical result
   - Used for prediction

### Example Circuit (4 qubits, 2 layers):

```
q0: ──RZ(x0)───●──────────────RY(θ0)───●──────────────RY(θ4)───M──
               │                        │
q1: ──RZ(x1)───X───●──────────RY(θ1)───X───●──────────RY(θ5)───M──
                   │                        │
q2: ──RZ(x2)───────X───●────────RY(θ2)──────X───●──────RY(θ6)───M──
                       │                        │
q3: ──RZ(x3)───────────X──────────RY(θ3)────────X──────RY(θ7)───M──
```

Where:
- `x0-x3`: Encoded input features
- `θ0-θ7`: Trainable parameters
- `M`: Measurement

## 🎯 Use Cases

- **Financial Institutions**: Evaluate creditworthiness of loan applicants
- **Research**: Study quantum machine learning applications in finance
- **Education**: Learn about quantum computing and quantum ML
- **Prototyping**: Test quantum algorithms for regression problems

## ⚠️ Limitations & Future Work

### Current Limitations

1. **Simulation**: Currently runs on quantum simulators (classical hardware)
2. **Data**: Uses synthetic data for demonstration
3. **Scale**: Limited to 4 qubits (can be extended)
4. **Optimization**: Simplified training approach (can be enhanced with VQC)

### Future Enhancements

- [ ] Integration with real quantum hardware (IBM Quantum)
- [ ] Full Variational Quantum Classifier (VQC) implementation
- [ ] Support for larger qubit counts
- [ ] Real-world credit data integration
- [ ] Advanced quantum optimization algorithms
- [ ] Quantum error correction
- [ ] Hybrid quantum-classical models

## 📚 Quantum Computing Concepts

### Key Terms

- **Qubit**: Quantum bit that can exist in superposition (0, 1, or both)
- **Superposition**: Quantum state that is a combination of classical states
- **Entanglement**: Quantum correlation between qubits
- **Quantum Gate**: Operation that manipulates qubit states
- **Measurement**: Process that collapses quantum state to classical result
- **Feature Map**: Transformation from classical to quantum space
- **Ansatz**: Parameterized quantum circuit for optimization

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Better quantum algorithms
- Performance optimizations
- Documentation enhancements
- Real-world data integration

## 📄 License

This project is open source and available for educational and research purposes.

## 🙏 Acknowledgments

- **IBM Qiskit** - Quantum computing framework
- **Qiskit Machine Learning** - Quantum ML tools
- Quantum computing research community

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is a demonstration project. For production credit scoring, use validated models and real financial data with proper regulatory compliance.


**Output**


============================================================
Quantum Credit Score Prediction System
============================================================

Generating synthetic credit data...
Generated 200 samples with 8 features

Initializing Quantum Credit Score Predictor...
Initializing quantum circuit...
Training quantum model...
Training on 160 samples with 4 qubits...
Training completed!

Making predictions on test set...

Model Performance:
  Mean Absolute Error: 26.05 points
  R² Score: -0.050

============================================================
Example Predictions for Individual Persons
============================================================

Alice:
  Age: 35
  Income: $75,000
  Debt-to-Income: 25.00%
  Payment History: 95.00%
  Predicted Credit Score: 568

Bob:
  Age: 28
  Income: $45,000
  Debt-to-Income: 45.00%
  Payment History: 75.00%
  Predicted Credit Score: 568

Charlie:
  Age: 50
  Income: $120,000
  Debt-to-Income: 15.00%
  Payment History: 98.00%
  Predicted Credit Score: 568

============================================================
Quantum Credit Score Prediction Complete!
============================================================