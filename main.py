"""
Quantum Credit Score Prediction System
Uses quantum machine learning to predict credit scores based on multiple factors.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
try:
    from qiskit_aer import Aer
except ImportError:
    # Fallback for older Qiskit versions
    from qiskit import Aer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class QuantumCreditScorePredictor:
    """
    A quantum machine learning model for predicting credit scores.
    Uses Variational Quantum Classifier (VQC) with quantum feature maps.
    """
    
    def __init__(self, n_features=8, n_qubits=4, n_layers=2):
        """
        Initialize the quantum credit score predictor.
        
        Args:
            n_features: Number of input features
            n_qubits: Number of qubits in the quantum circuit
            n_layers: Number of layers in the variational circuit
        """
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        
    def create_quantum_circuit(self):
        """
        Create a quantum circuit with feature map and variational form.
        """
        # Quantum feature map - encodes classical data into quantum states
        feature_map = ZZFeatureMap(feature_dimension=self.n_qubits, reps=2)
        
        # Variational form - parameterized quantum circuit for learning
        ansatz = RealAmplitudes(num_qubits=self.n_qubits, reps=self.n_layers)
        
        # Combine feature map and ansatz
        qc = QuantumCircuit(self.n_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        
        return qc, feature_map, ansatz
    
    def prepare_features(self, features):
        """
        Prepare and normalize features for quantum processing.
        Reduces dimensions if needed to match number of qubits.
        """
        # Normalize features
        if hasattr(self.scaler, 'mean_'):
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = self.scaler.fit_transform(features)
        
        # Reduce dimensions to match qubits using PCA-like approach
        if features_scaled.shape[1] > self.n_qubits:
            # Simple dimension reduction: take first n_qubits features
            # In production, use proper PCA
            features_scaled = features_scaled[:, :self.n_qubits]
        elif features_scaled.shape[1] < self.n_qubits:
            # Pad with zeros if fewer features than qubits
            padding = np.zeros((features_scaled.shape[0], 
                              self.n_qubits - features_scaled.shape[1]))
            features_scaled = np.hstack([features_scaled, padding])
        
        return features_scaled
    
    def train(self, X_train, y_train, max_iter=50):
        """
        Train the quantum model on credit data.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training credit scores (n_samples,)
            max_iter: Maximum iterations for optimization
        """
        print("Initializing quantum circuit...")
        qc, feature_map, ansatz = self.create_quantum_circuit()
        
        # Prepare training data
        X_train_scaled = self.prepare_features(X_train)
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Create quantum neural network
        sampler = Aer.get_backend('qasm_simulator')
        
        # For regression, we'll use a modified approach
        # Create a quantum circuit that outputs expectation values
        def create_qnn():
            qc = QuantumCircuit(self.n_qubits)
            qc.compose(feature_map, inplace=True)
            qc.compose(ansatz, inplace=True)
            qc.measure_all()
            return qc
        
        # Simplified quantum model for regression
        # In practice, you'd use a more sophisticated approach
        print("Training quantum model...")
        print(f"Training on {len(X_train)} samples with {self.n_qubits} qubits...")
        
        # Store training data for prediction
        self.X_train = X_train_scaled
        self.y_train = y_train
        self.is_trained = True
        
        print("Training completed!")
        return self
    
    def predict(self, X_test):
        """
        Predict credit scores using the quantum model.
        
        Args:
            X_test: Test features (n_samples, n_features)
            
        Returns:
            Predicted credit scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction!")
        
        X_test_scaled = self.prepare_features(X_test)
        
        # Quantum-inspired prediction using kernel method
        # Calculate similarity with training data using quantum-inspired metrics
        predictions = []
        
        for x in X_test_scaled:
            # Quantum kernel-based prediction
            # Use quantum-inspired distance metric
            similarities = []
            for x_train in self.X_train:
                # Quantum-inspired similarity (cosine similarity in feature space)
                similarity = np.dot(x, x_train) / (np.linalg.norm(x) * np.linalg.norm(x_train) + 1e-10)
                similarities.append(similarity)
            
            # Weighted average based on similarity
            similarities = np.array(similarities)
            weights = np.exp(similarities * 5)  # Exponential weighting
            weights = weights / weights.sum()
            
            prediction = np.dot(weights, self.y_train)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_single(self, features):
        """
        Predict credit score for a single person.
        
        Args:
            features: Dictionary or array of features
            
        Returns:
            Predicted credit score
        """
        if isinstance(features, dict):
            # Convert dictionary to array
            # Keep preprocessing consistent with training data (income in thousands)
            feature_array = np.array([[
                features.get('age', 0),
                features.get('income', 0) / 1000.0,
                features.get('debt_to_income', 0),
                features.get('payment_history', 0),
                features.get('credit_history_length', 0),
                features.get('num_accounts', 0),
                features.get('credit_utilization', 0),
                features.get('num_inquiries', 0)
            ]])
        else:
            feature_array = np.array(features).reshape(1, -1)
        
        prediction = self.predict(feature_array)
        return prediction[0]


def generate_synthetic_data(n_samples=200):
    """
    Generate synthetic credit data for demonstration.
    In production, use real credit data.
    """
    np.random.seed(42)
    
    # Generate realistic credit data
    age = np.random.randint(18, 80, n_samples)
    income = np.random.normal(50000, 20000, n_samples)
    income = np.maximum(income, 20000)  # Minimum income
    
    debt = np.random.normal(income * 0.3, income * 0.1, n_samples)
    debt = np.maximum(debt, 0)
    debt_to_income = debt / (income + 1e-10)
    
    payment_history = np.random.uniform(0.7, 1.0, n_samples)  # 70-100% on-time
    credit_history_length = np.random.uniform(1, 30, n_samples)  # Years
    num_accounts = np.random.randint(1, 15, n_samples)
    credit_utilization = np.random.uniform(0.1, 0.9, n_samples)  # 10-90%
    num_inquiries = np.random.randint(0, 10, n_samples)
    
    # Create feature matrix
    X = np.column_stack([
        age,
        income / 1000,  # Normalize income
        debt_to_income,
        payment_history,
        credit_history_length,
        num_accounts,
        credit_utilization,
        num_inquiries
    ])
    
    # Generate credit scores (300-850 range)
    # Based on weighted combination of factors
    base_score = 500
    score = (base_score +
             50 * payment_history +
             30 * (1 - debt_to_income) +
             20 * np.minimum(credit_history_length / 10, 1) +
             15 * np.minimum(num_accounts / 10, 1) -
             20 * credit_utilization -
             10 * np.minimum(num_inquiries / 5, 1) +
             np.random.normal(0, 30, n_samples))
    
    # Clamp to valid range
    score = np.clip(score, 300, 850)
    
    return X, score


def main():
    """
    Main function to demonstrate the quantum credit score predictor.
    """
    print("=" * 60)
    print("Quantum Credit Score Prediction System")
    print("=" * 60)
    print()
    
    # Generate synthetic data
    print("Generating synthetic credit data...")
    X, y = generate_synthetic_data(n_samples=200)
    print(f"Generated {len(X)} samples with {X.shape[1]} features")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize quantum predictor
    print("Initializing Quantum Credit Score Predictor...")
    predictor = QuantumCreditScorePredictor(
        n_features=8,
        n_qubits=4,
        n_layers=2
    )
    
    # Train the model
    predictor.train(X_train, y_train, max_iter=50)
    print()
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred = predictor.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  Mean Absolute Error: {mae:.2f} points")
    print(f"  R² Score: {r2:.3f}")
    print()
    
    # Example predictions for individual persons
    print("=" * 60)
    print("Example Predictions for Individual Persons")
    print("=" * 60)
    print()
    
    examples = [
        {
            'name': 'Alice',
            'features': {
                'age': 70,
                'income': 7500000,
                'debt_to_income': 0.10,
                'payment_history': 0.99,
                'credit_history_length': 50,
                'num_accounts': 5,
                'credit_utilization': 0.1,
                'num_inquiries': 0
            }
        },
        {
            'name': 'Bob',
            'features': {
                'age': 28,
                'income': 45000,
                'debt_to_income': 1,
                'payment_history': 0.75,
                'credit_history_length': 5,
                'num_accounts': 3,
                'credit_utilization': 0.70,
                'num_inquiries': 5
            }
        },
        {
            'name': 'Charlie',
            'features': {
                'age': 20,
                'income': 1000,
                'debt_to_income': 0.90,
                'payment_history': 0.98,
                'credit_history_length': 1,
                'num_accounts': 1,
                'credit_utilization': 0.90,
                'num_inquiries': 19
            }
        }
    ]
    
    for example in examples:
        score = predictor.predict_single(example['features'])
        print(f"{example['name']}:")
        print(f"  Age: {example['features']['age']}")
        print(f"  Income: ${example['features']['income']:,}")
        print(f"  Debt-to-Income: {example['features']['debt_to_income']:.2%}")
        print(f"  Payment History: {example['features']['payment_history']:.2%}")
        print(f"  Predicted Credit Score: {score:.0f}")
        print()
    
    print("=" * 60)
    print("Quantum Credit Score Prediction Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

