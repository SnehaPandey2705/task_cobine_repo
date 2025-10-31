# Manual Backpropagation for a 2-Layer Binary Classifier

This Task implements a **two-layer neural network** for **binary classification** using **manual backpropagation** (no autograd or ML libraries).  
The network uses only **NumPy** and is built completely from scratch for educational clarity.

---

## Architecture Overview

- **Input:** 3-dimensional vector `x ∈ R³`
- **Hidden Layer:** 4 neurons with **ReLU** activation
- **Output Layer:** 1 neuron with **Sigmoid** activation
- **Loss:** Binary Cross-Entropy (BCE)
- **Optimization:** Stochastic Gradient Descent (SGD)

---

## Network Flow

1. **Forward Pass:**  
   Computes activations, predicted probability, and loss.

2. **Backward Pass:**  
   Derives analytical gradients for all parameters (`W1`, `b1`, `W2`, `b2`).

3. **Gradient Check:**  
   Verifies correctness of backpropagation using finite difference approximation.

4. **SGD Update:**  
   Updates parameters using computed gradients.

---
## How to run
Run "back_propagation.py" python file and you will receive the output in terminal.

