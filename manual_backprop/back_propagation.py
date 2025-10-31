import numpy as np

# ------------------------------- Utility fns ------------------------------- #
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(sig):
    # derivative of sigmoid w.r.t. pre-activation z given sigmoid(z)=sig
    return sig * (1.0 - sig)

def relu(x):
    return np.maximum(0.0, x)

def relu_derivative(z):
    # derivative of ReLU wrt pre-activation z
    return (z > 0.0).astype(float)

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-10
    y_pred_clip = np.clip(y_pred, eps, 1.0 - eps)
    return - (y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))

# ------------------------------- Forward pass ------------------------------- #
def forward(x, y, params):
    """
    Forward pass. Returns scalar loss and cache dict.
    Shapes are inferred from params, so this is robust to hidden size.
    """
    W1 = params["W1"]  
    b1 = params["b1"]  
    W2 = params["W2"]  
    b2 = params["b2"]  

    # ensure arrays are 1D where expected
    x = np.asarray(x).reshape(-1)   
    b1 = np.asarray(b1).reshape(-1)
    b2 = np.asarray(b2).reshape(-1)

    # Hidden layer
    z1 = W1.dot(x) + b1            
    a1 = relu(z1)                  

    # Output layer
    z2 = W2.dot(a1) + b2           
    a2 = sigmoid(z2)               

    # Loss (return as numpy scalar or 0-d array)
    loss = binary_cross_entropy(y, a2)  

    cache = {"x": x, "y": np.asarray(y).reshape(-1), "z1": z1, "a1": a1, "z2": z2, "a2": a2}
    return loss, cache

# ------------------------------- Backward pass ------------------------------- #
def backward(params, cache):
    """
    Returns gradients dict with shapes matching params.
    dW1: (H, 3), db1: (H,), dW2: (1, H), db2: (1,)
    """
    W1 = params["W1"]
    W2 = params["W2"]

    x = cache["x"]          
    y = cache["y"]          
    z1 = cache["z1"]        
    a1 = cache["a1"]       
    z2 = cache["z2"]       
    a2 = cache["a2"]       

    H = W1.shape[0]
    assert W1.shape == (H, x.shape[0]), f"W1 shape mismatch: {W1.shape}"
    assert W2.shape == (1, H), f"W2 shape mismatch: {W2.shape}"

    # dL/da2 for BCE with single sample: -(y/a2) + (1-y)/(1-a2)
    # ensure a2 is scalar for computation
    a2_s = np.squeeze(a2)   
    y_s = np.squeeze(y)     

    dL_da2 = - (y_s / (a2_s + 1e-12)) + ((1.0 - y_s) / (1.0 - a2_s + 1e-12))  
    da2_dz2 = sigmoid_derivative(a2_s)  
    dL_dz2 = dL_da2 * da2_dz2           

    # Gradients for output layer
    # dW2 = dL/dz2 * a1^T -> shape (1, H)
    dW2 = (dL_dz2 * a1).reshape(1, -1)
    db2 = np.array([dL_dz2]).reshape(1,)

    # Hidden layer gradients
    # dL/da1 = W2^T * dL/dz2 (W2.T shape (H,1) times scalar)
    dL_da1 = (W2.T * dL_dz2).reshape(H,)   
    da1_dz1 = relu_derivative(z1)          
    dL_dz1 = dL_da1 * da1_dz1              

    # dW1 = dL_dz1 (H,) outer x (3,) -> (H,3)
    dW1 = dL_dz1.reshape(-1, 1) @ x.reshape(1, -1)
    db1 = dL_dz1.reshape(-1,)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

# ------------------------------- SGD update ------------------------------- #

def sgd_update(params, grads, lr=0.1):

    params["W1"] = params["W1"] - lr * grads["dW1"]
    params["b1"] = params["b1"] - lr * grads["db1"]
    params["W2"] = params["W2"] - lr * grads["dW2"]
    params["b2"] = params["b2"] - lr * grads["db2"]
    return params

# ------------------------------- Gradient checking ------------------------------- #

def gradient_check(x, y, params, grads, eps=1e-5):
    """
    Numerically estimate gradients for W1 and W2 using symmetric difference.
    Uses copies of params for +eps and -eps to avoid in-place issues.
    Returns max absolute difference between analytical and numerical grads for W1 and W2.
    """
    numeric = {}

    for key in ("W1", "W2"):
        param = params[key]
        num_grad = np.zeros_like(param, dtype=float)
        it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = param[idx]

            params_plus = {k: v.copy() for k, v in params.items()}
            params_minus = {k: v.copy() for k, v in params.items()}

            params_plus[key][idx] = orig + eps
            params_minus[key][idx] = orig - eps

            loss_plus, _ = forward(x, y, params_plus)
            loss_minus, _ = forward(x, y, params_minus)

            loss_plus = float(np.squeeze(loss_plus))
            loss_minus = float(np.squeeze(loss_minus))

            num_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
            it.iternext()

        numeric[key] = num_grad

    assert grads["dW1"].shape == params["W1"].shape, f"dW1 shape mismatch {grads['dW1'].shape} vs {params['W1'].shape}"
    assert grads["dW2"].shape == params["W2"].shape, f"dW2 shape mismatch {grads['dW2'].shape} vs {params['W2'].shape}"

    diff_W1 = np.max(np.abs(grads["dW1"] - numeric["W1"]))
    diff_W2 = np.max(np.abs(grads["dW2"] - numeric["W2"]))
    return diff_W1, diff_W2

# ------------------------------- Main script ------------------------------- #
if __name__ == "__main__":
    np.random.seed(42)

    x = np.array([0.2, -0.4, 0.1])   
    y = np.array([1.0])              

    hidden_neurons = 4
    params = {
        "W1": np.random.RandomState(42).randn(hidden_neurons, x.shape[0]) * 0.1,  
        "b1": np.zeros(hidden_neurons),                                           
        "W2": np.random.RandomState(42).randn(1, hidden_neurons) * 0.1,           
        "b2": np.zeros(1),                                                       
    }

    loss, cache = forward(x, y, params)
    print("Initial loss:", np.squeeze(loss))

    grads = backward(params, cache)
    print("\nAnalytical gradients:")
    print("dW1:\n", grads["dW1"])
    print("dW2:\n", grads["dW2"])

    diff_W1, diff_W2 = gradient_check(x, y, params, grads, eps=1e-5)
    print(f"\nMax abs difference (W1): {diff_W1:.10f}")
    print(f"Max abs difference (W2): {diff_W2:.10f}")

    params = sgd_update(params, grads, lr=0.1)
    print("\nUpdated W1:\n", params["W1"])
    print("Updated W2:\n", params["W2"])

    new_loss, _ = forward(x, y, params)
    print("\nFinal loss after update:", np.squeeze(new_loss))
    print("Loss decreased:", float(np.squeeze(new_loss)) < float(np.squeeze(loss)))
