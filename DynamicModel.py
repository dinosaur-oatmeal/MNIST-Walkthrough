import openml
import numpy as np


# Function to load MNIST dataset
def load_mnist():
    # Download MNIST dataset from OpenML
    mnist = openml.datasets.get_dataset(554)    # ID for MNIST

    # Extract data and labels from dataset
    x, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)

    # Normalize pixel values and reshape data
    x = x.to_numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y = y.astype(int)

    # Manually split data into training and testing sets
    x_train, x_test = x[:60000].T, x[60000:].T  # Transpose to make (784, 60000)
    y_train, y_test = y[:60000], y[60000:]

    return x_train, y_train, x_test, y_test

# Initialize parameters
def initialize_parameters(layer_dims):

    # Arrays to store our list of weights and biases
    weights = []
    biases = []

    # Loop through number of layers for initialization
        # Weights should be random variables
        # Biases should be set to 0
    for i in range(1, len(layer_dims)):

        # He initialization
        w = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(2. / layer_dims[i - 1])
        b = np.zeros((layer_dims[i], 1))
        weights.append(w)
        biases.append(b)

    return weights, biases

# Activation functions
def sigmoid(z):

    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):

    return a * (1 - a)

def relu(z):

    return np.maximum(0, z)

def relu_derivative(a):

    return (a > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))

    return exp_z / exp_z.sum(axis=0, keepdims=True)

# Compute loss
def compute_loss(aL, y):

    m = y.shape[0]
    epsilon = 1e-15
    aL_clipped = np.clip(aL, epsilon, 1 - epsilon)
    log_probs = -np.log(aL_clipped[y, np.arange(m)])
    loss = np.sum(log_probs) / m

    return loss

# Forward propagation
def forward_propagation(x, weights, biases):

    activations = [x]
    zs = []

    for i in range(len(weights) - 1):

        z = np.dot(weights[i], activations[-1]) + biases[i]
        zs.append(z)

        a = relu(z)
        activations.append(a)

    z = np.dot(weights[-1], activations[-1]) + biases[-1]
    zs.append(z)
    a = softmax(z)
    activations.append(a)

    cache = {'activations': activations, 'zs': zs}

    return activations[-1], cache

# Backward propagation
def back_propagation(y, cache, weights):

    m = y.shape[0]
    activations = cache['activations']

    gradients_w = []
    gradients_b = []

    y_one_hot = np.zeros_like(activations[-1])
    y_one_hot[y, np.arange(m)] = 1

    delta = activations[-1] - y_one_hot
    dw = np.dot(delta, activations[-2].T) / m
    db = np.sum(delta, axis=1, keepdims=True) / m

    gradients_w.append(dw)
    gradients_b.append(db)

    for l in range(len(weights) - 2, -1, -1):

        delta = np.dot(weights[l + 1].T, delta) * relu_derivative(activations[l + 1])

        dw = np.dot(delta, activations[l].T) / m
        db = np.sum(delta, axis=1, keepdims=True) / m

        gradients_w.insert(0, dw)
        gradients_b.insert(0, db)

    gradients = {'dw': gradients_w, 'db': gradients_b}

    return gradients

# Update parameters
def update_parameters(weights, biases, gradients, learning_rate):

    for i in range(len(weights)):

        weights[i] -= learning_rate * gradients['dw'][i]
        biases[i] -= learning_rate * gradients['db'][i]

    return weights, biases

def predict(x, weights, biases):
    
    # Calculate output activations
    a2, _ = forward_propagation(x, weights, biases)

    # Array containing predicted labels for each input sample
    predictions = np.argmax(a2, axis=0)

    return predictions

def compute_accuracy(y_true, y_pred):
    
    return np.mean(y_true == y_pred) * 100

# Train Neural Network
def train_neural_network(x_train, y_train, weights, biases, learning_rate, num_epochs, batch_size, x_val=None, y_val=None):
    
    m = x_train.shape[1]
    loss_history = []
    accuracy_history = []

    for epoch in range(1, num_epochs + 1):
        # Shuffle the training data
        permutation = np.random.permutation(m)
        x_train_shuffled = x_train[:, permutation]
        y_train_shuffled = y_train[permutation]

        for i in range(0, m, batch_size):
            x_batch = x_train_shuffled[:, i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            # Forward propagation
            aL, cache = forward_propagation(x_batch, weights, biases)

            # Backward propagation
            gradients = back_propagation(y_batch, cache, weights)

            # Update parameters
            weights, biases = update_parameters(weights, biases, gradients, learning_rate)

        # Compute loss after each epoch
        aL_full, _ = forward_propagation(x_train, weights, biases)
        loss = compute_loss(aL_full, y_train)
        loss_history.append(loss)

        # Compute accuracy
        if x_val is not None and y_val is not None:
            # Validation accuracy
            y_pred_val = predict(x_val, weights, biases)
            accuracy = compute_accuracy(y_val, y_pred_val)
            accuracy_history.append(accuracy)
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
        else:
            # Training accuracy
            y_pred_train = predict(x_train, weights, biases)
            accuracy = compute_accuracy(y_train, y_pred_train)
            accuracy_history.append(accuracy)
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}, Training Accuracy: {accuracy:.2f}%")

    return weights, biases, loss_history, accuracy_history

# Evaluate Model
def evaluate_model(x_test, y_test, weights, biases):

    aL, _ = forward_propagation(x_test, weights, biases)

    y_pred = np.argmax(aL, axis=0)

    accuracy = np.mean(y_pred == y_test)

    return accuracy * 100

# Main function
def main():

    # Load data
    print("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist()
    print("MNIST dataset loaded.")

    # Define network architecture
    input_size = 28 * 28
    hidden_sizes = [32, 16]
    output_size = 10
    layer_dims = [input_size] + hidden_sizes + [output_size]

    # Initialize parameters
    weights, biases = initialize_parameters(layer_dims)

    # Training hyperparameters
    learning_rate = 0.01
    num_epochs = 10
    activation_function = relu
    activation_function_derivative = relu_derivative

    # Print network summary
    print("Neural Network Configuration:")
    print("-" * 30)
    print(f"Number of layers: {len(layer_dims)}")
    for i in range(len(layer_dims) -1):
        print(f"Layer {i+1} ({layer_dims[i]}) -> Layer {i+2} ({layer_dims[i+1]}): "
              f"Weights shape: {weights[i].shape}, Biases shape: {biases[i].shape}")
    print(f"Activation function: {activation_function.__name__}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")
    print("-" * 30)

    # Train the neural network
    print("Starting training...")
    weights, biases, loss_history, accuracy_history = train_neural_network(
        x_train, y_train, weights, biases, learning_rate, num_epochs, batch_size=64
    )
    print("Training completed.")

    # Evaluate the model
    test_accuracy = evaluate_model(x_test, y_test, weights, biases)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
