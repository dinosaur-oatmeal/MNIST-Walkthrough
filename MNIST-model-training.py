import openml
import numpy as np

# Loads dataset from openml and splits into train and test sets
def load_mnist():
    # Download MNIST dataset from OpenML
    mnist = openml.datasets.get_dataset(554)    # ID for MNIST

    # Extract data and labels from dataset
    x, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)

    # Normalize pixel values and reshape data
    x = x.to_numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y = y.astype(int)

    # Shuffle the dataset before splitting
    np.random.seed(42)
    #permutation = np.random.permutation(x.shape[0])
    #x, y = x[permutation], y[permutation]

    # Split data into training and testing sets
    x_train, x_test = x[:60000].T, x[60000:].T  # Shape: (784, 60000) and (784, 10000)
    y_train, y_test = y[:60000], y[60000:]

    # Verification of shapes
    '''
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    '''

    return x_train, y_train, x_test, y_test

# Initializes weights matrices and baises vectors
def initialize_parameters(input_size, hidden_size, output_size):
    # Random initialization of weights for input-to-hidden and hidden-to-output layers
    np.random.seed(42)

    '''
    Generate values from a standard normal distribution (mean 0, variance 1)
    Multiply values by 0.01 to keep weights small and help with training stability
    A row corresponds to the connections between all neurons in the first layer and
        a single neuron in the next layer (calculating activation for)
    A column corresponds to all weights coming out of a particular neuron in first layer
    '''
    weights = {
        'w1': np.random.randn(hidden_size, input_size) * np.sqrt(1. / input_size),  # Weight matrix from input to hidden
        'w2': np.random.randn(output_size, hidden_size) * np.sqrt(1. / hidden_size) # Weight matrix from hidden to output
    }

    '''
    Initialize bias vectors to be 0 all the way down
    '''
    biases = {
        'b1': np.zeros((hidden_size, 1)),   # Bias vector for hidden layer
        'b2': np.zeros((output_size, 1))    # Bias vector for output layer
    }

    return weights, biases

'''
Sigmoid function (demonstrate math behind it)
Squishes data to be between 0 and 1 for calculations
'''
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

'''
Sigmoid derivative (demonstrate math behind it)
Squishes data to be between 0 and 1 for calculations
'''
def sigmoid_derivative(a):
    return a * (1 - a)

'''
Transform raw scores into probabilities that sum to 1
softmax(z_j) = \frac{ e^{z_j} } { \sum_{k=1}^n * e^{z_k} }
'''
def softmax(z):
    '''
    subtract max value from z to ensure that the largest value in z becomes 0
        prevents large exponentials from occuring while preserving relative differences
    '''
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))

    # return probability distribution over the classes
    return exp_z / exp_z.sum(axis=0, keepdims=True)

'''
Calculate the loss during training
    (how far off the model's prediction is from the correct label)
    DOES NOT DO PREDICTION
Use cross-entropy loss for multi-class classification
    (popular because it penalizes incorrect predictions with large probabilities very harshly)

L = - \sum_{j = 1}^{C} * y_{j} * log(a_{2, j})
    C = # of classes
    y_{j} = true label (1 if correct, 0 o.w.)
    a_{2, j} = predicted probability for class j from softmax function

Full loss for batch data: average of the cross-entropy loss over all samples in the batch

Arguments:
    a2 -- output of the network after softmax (shape: output_size, number_of_samples)
    y -- true labels (shape: number_of_samples,)
    
Returns:
    loss -- cross-entropy loss
'''
def compute_loss(a2, y):

    # Number of samples
    m = y.shape[0]

    # small constant to prevent log(0)
    epsilon = 1e-15

    # Clip a2 to prevent log(0) leading to unknown behavior
    a2_clipped = np.clip(a2, epsilon, 1 - epsilon)

    '''
    Compute cross-entropy loss
    Single sample: L = -log(a_{2, j})
        j is the index of the correct class (true label)
    Then, find the average over batch
    '''
    log_probability = -np.log(a2_clipped[y, np.arange(m)])    # Log of correct class probability
    loss = np.sum(log_probability) / m                        # Average loss across all samples

    return loss

'''
Forwardpropagation

Take input data and pass through network to compute output and intermediate values
    Output is the predicted probabilities the network comes up with

    Input Layer -> Hidden Layer (Sigmoid activation) -> Output Layer (Softmax activation)

Arguments:
    x -- input data (shape: number_of_samples, input_size)
    weights -- dictionary containing the current weights w1 (input -> hidden) and w2 (hidden -> output)
    biases -- dictionary containing the current biases b1 (hidden) and b2 (output)

Returns:
    a2 -- activations at output layer (probability distribution over classes)
    cache -- dictionary containing z1, a1, z2, a2 needed for backpropagation
'''
def forward_propagation(x, weights, biases):

    # z1 is the pre-activation values for the hidden layer
    z1 = np.dot(weights['w1'], x) + biases['b1']

    # a1 is the Sigmoid activation for the hidden layer (squishes z1 between 0 and 1)
    a1 = sigmoid(z1)

    # z2 is the pre-activation values for the output layer
    z2 = np.dot(weights['w2'], a1) + biases['b2']

    # a2 is the softmax activation for the output layer (probability distribution function)
    a2 = softmax(z2)

    cache = {
        'z1' : z1,
        'a1' : a1,
        'z2' : z2,
        'a2' : a2
    }

    return a2, cache

'''
Backpropagation

Overall Goal

    Compute gradient of loss function with respect to each weight and bias
        Gradients tell us how to adjust weights and biases to minimize loss
            walk down the inverse gradient and find local minima of loss function
            As loss decreases, our model gets better at identifying numbers

    Compute error at output layer
        (how far the guess was compared to the actual label)
    Propagate error backward through the network to see how each weight and bias contributed to the error
    Use the errors to update the weights and biases via gradient descent

Gradients

    Gradients are the generalization of the derivative to function of multiple variables
        Vector that contains partial derivatives with respect to all variables of the function
        Tells us the direction of steepest ascent for the loss
            Gradient Descent: doing the inverse to decrease the loss by steepest descent

    Relies on the chain-rule in calculus to calculate gradients
        \frac{ \delta L }{ \delta W} and \frac{ \delta L }{ \delta b }
        L is loss function
        W is weight matrix
        b is bias vector

Arguments:
    x -- input data (shape: number_of_samples, input_size)
    y -- true labels (shape: number_of_samples,)
    cache -- dictionary containing z1, a1, z2, a2 (from forward propagation)
    weights -- dictionary containing w1 and w2
    
Returns:
    gradients -- dictionary containing the gradients with respect to w1, w2, b1, b2
'''
def back_propagation(x, y, cache, weights):

    # Number of samples
    m = y.shape[0]

    '''
    Retrieve cached values from forward propagation
        a1: activations at hidden layer (after applying sigmoid)
            These become the input for the next layer (output)
        a2: activations at output layer (probability distribution over classes)
    '''
    a1, a2 = cache['a1'], cache['a2']

    '''
    One-hot encode labels to be compatible with function
    Initialize matrix filled with 0s
    '''
    y_one_hot = np.zeros_like(a2)

    '''
    Generate array of indices corresponding to number of samples
    y = 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    '''
    y_one_hot[y, np.arange(m)] = 1

    # Step 1: Output Layer Error (Softmax + Cross-Entropy Derivative)
    dz2 = a2 - y_one_hot # Error at output layer (probability distribution - actual label)

    # Step 2: Gradient of weights and biases for the output layer
    dw2 = np.dot(dz2, a1.T) / m # Gradient of weights (dot product between error and hidden layer activations) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m # Gradient of biases (summation of errors for each output) / m

    # Step 3: Backpropagate the error to the hidden layer
    da1 = np.dot(weights['w2'].T, dz2)  # Error at hidden layer before applying Sigmoid derivative

    # Step 4: Sigmoid derivative to the error at the hidden layer
    dz1 = da1 * sigmoid_derivative(cache['a1'])

    # Step 5: Gradient of weights and biases for hidden layer
    dw1 = np.dot(dz1, x.T) / m  # Gradient loss with respect to weights connecting the input layer to hidden layer
    db1 = np.sum(dz1, axis=1, keepdims=True) / m    # Graident biases between input layer and hidden layer

    # Store gradients in a dictionary for later use
    gradients = {
        'dw1' : dw1,
        'db1' : db1,
        'dw2' : dw2,
        'db2' : db2
    }

    return gradients

'''
Update Parameters

Updates the weights and biases during learning using gradient descent
    subtract product of learning_rate and gradient from current weights
            moves the weights in the opposite direction of gradient, reducing loss

Arguments:
    weights -- dictionary containing the current weights w1 and w2
    biases -- dictionary containing the current biases b1 and b2
    gradients -- dictionary containing the gradients dw1, dw2, db1, db2 (from backpropagation)
    learning_rate -- the learning rate for gradient descent (a small positive number)
    
Returns:
    weights -- updated weights
    biases -- updated biases
'''
def update_parameters(weights, biases, gradients, learning_rate):

    # Update weights
    weights['w1'] -= learning_rate * gradients['dw1']
    weights['w2'] -= learning_rate * gradients['dw2']

    # Update biases
    biases['b1']  -= learning_rate * gradients['db1']
    biases['b2']  -= learning_rate * gradients['db2']
    return weights, biases

'''
Makes predictions based on the current state of the neural network

Arguments:
    x -- input data (shape: number_of_samples, input_size)
    weights -- dictionary containing the current weights w1 and w2
    biases -- dictionary containing the current biases b1 and b2

Returns:
    predictions -- array containing predicted labels for each input

'''
def predict(x, weights, biases):

    # Calculate output activations
    a2, _ = forward_propagation(x, weights, biases)

    # Array containing predicted labels for each input sample
    predictions = np.argmax(a2, axis=0)

    return predictions

'''
Computes the accuracy of predictions by comparing the predicted labels with the true labels

Arguments:
    y_true -- array containing the true labels for each input sample
    y_pred -- array containing the predicted labels for each input sample

Returns:
    fraction of correct predictions by finding the mean of a boolean array * 100
'''
def compute_accuracy(y_true, y_pred):

    # Fraction of correct predictions
    return np.mean(y_true == y_pred) * 100

'''
Training the neural network using gradient descent

Arguments:
    x_train -- input data (shape: input_size, number_of_samples)
    y_train -- true labels (shape: number_of_samples,)
    weights -- dictionary containing the initial weights w1 and w2
    biases -- dictionary containing the initial biases b1 and b2
    learning_rate -- learning rate for gradient descent
        Controls the step size when updating weights (keep small)
    num_epochs -- number of training epochs
        How many times the weights and biases are updated
    
Returns:
    weights -- updated weights after training
    biases -- updated biases after training
'''
def train_neural_network(x_train, y_train, weights, biases, learning_rate, num_epochs):

    # Loop through updates to weights and biases for each epoch
    for epoch in range(1, num_epochs + 1):

        # Step 1: Forward propagation
        a2, cache = forward_propagation(x_train, weights, biases)

        # Step 2: Compute loss (monitors performance)
        loss = compute_loss(a2, y_train)

        # Step 3: Backpropagation
        gradients = back_propagation(x_train, y_train, cache, weights)

        # Step 4: Update weights and biases using gradient descent
        weights, biases = update_parameters(weights, biases, gradients, learning_rate)

        # Compute training accuracy during training
        y_pred_train = np.argmax(a2, axis=0)
        accuracy_train = compute_accuracy(y_train, y_pred_train)

        # Print loss every 10 epochs to track progress
        if epoch % 1 == 0:
            print(f"Epoch {epoch}/{num_epochs} - Loss: {loss:.4f} - Training Accuracy: {accuracy_train:.2f}%")

    return weights, biases

'''
See model's performance on a dataset it's never seen

Arguments:
    x_test -- input data to test
    y_test -- true labels to test
    weights -- dictionary containing the initial weights w1 and w2
    biases -- dictionary containing the initial biases b1 and b2

Returns:
    accuracy -- percentage of how accurate the model is
'''
def evaluate_model(x_test, y_test, weights, biases):

    # Find predictions on the test data
    predictions = predict(x_test, weights, biases)

    # Calculate accuracy by comparing predicted labels to true labels
    accuracy = compute_accuracy(y_test, predictions)

    return accuracy

def main():
    
    # Step 1: Load the MNIST dataset
    x_train, y_train, x_test, y_test = load_mnist()

    # Step 2: Initialize parameters
    input_size = x_train.shape[0]   # Each image is 28x28 pixels = 784
    hidden_size = 128                # Number of neurons in the hidden layer
    output_size = 10                # 10 output neurons for the 10 digits (0-9)

    weights, biases = initialize_parameters(input_size, hidden_size, output_size)

    # Step 3: Set hyperparameters
    learning_rate = 0.1  # Small step size for gradient descent
    num_epochs = 10     # Number of training epochs

    # Step 4: Train the neural network
    weights, biases = train_neural_network(x_train, y_train, weights, biases, learning_rate, num_epochs)

    # Step 5: Evaluate the model
    test_accuracy = evaluate_model(x_test, y_test, weights, biases)
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()