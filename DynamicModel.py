import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import openml

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
    np.random.seed(42)
    weights = []
    biases = []
    for i in range(1, len(layer_dims)):
        # He initialization for ReLU
        w = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(2. / layer_dims[i - 1])
        b = np.zeros((layer_dims[i], 1))
        weights.append(w)
        biases.append(b)
    return weights, biases

# Activation functions
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
def forward_propagation(x, weights, biases, activation_func):
    activations = [x]
    zs = []
    for i in range(len(weights) - 1):
        z = np.dot(weights[i], activations[-1]) + biases[i]
        zs.append(z)
        a = activation_func(z)
        activations.append(a)
    z = np.dot(weights[-1], activations[-1]) + biases[-1]
    zs.append(z)
    a = softmax(z)
    activations.append(a)
    cache = {'activations': activations, 'zs': zs}
    return activations[-1], cache

# Backward propagation
def back_propagation(y, cache, weights, activation_derivative):
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
        delta = np.dot(weights[l + 1].T, delta) * activation_derivative(activations[l + 1])
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

# Train Neural Network
def train_neural_network(x_train, y_train, weights, biases, learning_rate, num_epochs,
                         activation_func, activation_derivative, batch_size=64):
    m = x_train.shape[1]
    for epoch in range(1, num_epochs + 1):
        permutation = np.random.permutation(m)
        x_shuffled = x_train[:, permutation]
        y_shuffled = y_train[permutation]
        for i in range(0, m, batch_size):
            x_batch = x_shuffled[:, i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            aL, cache = forward_propagation(x_batch, weights, biases, activation_func)
            gradients = back_propagation(y_batch, cache, weights, activation_derivative)
            weights, biases = update_parameters(weights, biases, gradients, learning_rate)
        if epoch % 1 == 0 or epoch == 1:
            aL_full, _ = forward_propagation(x_train, weights, biases, activation_func)
            loss = compute_loss(aL_full, y_train)
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}")
    return weights, biases

# Evaluate Model
def evaluate_model(x_test, y_test, weights, biases, activation_func):
    aL, _ = forward_propagation(x_test, weights, biases, activation_func)
    y_pred = np.argmax(aL, axis=0)
    accuracy = np.mean(y_pred == y_test)
    return accuracy

# Neural Network Visualization GUI Class
class NeuralNetworkGUI:
    """
    This class manages the Tkinter GUI for the Neural Network Visualization application.
    It displays the network layers, nodes, and allows users to interact with nodes to view weights and biases.
    """
    def __init__(self, root, weights, biases):
        """
        Initialize the NeuralNetworkGUI with the main Tkinter window.

        Parameters:
            root (tk.Tk): The main Tkinter window.
            weights (list): List of weight matrices for each layer.
            biases (list): List of bias vectors for each layer.
        """
        self.root = root
        self.root.title("Neural Network Visualization")
        self.weights = weights
        self.biases = biases
        self.layer_dims = [weights[0].shape[1]] + [w.shape[0] for w in weights]
        self.layers = len(self.layer_dims)
        self.node_radius = 10  # Reduced radius for better visibility
        self.node_spacing_x = 300  # Increased spacing between layers
        self.node_spacing_y = 15   # Adjusted spacing between nodes
        self.node_coords = []      # List of lists containing (x, y) tuples for each layer
        self.canvas_items = {}     # Dictionary to map node IDs to layer and node index
        self.setup_ui()
        self.draw_network()

    def setup_ui(self):
        """
        Set up the user interface, including canvas and scrollbars.
        """
        # Create a frame to hold the Canvas and Scrollbars
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Create Canvas where the network will be drawn
        self.canvas = tk.Canvas(self.canvas_frame, bg="white", width=1200, height=800, scrollregion=(0, 0, 3000, 2000))
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbars
        self.h_scroll = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scroll = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        # Bind events for panning
        self.canvas.bind("<ButtonPress-3>", self.on_pan_start)  # Right mouse button press
        self.canvas.bind("<B3-Motion>", self.on_pan_move)       # Right mouse button drag

    def on_pan_start(self, event):
        """
        Handle the start of a panning action.

        Parameters:
            event: The Tkinter event.
        """
        self.canvas.scan_mark(event.x, event.y)

    def on_pan_move(self, event):
        """
        Handle the movement during a panning action.

        Parameters:
            event: The Tkinter event.
        """
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def draw_network(self):
        """
        Draw the neural network on the canvas by positioning nodes and adding interactivity.
        """
        # Clear any existing drawings
        self.canvas.delete("all")
        self.node_coords = [[] for _ in range(self.layers)]
        self.canvas_items = {}

        # Assign positions to each layer
        for l in range(self.layers):
            num_nodes = self.layer_dims[l]
            layer_x = 100 + l * self.node_spacing_x
            # For input and hidden layers, show only top and bottom 1/8 nodes
            if l < self.layers -1:  # Input and Hidden Layers
                display_fraction = 1/4  # Show top 1/8 and bottom 1/8
                num_display = max(2, int(num_nodes * display_fraction / 2))  # Number per top/bottom
                top_nodes = list(range(num_display))
                bottom_nodes = list(range(num_nodes - num_display, num_nodes))
                display_nodes = top_nodes + bottom_nodes
                has_more = num_display * 2 < num_nodes
            else:
                # Output layer: show all nodes
                display_nodes = list(range(num_nodes))
                has_more = False

            # Calculate vertical spacing based on number of nodes to display
            display_count = len(display_nodes)
            if display_count > 1:
                layer_spacing_y = (800) / (display_count + 1)
            else:
                layer_spacing_y = 400
            for n in display_nodes:
                layer_y = 50 + (display_nodes.index(n) + 1) * layer_spacing_y
                self.node_coords[l].append((layer_x, layer_y))

        # Draw nodes
        for l in range(self.layers):
            for n, (x, y) in enumerate(self.node_coords[l]):
                color = self.get_node_color(l)
                node_id = self.canvas.create_oval(
                    x - self.node_radius, y - self.node_radius,
                    x + self.node_radius, y + self.node_radius,
                    fill=color,
                    outline="black",
                    width=1
                )
                self.canvas.tag_bind(node_id, '<Button-1>', lambda event, layer=l, node=n: self.on_node_click(layer, node))
                # Label output nodes
                if l == self.layers -1:
                    self.canvas.create_text(x, y, text=str(n), fill="white", font=("Arial", 8, "bold"))

        # Special Handling for Input Layer: Represent as a condensed grid with ellipses
        self.draw_input_layer()

        # Special Handling for Hidden Layers: Add ellipses if nodes are condensed
        self.draw_hidden_layers()

        # Initial Zoom: Center the view and zoom in
        self.canvas.scale("all", 0, 0, 1.0, 1.0)  # Adjust scaling if needed
        self.canvas.xview_moveto(1.00)
        self.canvas.yview_moveto(1.00)

    def get_node_color(self, layer):
        """
        Get the color based on the layer.

        Parameters:
            layer (int): The layer index.

        Returns:
            str: Color string.
        """
        if layer == 0:
            return "lightblue"
        elif layer == self.layers -1:
            return "lightgreen"
        else:
            return "orange"

    def draw_input_layer(self):
        """
        Draw the input layer as a condensed grid with ellipses indicating more nodes.
        """
        input_layer_index = 0
        grid_size = 28
        pixel_size = 10  # Size of each pixel rectangle
        x = self.node_coords[input_layer_index][0][0]
        y = self.node_coords[input_layer_index][0][1]
        # Draw top 1/8 and bottom 1/8 as separate grids
        fraction = 1/8
        num_top = max(1, int(grid_size * fraction))
        num_bottom = max(1, int(grid_size * fraction))
        # Ellipse to indicate more nodes
        ellipse_width = grid_size * pixel_size
        ellipse_height = 20
        self.canvas.create_oval(
            x - ellipse_width /2, y - grid_size /2 * pixel_size + num_top * pixel_size,
            x + ellipse_width /2, y - grid_size /2 * pixel_size + num_top * pixel_size + ellipse_height,
            fill="white",
            outline="black"
        )
        self.canvas.create_text(x, y - grid_size /2 * pixel_size + num_top * pixel_size + ellipse_height/2,
                                text="...", font=("Arial", 12, "bold"))
        # Bottom grid
        for i in range(num_bottom):
            for j in range(grid_size):
                px = x - (grid_size / 2) * pixel_size + j * pixel_size
                py = y + (grid_size / 2) * pixel_size - (num_bottom - i) * pixel_size
                rect = self.canvas.create_rectangle(
                    px, py, px + pixel_size, py + pixel_size,
                    fill="lightgray",
                    outline="black",
                    tags=("input_pixel", f"pixel_bottom_{i}_{j}")
                )
                # Bind click event to show pixel activation or weights
                self.canvas.tag_bind(rect, '<Button-1>', lambda event, row=i, col=j: self.on_input_pixel_click(row, col))

    def draw_hidden_layers(self):
        """
        Draw ellipses for hidden layers to indicate more nodes.
        """
        for l in range(1, self.layers -1):
            num_nodes = self.layer_dims[l]
            display_fraction = 1/4
            num_display = max(2, int(num_nodes * display_fraction / 2))  # Number per top/bottom
            if num_display * 2 < num_nodes:
                # Draw ellipses between top and bottom displayed nodes
                layer_x = 100 + l * self.node_spacing_x
                layer_y_top = self.node_coords[l][0][1] + self.node_spacing_y * (num_display)
                layer_y_bottom = self.node_coords[l][-1][1] - self.node_spacing_y * (num_display)
                ellipse_width = self.node_spacing_y * 5
                ellipse_height = 20
                self.canvas.create_oval(
                    layer_x - ellipse_width /2, layer_y_top,
                    layer_x + ellipse_width /2, layer_y_top + ellipse_height,
                    fill="white",
                    outline="black"
                )
                self.canvas.create_text(layer_x, layer_y_top + ellipse_height /2,
                                        text="...", font=("Arial", 12, "bold"))

    def on_input_pixel_click(self, row, col):
        """
        Handle the event when an input pixel is clicked.

        Parameters:
            row (int): Row index of the pixel.
            col (int): Column index of the pixel.
        """
        # For input pixels, display their position or related information
        messagebox.showinfo("Input Pixel", f"Pixel Position: ({row}, {col})\nClick on a hidden node to see corresponding weights.")

    def on_node_click(self, layer, node):
        """
        Handle the event when a node is clicked.

        Parameters:
            layer (int): Layer index of the node.
            node (int): Node index within the layer.
        """
        if layer == 0:
            # Input layer pixels are handled separately
            return
        elif layer == self.layers -1:
            # Output layer
            self.show_output_node_details(layer, node)
        else:
            # Hidden layer
            self.show_hidden_node_details(layer, node)

    def show_hidden_node_details(self, layer, node):
        """
        Display a window with details of a hidden node, including its weights and bias.

        Parameters:
            layer (int): Layer index of the node.
            node (int): Node index within the layer.
        """
        # Extract weights and bias for the selected hidden node
        # weights[layer -1] corresponds to weights connecting previous layer to current layer
        hidden_weights = self.weights[layer -1][node].reshape(28, 28)
        hidden_bias = self.biases[layer -1][node][0]

        # Create a new window
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Hidden Node {node} Details")

        # Display Bias
        bias_label = tk.Label(detail_window, text=f"Bias: {hidden_bias:.4f}", font=("Arial", 12, "bold"))
        bias_label.pack(pady=10)

        # Display Weight Grid using matplotlib
        fig, ax = plt.subplots(figsize=(4,4))
        cax = ax.imshow(hidden_weights, cmap='rwg')
        ax.set_title(f"Weights (28x28 Grid)")
        fig.colorbar(cax)
        plt.axis('off')

        # Embed the plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=detail_window)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def show_output_node_details(self, layer, node):
        """
        Display a window with details of an output node, including its weights and bias.

        Parameters:
            layer (int): Layer index of the node.
            node (int): Node index within the layer.
        """
        # Extract weights and bias for the selected output node
        # weights[layer -1] corresponds to weights connecting previous layer to current layer
        output_weights = self.weights[layer -1][node]
        output_bias = self.biases[layer -1][node][0]

        # Create a new window
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Output Node {node} Details")

        # Display Bias
        bias_label = tk.Label(detail_window, text=f"Bias: {output_bias:.4f}", font=("Arial", 12, "bold"))
        bias_label.pack(pady=10)

        # Display Weights
        weights_label = tk.Label(detail_window, text=f"Weights (from Hidden Layer to Output Node {node}):", font=("Arial", 10))
        weights_label.pack(pady=5)

        # Since output weights connect to hidden layer, visualize as bar graph
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(range(len(output_weights)), output_weights, color='purple')
        ax.set_title(f"Weights from Hidden Layer to Output Node {node}")
        ax.set_xlabel("Hidden Node Index")
        ax.set_ylabel("Weight Value")
        plt.tight_layout()

        # Embed the plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=detail_window)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

# Main function
def main():
    # Load data
    print("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist()
    print("MNIST dataset loaded.")

    # Define network architecture
    input_size = 28 * 28
    hidden_sizes = [128]
    output_size = 10
    layer_dims = [input_size] + hidden_sizes + [output_size]

    # Initialize parameters
    weights, biases = initialize_parameters(layer_dims)

    # Hyperparameters
    learning_rate = 0.1
    num_epochs = 10

    # Print network summary
    print("Neural Network Configuration:")
    print("-" * 50)
    print(f"Number of layers: {len(layer_dims)}")
    for i in range(len(layer_dims) -1):
        print(f"Layer {i+1} ({layer_dims[i]}) -> Layer {i+2} ({layer_dims[i+1]}): "
              f"Weights shape: {weights[i].shape}, Biases shape: {biases[i].shape}")
    print(f"Activation function: ReLU")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")
    print("-" * 50)

    # Train the neural network
    print("Starting training...")
    weights, biases = train_neural_network(
        x_train, y_train, weights, biases, learning_rate, num_epochs,
        relu, relu_derivative, batch_size=64
    )
    print("Training completed.")

    # Evaluate the model
    test_accuracy = evaluate_model(x_test, y_test, weights, biases, relu)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Launch GUI
    root = tk.Tk()
    gui = NeuralNetworkGUI(root, weights, biases)
    root.mainloop()

if __name__ == "__main__":
    main()
