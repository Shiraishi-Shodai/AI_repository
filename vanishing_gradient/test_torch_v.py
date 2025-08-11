import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_v import NeuralNetwork # Assuming test file is in the same directory

# Set seeds for reproducibility in tests
torch.manual_seed(0)
np.random.seed(0)

class TestNeuralNetworkPyTorch(unittest.TestCase):

    def setUp(self):
        self.input_dim = 1
        self.out_dim = 1
        self.data_len = 50
        
        # Dummy data
        self.X_dummy = torch.randn(self.data_len, self.input_dim).float()
        self.y_dummy = torch.randn(self.data_len, self.out_dim).float()

    def test_model_architecture_simple(self):
        hidden_sizes = [10]
        model = NeuralNetwork(input_dim=self.input_dim, hidden_sizes=hidden_sizes, out_dim=self.out_dim)
        
        # Expect 2 layers in hidden_layers: Linear and ReLU
        self.assertEqual(len(model.hidden_layers), 2)
        self.assertIsInstance(model.hidden_layers[0], nn.Linear)
        self.assertIsInstance(model.hidden_layers[1], nn.ReLU)
        
        # Check dimensions of the linear layers
        self.assertEqual(model.hidden_layers[0].in_features, self.input_dim)
        self.assertEqual(model.hidden_layers[0].out_features, hidden_sizes[0])
        self.assertEqual(model.output_layer.in_features, hidden_sizes[0])
        self.assertEqual(model.output_layer.out_features, self.out_dim)

    def test_model_architecture_multiple_hidden(self):
        hidden_sizes = [5, 10, 5]
        model = NeuralNetwork(input_dim=self.input_dim, hidden_sizes=hidden_sizes, out_dim=self.out_dim)
        
        # Expect (Linear, ReLU) pairs for each hidden layer
        self.assertEqual(len(model.hidden_layers), len(hidden_sizes) * 2)
        
        # Check specific layer dimensions
        self.assertEqual(model.hidden_layers[0].in_features, self.input_dim)
        self.assertEqual(model.hidden_layers[0].out_features, hidden_sizes[0])
        
        self.assertEqual(model.hidden_layers[2].in_features, hidden_sizes[0]) # Second linear layer
        self.assertEqual(model.hidden_layers[2].out_features, hidden_sizes[1])
        
        self.assertEqual(model.hidden_layers[4].in_features, hidden_sizes[1]) # Third linear layer
        self.assertEqual(model.hidden_layers[4].out_features, hidden_sizes[2])
        
        self.assertEqual(model.output_layer.in_features, hidden_sizes[-1])
        self.assertEqual(model.output_layer.out_features, self.out_dim)

    def test_forward_pass_shape(self):
        hidden_sizes = [10, 20]
        model = NeuralNetwork(input_dim=self.input_dim, hidden_sizes=hidden_sizes, out_dim=self.out_dim)
        output = model(self.X_dummy)
        self.assertEqual(output.shape, (self.data_len, self.out_dim))
        self.assertTrue(torch.is_floating_point(output))

    def test_training_step(self):
        hidden_sizes = [5]
        model = NeuralNetwork(input_dim=self.input_dim, hidden_sizes=hidden_sizes, out_dim=self.out_dim)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Get initial parameter values
        initial_weight = model.hidden_layers[0].weight.clone().detach()
        
        # Perform one training step
        y_pred = model(self.X_dummy)
        loss = criterion(y_pred, self.y_dummy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check if loss is a scalar
        self.assertIsInstance(loss.item(), float)
        
        # Check if gradients were computed (at least for one parameter)
        self.assertIsNotNone(model.hidden_layers[0].weight.grad)
        
        # Check if parameters have changed
        self.assertFalse(torch.equal(initial_weight, model.hidden_layers[0].weight))

    def test_loss_reduction_over_epochs(self):
        # Create a simple linear dataset for easier learning
        X_train = torch.linspace(-1, 1, 100).reshape(-1, 1).float()
        y_train = (2 * X_train + 0.5 + torch.randn(100, 1) * 0.1).float() # y = 2x + 0.5 + noise

        model = NeuralNetwork(input_dim=1, hidden_sizes=[10], out_dim=1)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.05) # Higher LR for faster convergence

        initial_loss = criterion(model(X_train), y_train).item()
        
        epochs = 200 # Reduced epochs for test speed
        for epoch in range(epochs):
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        final_loss = criterion(model(X_train), y_train).item()
        
        self.assertLess(final_loss, initial_loss, "Loss should decrease after training")
        # self.assertLess(final_loss, 0.1, "Final loss should be low for simple data")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)