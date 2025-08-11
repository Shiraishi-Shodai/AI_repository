import unittest
import numpy as np
import pandas as pd
from patern1 import NeuralNetwork # Assuming test file is in the same directory

class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        # Common setup for tests
        self.data_len = 50
        self.input_dim = 1
        self.hidden1 = 5
        self.hidden2 = 14
        self.hidden3 = 10
        self.out_dim = 1
        self.epochs = 100 # Reduced for faster tests
        self.lr = 0.001

        self.nn = NeuralNetwork(
            data_len=self.data_len,
            input_dim=self.input_dim,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            hidden3=self.hidden3,
            out_dim=self.out_dim,
            epochs=self.epochs,
            lr=self.lr
        )
        
        # Dummy data for testing
        self.X_dummy = np.random.rand(self.data_len, self.input_dim)
        self.y_dummy = np.random.rand(self.data_len, self.out_dim)

    def test_initialization(self):
        # Check if parameters are initialized with correct shapes
        params = self.nn.params
        self.assertIn("w1", params)
        self.assertIn("b1", params)
        self.assertIn("w2", params)
        self.assertIn("b2", params)
        self.assertIn("w3", params)
        self.assertIn("b3", params)
        self.assertIn("w4", params)
        self.assertIn("b4", params)

        self.assertEqual(params["w1"].shape, (self.input_dim, self.hidden1))
        self.assertEqual(params["b1"].shape, (1, self.hidden1))
        self.assertEqual(params["w2"].shape, (self.hidden1, self.hidden2))
        self.assertEqual(params["b2"].shape, (1, self.hidden2))
        self.assertEqual(params["w3"].shape, (self.hidden2, self.hidden3))
        self.assertEqual(params["b3"].shape, (1, self.hidden3))
        self.assertEqual(params["w4"].shape, (self.hidden3, self.out_dim))
        self.assertEqual(params["b4"].shape, (1, self.out_dim))

        # Check if biases are initialized to zeros
        self.assertTrue(np.all(params["b1"] == 0))
        self.assertTrue(np.all(params["b2"] == 0))
        self.assertTrue(np.all(params["b3"] == 0))
        self.assertTrue(np.all(params["b4"] == 0))

    def test_loss_func(self):
        y_pred = np.array([[1.0], [2.0], [3.0]])
        y_true = np.array([[1.5], [2.0], [2.5]])
        # Expected loss: 0.5 * ((0.5)^2 + (0)^2 + (-0.5)^2) / 3 = 0.5 * (0.25 + 0 + 0.25) / 3 = 0.25 / 3 = 0.08333...
        expected_loss = np.mean(0.5 * ((y_pred - y_true)**2))
        calculated_loss = self.nn.loss_func(y_pred, y_true)
        self.assertAlmostEqual(calculated_loss, expected_loss)

    def test_ReLU(self):
        u = np.array([[-1.0, 0.0, 1.0], [-2.0, 3.0, 0.5]])
        expected_relu = np.array([[0.0, 0.0, 1.0], [0.0, 3.0, 0.5]])
        self.assertTrue(np.array_equal(self.nn.ReLU(u), expected_relu))

    def test_dReLU(self):
        u = np.array([[-1.0, 0.0, 1.0], [-2.0, 3.0, 0.5]])
        dz = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        # Expected dReLU:
        # u > 0: 1, u <= 0: 0
        # [[0, 0, 1], [0, 1, 1]] * dz
        expected_drelu = np.array([[0.0, 0.0, 0.3], [0.0, 0.5, 0.6]])
        self.assertTrue(np.array_equal(self.nn.dReLU(u, dz), expected_drelu))

    def test_forward_pass_shapes(self):
        # Perform a forward pass with dummy data
        output = self.nn.foward(self.X_dummy)
        self.assertEqual(output.shape, (self.data_len, self.out_dim))
        
        # Check intermediate shapes
        self.assertEqual(self.nn.u1.shape, (self.data_len, self.hidden1))
        self.assertEqual(self.nn.z1.shape, (self.data_len, self.hidden1))
        self.assertEqual(self.nn.u2.shape, (self.data_len, self.hidden2))
        self.assertEqual(self.nn.z2.shape, (self.data_len, self.hidden2))
        self.assertEqual(self.nn.u3.shape, (self.data_len, self.hidden3))
        self.assertEqual(self.nn.z3.shape, (self.data_len, self.hidden3))
        self.assertEqual(self.nn.u4.shape, (self.data_len, self.out_dim))
        self.assertEqual(self.nn.z4.shape, (self.data_len, self.out_dim))

    def test_backward_pass_shapes(self):
        # First, perform a forward pass to populate intermediate values
        y_pred = self.nn.foward(self.X_dummy)
        
        # Then, perform a backward pass
        grads = self.nn.backward(y_pred, self.y_dummy)
        
        # Check shapes of gradients
        self.assertEqual(grads["dw1"].shape, (self.input_dim, self.hidden1))
        self.assertEqual(grads["db1"].shape, (1, self.hidden1))
        self.assertEqual(grads["dw2"].shape, (self.hidden1, self.hidden2))
        self.assertEqual(grads["db2"].shape, (1, self.hidden2))
        self.assertEqual(grads["dw3"].shape, (self.hidden2, self.hidden3))
        self.assertEqual(grads["db3"].shape, (1, self.hidden3))
        self.assertEqual(grads["dw4"].shape, (self.hidden3, self.out_dim))
        self.assertEqual(grads["db4"].shape, (1, self.out_dim))

    def test_fit_loss_reduction(self):
        # Create a simple linear dataset for easier learning
        X_train = np.linspace(-1, 1, 100).reshape(-1, 1)
        y_train = 2 * X_train + 0.5 + np.random.randn(100, 1) * 0.1 # y = 2x + 0.5 + noise

        # Re-initialize NN with specific parameters for this test
        test_nn = NeuralNetwork(
            data_len=X_train.shape[0],
            input_dim=X_train.shape[1],
            hidden1=5, hidden2=5, hidden3=5, # Simpler network
            out_dim=1,
            epochs=500, # More epochs for convergence
            lr=0.01 # Higher learning rate
        )

        initial_loss = test_nn.loss_func(test_nn.foward(X_train), y_train)
        test_nn.fit(X_train, y_train)
        final_loss = test_nn.loss_func(test_nn.foward(X_train), y_train)
        
        self.assertLess(final_loss, initial_loss, "Loss should decrease after training")
        # Optionally, assert that loss is below a certain threshold if convergence is expected
        # self.assertLess(final_loss, 0.1, "Final loss should be low")

    def test_predict(self):
        # Predict should just perform a forward pass
        predictions = self.nn.predict(self.X_dummy)
        self.assertEqual(predictions.shape, (self.data_len, self.out_dim))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)