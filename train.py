import numpy as np
import pandas as pd
import pickle

class DenseLayer:
    def __init__(self, input_dim, output_dim, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Initialize weights and biases
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros((1, output_dim))
        
        # Adam optimizer parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Adam optimizer moment variables
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)
        
        # Time step for Adam
        self.t = 0

    def forward(self, X, training=True):
        # Linear forward pass
        self.input = X
        self.output = np.dot(X, self.weights) + self.biases
        return self.output

    def backward(self, d_output):
        # Calculate gradients
        self.d_weights = np.dot(self.input.T, d_output)
        self.d_biases = np.sum(d_output, axis=0, keepdims=True)
        self.d_input = np.dot(d_output, self.weights.T)
        
        self.update_params()
        return self.d_input

    def update_params(self):
        # Increment time step
        self.t += 1

        # Update weights with Adam
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * self.d_weights
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (self.d_weights ** 2)

        # Bias correction for moment estimates
        m_weights_corrected = self.m_weights / (1 - self.beta1 ** self.t)
        v_weights_corrected = self.v_weights / (1 - self.beta2 ** self.t)

        # Apply updates
        self.weights -= self.learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + self.epsilon)

        # Update biases with Adam
        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * self.d_biases
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (self.d_biases ** 2)

        # Bias correction for moment estimates
        m_biases_corrected = self.m_biases / (1 - self.beta1 ** self.t)
        v_biases_corrected = self.v_biases / (1 - self.beta2 ** self.t)

        # Apply updates
        self.biases -= self.learning_rate * m_biases_corrected / (np.sqrt(v_biases_corrected) + self.epsilon)


class CrossEntropyLoss:
    def forward(self, Z, training=True):
        # Apply softmax activation
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Stabilize by subtracting max
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def get_gradient(self, predictions, labels):
        # Gradient of cross-entropy loss w.r.t. softmax input
        m = labels.shape[0]
        grad = predictions.copy()
        grad[range(m), labels] -= 1
        grad = grad / m
        return grad
      
    def backward(self, d_output):
      return d_output
      
    def loss(self, predictions, labels):
      """
      Calculate the cross-entropy loss between predictions and labels.
      
      Parameters
      ----------
      predictions : array, shape (m, n)
        The output of the softmax layer.
      labels : array, shape (m,)
        The true labels.
      
      Returns
      -------
      loss : float
        The cross-entropy loss.
      """
      m = labels.shape[0]
      log_likelihood = -np.log(predictions[range(m), labels])
      return np.sum(log_likelihood) / m

      
class ReLU:
    def forward(self, Z, training=True):
        # Store input for backward pass
        self.input = Z
        # Apply ReLU activation
        return np.maximum(0, Z)

    def backward(self, d_output):
        # Gradient of ReLU is 1 for positive input, 0 otherwise
        dZ = d_output * (self.input > 0)
        return dZ

    
def accuracy(predictions, labels):
  """
  Calculate the accuracy of predictions.
  
  Parameters
  ----------
  predictions : array, shape (m, n)
    The output of the softmax layer.
  labels : array, shape (m,)
    The true labels.
  
  Returns
  -------
  accuracy : float
    The accuracy.
  """
  predicted_labels = np.argmax(predictions, axis=1)
  accuracy = np.mean(predicted_labels == labels)
  return accuracy

class BatchNormalization:
    def __init__(self, input_dim, momentum=0.9, epsilon=1e-5):
        """
        Initialize the Batch Normalization layer.
        
        Parameters
        ----------
        input_dim : int
            The number of features (input dimensions).
        momentum : float
            The momentum for the moving average of the mean and variance.
        epsilon : float
            A small value to avoid division by zero.
        """
        self.input_dim = input_dim
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Parameters for scaling and shifting
        self.gamma = np.ones(input_dim)  # Scale factor
        self.beta = np.zeros(input_dim)   # Shift factor
        
        # Running averages
        self.running_mean = np.zeros(input_dim)
        self.running_var = np.ones(input_dim)
        
        # Cache for backward pass
        self.cache = None

    def forward(self, X, training=True):
        """
        Forward pass for Batch Normalization.
        
        Parameters
        ----------
        X : array, shape (batch_size, input_dim)
            The input data.
        training : bool
            Whether the model is in training mode.
        
        Returns
        -------
        output : array, same shape as X
            The batch-normalized output.
        """
        if training:
            # Calculate mean and variance for the current batch
            batch_mean = np.mean(X, axis=0)
            batch_var = np.var(X, axis=0)
            
            # Normalize the batch
            X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
            # Scale and shift
            output = self.gamma * X_norm + self.beta
            
            # Update running averages
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Cache for backward pass
            self.cache = (X, X_norm, batch_mean, batch_var)
        else:
            # During inference, use running averages
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            output = self.gamma * X_norm + self.beta
        
        return output

    def backward(self, d_output):
        """
        Backward pass for Batch Normalization.
        
        Parameters
        ----------
        d_output : array, shape (batch_size, input_dim)
            Gradient from the next layer.
        
        Returns
        -------
        d_input : array, same shape as d_output
            Gradient to propagate to the previous layer.
        """
        X, X_norm, batch_mean, batch_var = self.cache
        m = X.shape[0]

        # Compute gradients
        dX_norm = d_output * self.gamma
        dgamma = np.sum(d_output * X_norm, axis=0)
        dbeta = np.sum(d_output, axis=0)

        dvar = np.sum(dX_norm * (X - batch_mean) * -0.5 * (batch_var + self.epsilon) ** (-1.5), axis=0)
        dmean = np.sum(dX_norm * -1 / np.sqrt(batch_var + self.epsilon), axis=0) + dvar * np.mean(-2 * (X - batch_mean), axis=0)

        d_input = (dX_norm / np.sqrt(batch_var + self.epsilon)) + (dvar * 2 * (X - batch_mean) / m) + (dmean / m)
        
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        
        return d_input

class Dropout:
    def __init__(self, dropout_rate=0.5):
        """
        Initialize the dropout layer with the given dropout rate.
        
        Parameters
        ----------
        dropout_rate : float
            The fraction of input units to drop during training.
        """
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, X, training=True):
        """
        Apply dropout to the input during training.
        
        Parameters
        ----------
        X : array, shape (batch_size, input_dim)
            The input data to apply dropout to.
        training : bool
            Whether the model is in training mode.
        
        Returns
        -------
        output : array, same shape as X
            The output after applying dropout (during training).
        """
        if training:
            self.mask = (np.random.rand(*X.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            return X * self.mask
        else:
            return X

    def backward(self, d_output):
        """
        Backward pass for dropout layer.
        
        Parameters
        ----------
        d_output : array, shape (batch_size, input_dim)
            Gradient from the next layer.
        
        Returns
        -------
        d_input : array, same shape as d_output
            Gradient to propagate to the previous layer.
        """
        return d_output * self.mask

class Model:
    def __init__(self, layers):
        """
        Initialize the model with layers and activations.
        
        Parameters
        ----------
        layers : list
            List of layer objects.
        activations : list
            List of activation objects.
        """
        self.layers = layers
        self.training = True
    
    def set_train(self, training):
        self.training = training

    def forward(self, X):
        """
        Perform forward pass through all layers and activations.
        
        Parameters
        ----------
        X : array
            Input data.
            
        Returns
        -------
        array
            The output after the forward pass.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, d_output):
        """
        Perform backward pass through all layers and activations.
        
        Parameters
        ----------
        d_output : array
            Gradient of the loss with respect to the output.
            
        Returns
        -------
        array
            The gradient of the loss with respect to the input data.
        """
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output)
        return d_output
      
def save_model_parameters(model, filename):
    """
    Save model parameters to a file using pickle.
    
    Parameters
    ----------
    model : Model
        The model to save parameters for.
    filename : str
        The name of the file to save parameters.
    """
    parameters = {}
    
    for i, layer in enumerate(model.layers):
        if isinstance(layer, DenseLayer):
            parameters[f'dense_layer_{i}_weights'] = layer.weights
            parameters[f'dense_layer_{i}_biases'] = layer.biases
        elif isinstance(layer, BatchNormalization):
            parameters[f'batch_norm_layer_{i}_gamma'] = layer.gamma
            parameters[f'batch_norm_layer_{i}_beta'] = layer.beta
    
    with open(filename, 'wb') as file:
        pickle.dump(parameters, file)
    print(f"Model parameters saved to {filename}")


def load_model_parameters(model, filename):
    """
    Load model parameters from a file using pickle.
    
    Parameters
    ----------
    model : Model
        The model to load parameters for.
    filename : str
        The name of the file to load parameters from.
    """
    with open(filename, 'rb') as file:
        parameters = pickle.load(file)

    for i, layer in enumerate(model.layers):
        if isinstance(layer, DenseLayer):
            layer.weights = parameters[f'dense_layer_{i}_weights']
            layer.biases = parameters[f'dense_layer_{i}_biases']
        elif isinstance(layer, BatchNormalization):
            layer.gamma = parameters[f'batch_norm_layer_{i}_gamma']
            layer.beta = parameters[f'batch_norm_layer_{i}_beta']
    
    print(f"Model parameters loaded from {filename}")


if __name__ == "__main__":
  np.random.seed(42)

  # Hyperparameters
  input_dim = 784 
  output_dim = 10
  hidden = 256
  learning_rate = 0.001
  epochs = 30

  # Create a Dense layer and Softmax activation layer
  CEloss = CrossEntropyLoss()
  model = Model([
    DenseLayer(input_dim, hidden, learning_rate),
    BatchNormalization(hidden),
    Dropout(0.1),
    ReLU(),
    DenseLayer(hidden, output_dim, learning_rate),
    CEloss
  ])
  
  # loading data
  train = pd.read_csv('fashion-mnist_train.csv')
  test = pd.read_csv('fashion-mnist_test.csv')
  
  train_x = train.drop(['label'], axis=1)
  train_y = train['label']
  test_x = test.drop(['label'], axis=1)
  test_y = test['label']
  
  batch_size = 16384

  for epoch in range(epochs):
    epoch_loss = []
    epoch_acc = []
    for index in range(0, train_x.shape[0], batch_size):
      batch_x = train_x.iloc[index:index + batch_size]
      batch_y = train_y.iloc[index:index + batch_size]
      
      # Forward pass
      predictions = model.forward(train_x)
      
      epoch_loss.append(CEloss.loss(predictions, train_y))
      epoch_acc.append(accuracy(predictions, train_y))

      # Backward pass
      d_loss = CEloss.get_gradient(predictions, train_y)
      d_loss = model.backward(d_loss)
      
    print(f"Epoch {epoch + 1}, Loss: {sum(epoch_loss)/len(epoch_loss):.4f}, Acc: {sum(epoch_acc)/len(epoch_acc):.4f}")
    
    # Test
    model.set_train(False)
    predictions = model.forward(test_x)
    
    loss = CEloss.loss(predictions, test_y)
    acc = accuracy(predictions, test_y)
    
    print(f"Test Epoch {epoch + 1}, Loss: {loss:.4f}, Acc: {acc:.4f}")
    model.set_train(True)
  
  
  # Saving and loading model  
  save_model_parameters(model, 'model.pkl')
  
  new_model = Model([
    DenseLayer(input_dim, hidden, learning_rate),
    BatchNormalization(hidden),
    Dropout(0.1),
    ReLU(),
    DenseLayer(hidden, output_dim, learning_rate),
    CEloss
  ])
  
  load_model_parameters(model, 'model.pkl')
  
  
  model.set_train(False)
  predictions = model.forward(test_x)
  
  loss = CEloss.loss(predictions, test_y)
  acc = accuracy(predictions, test_y)
  
  print(f"Test Epoch {epoch + 1}, Loss: {loss:.4f}, Acc: {acc:.4f}")
  model.set_train(True)
    

