import numpy as np

def leaky_relu(arr, derivative=False):
    if derivative:
        return np.where(arr > 0, 1, 0.01)
    return np.where(arr > 0, arr, arr * 0.01)

import numpy as np

def sigmoid(x, derivative=False):
    """
    Sigmoid activation function.

    Parameters:
    x : np.ndarray or float
        Input value(s)
    derivative : bool
        If True, returns the derivative of the sigmoid function

    Returns:
    np.ndarray or float
    """
    # Clip to avoid overflow in exp
    x = np.clip(x, -500, 500)
    s = 1 / (1 + np.exp(-x))
    if derivative:
        return s * (1 - s)
    return s


def tanh(x, derivative=False):
    if derivative:
        t = np.tanh(x)
        return 1 - t**2
    return np.tanh(x)




class Perceptron:
    """
    Perceptron implementation.
    """

    def __init__(self, inpt_shape: int, activation_fun=leaky_relu):
        """
        Initialise the Perceptron:
        - Initializes weights from N(0, 1), including bias.
        - Sets the activation function.

        inpt_shape: int, the number of input features (excluding bias)
                in other words, nr of lines entering the Perceptron.
        activation_fun: callable, the activation function (e.g., ReLU or sigmoid)
        """
        #initialise the weights randomly [weight + bias]
        self.weights = np.random.normal(size=inpt_shape+1) #+1 to account for bias!
        #add the score function
        self.activation_fun = activation_fun

    def predict(self, perc_input):
      """
      Outputs the Result of the Perceptron.
      perc_input: the input that is fed into the neuron.
      Result:
      The output of perceptron.
      """  
      #Check if inputs shapes agree
      assert perc_input.shape[0] == self.weights.shape[0] - 1, \
        f"Expected input of shape {(self.weights.shape[0] - 1,)}"

      net_perc = np.dot(self.weights[:-1], perc_input) + self.weights[-1]
      self.net_perc = net_perc
      self.perc_output = self.activation_fun(net_perc)
      return self.perc_output
          

class Layer:
    """
    Represents the single layer of NN.
    neuron_nets: the list of net sums of evry perceptron in the layer
    """
    def __init__(self, input_shape: int, num_neurons: int, activation_function=leaky_relu):
        """
        Initialise the Layer. 
        input_shape - the shape of the input that is fed into the NN
        num_neurons - number of neurons (Perceptrons) in the layer
        """
        self.neurons = [
            Perceptron(input_shape, activation_fun=activation_function)
            for _ in range(num_neurons)
        ]
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.neuron_nets = np.zeros(num_neurons)

    def get_weights(self, only_weights=False):
        """
        Returns the weights of the Layer
        only_weights (bool): Include the bias term in the returned weights.
        """
        if only_weights:
            return np.array([neuron.weights[:-1] for neuron in self.neurons])
        return np.array([neuron.weights for neuron in self.neurons])
    
    def _update_weights(self, new_weights):
        """
        Updates the weights of the layer (includig the bias).
        new_weights(np.array) - new  weights to be put into the layer
        """
        new_weights = np.asarray(new_weights)
        assert new_weights.shape == (self.num_neurons, self.neurons[0].weights.shape[0]), \
            f"Expected shape {(self.num_neurons, self.neurons[0].weights.shape[0])}, got {new_weights.shape}"
        for i, neuron in enumerate(self.neurons):
            neuron.weights = new_weights[i]


    def _get_net_values(self) -> np.ndarray:
        """
        Returns the net values of every neuron in the layer.
        """
        return np.array([neuron.net_perc for neuron in self.neurons])
        
    def predict(self, layer_input: np.ndarray) -> np.ndarray:
        """
        Produce the output of the NN's layer and store the result.
        """
        self.neuron_nets = []

        results = np.zeros(self.num_neurons)
        for i in range(self.num_neurons):
            results[i] = self.neurons[i].predict(layer_input)
            self.neuron_nets.append(self.neurons[i].net_perc)

        self.neuron_nets = np.array(self.neuron_nets)
        self.layer_output = results
        return results
        

class NN:
    """A simple feedforward neural network implementation."""

    def __init__(self, input_size: int):
        """
        Initialize the neural network.


        Parameters:
        input_size (int): Defines the size for the input layer (NN's input). 
        
        input_size:int,  defines the input size fot the first layer
        IMPORTANT
        First layer of the NN contains the input and the neurons.
        Usually first layer is just the input without neurons.
        """
        self.input_size = input_size
        #list for storing the layers
        self.layers = []
        #stores the activations of every node in every layer of NN
        self.activations = []
    
    def add_layer(self, num_neurons: int, activation_function = leaky_relu):
        """
        Add a fully connected layer to the network.

        Adds a layer consitting of num_nurons neurons.
        The input of every neuron in the layer is either
        the input_size for entire NN, and if this is a hidden layer
        then each neurons's input_size is the number of neurons of the last layer.
        Input size defines the number of weights that each neuron has.

        Parameters:
        num_neurons (int): Number of neurons in the layer.
        activation_func (function): Activation function for the layer.
        """
        #if first layer then set the NN's input otherwise last layers output as the input to every node in the 
        input_dim = self.layers[-1].num_neurons if self.layers else self.input_size
        self.layers.append(Layer(input_shape=input_dim, num_neurons=num_neurons, activation_function=activation_function))

    def _predict(self, nn_input:np.ndarray):
        """
        Perform a forward pass and return the network output.
        Params:
        nn_input (np.ndarray): Input to the network (eg. image).

        Resets neuron_nets and activations!!!

        Returns:
        np.ndarray: Network output.

        """
        #save the input for later use
        self.nn_input = nn_input

        self.neuron_nets = [] 
        self.activations = [] 
        
        nr_of_layers = len(self.layers)

        self.activations.append(nn_input)
        #calculate the result for the first layer and save it
        self.activations.append(self.layers[0].predict(nn_input))
        #update the rest of nn'a layesrs
        for i in range(1,nr_of_layers):
            #get the output from last layer (input to the next one)
            output_from_last = self.activations[i]
            #predict on next layer
            self.activations.append(self.layers[i].predict(output_from_last))

        return self.layers[-1].layer_output

    def error(self, y_true: np.ndarray):
        """
        Returns the L2 Loss of NN prediction:

        Parameters:
        y_true (np.ndarray): Expected output.

        Returns:
        float: Loss value.
        """
        return 0.5*np.sum((self.layers[-1].layer_output - y_true)**2)

    def _get_delta_values(self, y_true: np.ndarray):
        """
        Compute delta (error term) for each layer using backpropagation.
        With the use of the equation: 
        δl−1:=(fl−1)′∘(Wl)T⋅δl
        Parameters:
        y_true (np.ndarray) : Expected result of neural network.

        Returns:
        list[np.ndarray]: List containing delta
         valuess for every node in evry layer, going 
         from the begining to the end of layers list in NN.
        [1st layer delta, 2nd delta, ...., Lth delta])
        """
        delta_arr = []
        
        # find the last layer's delta value
        last_layer = self.layers[-1]
        last_layer_output = last_layer.layer_output
        last_nets = last_layer.neuron_nets
        delta_1 = -(y_true - last_layer_output)*last_layer.activation_function(
            last_nets,derivative=True)

        delta_arr.append(delta_1)
        nr_of_layers = len(self.layers)

        del_i = 0 #for indexing delta_arr
        #get the sigma values for the rest of layers
        for i in range(nr_of_layers - 2, -1, -1):
            current_layer = self.layers[i]
            f = current_layer.activation_function #get the act fun from this layer
            #layer right to current layer 
            layer_right = self.layers[i+1]
            W = layer_right.get_weights(only_weights=True)

            #get the sigma from the right layer
            del_old = delta_arr[del_i]
            #derivative
            f_prim = f(current_layer._get_net_values(), derivative=True)
            #calculate the sigma for current layer
            delta = f_prim * (W.T @ del_old)
            delta_arr.append(delta)
            del_i+=1
        
        #make the delta_arr in the right order
        delta_arr = delta_arr[::-1]
        return delta_arr
        
    def backward(self, y_true: np.ndarray, lr: float = 0.1) -> float:
        """
        Perform backprop on the last forward pass and update weights.
        Returns the new loss.

        Remember to call predict after calling this method,
        so for new Error weights will be updated.

        Parameters:
        y_true: the true values that NN should output.
        lr: learning rate

        Returns: The L2 loss after updating the layers
        """
        
        if len(self.layers) < 2:
            raise ValueError("Cannot do backpropagation with only one layer.")

        delta_arr = self._get_delta_values(y_true)
       
        #go trough every node
        for l_ix in range(len(delta_arr)):
            #gradient for weights and biases
            grad_w = np.outer(delta_arr[l_ix], self.activations[l_ix]) # ∂E/∂W
            grad_b = delta_arr[l_ix]                          # ∂E/∂b
            gradient = np.column_stack((grad_w, grad_b))
            
            #Update weights
            layer = self.layers[l_ix]
            old_weights = layer.get_weights()
            new_weights = old_weights - lr*gradient
            layer._update_weights(new_weights)

        self._predict(self.nn_input)

        return self.error(y_true)
    
    def train_sgd(model, X_train, y_train, epochs=10, lr=0.1, shuffle=True):
        """
        Trains the neural network using Stochastic Gradient Descent (SGD).
        

        Parameters:
        model : NN
            Instance of the NN class.
        X_train : np.ndarray
            Training inputs (num_samples x input_dim).
        y_train : np.ndarray
            Training targets (num_samples x output_dim).
        epochs : int
            Number of times to iterate over the entire training data.
        lr : float
            Learning rate.
        shuffle : bool
            Whether to shuffle data each epoch.
        """
        num_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            if shuffle:
                indices = np.random.permutation(num_samples)
                X_train = X_train[indices]
                y_train = y_train[indices]

            epoch_loss = 0
            for x, y in zip(X_train, y_train):
                output = model.predict(x)
                loss = model.backward(y, lr)
                epoch_loss += loss

            avg_loss = epoch_loss / num_samples
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")



    