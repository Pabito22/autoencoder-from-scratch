import numpy as np

def leaky_relu(arr, derivative=False):
    if derivative:
        return np.where(arr > 0, 1, 0.01)
    return np.where(arr > 0, arr, arr * 0.01)


class Perceptron:
    """
    A class that is supposed to simulate a single node in NN.
    """

    def __init__(self, inpt_shape: int, activation_fun=leaky_relu):
        """
        Initialise the Perceptron:
        - Initializes weights from N(0, 1), including bias.
        - Sets the activation function.

        inpt_shape: int, the number of input features (excluding bias)
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
    """This class represents the whole Neural Network."""

    def __init__(self, nn_input_size):
        """
        Initializes the NN

        activation: contains activations of every layer, including the input

        nn_input_size: defines the input size fot the first layer
        For the first layer you create 'input-layer -' !! Not only input! 
        """
        self.nn_input_size = nn_input_size
        #list for storing the layers
        self.nn_layer_list = []
        #for storing the activations + input (gradient calculation)
        self._a = []
    
    def add_layer(self, num_neurons, activation_funct = leaky_relu):
        """
        Adds a new layer to the NN.
        Parameters:
        num_neurons: number of neurons in the new layer (i.e., layer output size)
        activation_funct: activation function for all neurons in the layer
        """
        #if list with layers not empty -> take the last outpts!
        if self.nn_layer_list:
            nr_of_layers = len(self.nn_layer_list)
            #save the output size of the last layer and put as input to the new layer
            next_layer_input_size = self.nn_layer_list[nr_of_layers-1].num_neurons
            self.nn_layer_list.append(Layer(input_shape=next_layer_input_size,
                                            num_neurons=num_neurons,activation_function=activation_funct))
        else:
            #First layer
            first_layer_input_size = self.nn_input_size
            self.nn_layer_list.append(Layer(input_shape=first_layer_input_size,
                                            num_neurons=num_neurons,activation_function=activation_funct))

    def predict(self, nn_input):
        """
        Gives the output from the NN.
        Params:
        nn_input: the input(e.g. image) fed into the nn

        Resets neuron_nets and _a!!!
        """
        self.neuron_nets = [] 
        self._a = [] 
        
        nr_of_layers = len(self.nn_layer_list)

        self._a.append(nn_input)
        #calculate the result for the first layer and save it
        self._a.append(self.nn_layer_list[0].predict(nn_input))
        #update the rest of nn'a layesrs
        for i in range(1,nr_of_layers):
            #get the output from last layer (input to the next one)
            output_from_last = self._a[i]
            #predict on next layer
            self._a.append(self.nn_layer_list[i].predict(output_from_last))

        return self.nn_layer_list[-1].layer_output

    def error(self, expected_output):
        """
        Returns the Loss of NN prediction:

        expected_output : what the NN should return (the same shape as the last layer of NN!.
        """
        return 0.5*np.sum((self.nn_layer_list[-1].layer_output - expected_output)**2)

    def _get_sigma_values(self, exp_output):
        """
        This function calculates the sigma value for every node in evry
        layer of the NN, and returns an array.
        Parameters:
        exp_output : Expected result of neural network.
        Returns:
        sigma_arr: Array containing sigma vals for every node in evry layer
        np.array([1 sigma val, 2 sigma val, ...., L sigma val])
        """
        sigma_arr = []
        
        # find the first's layer sigma value
        last_layer = self.nn_layer_list[-1]
        last_layer_output = last_layer.layer_output
        last_nets = last_layer.neuron_nets
        sigma_1 = -(exp_output - last_layer_output)*last_layer.activation_function(
            last_nets,derivative=True)
        print("first Sigma:")
        print(sigma_1)
        sigma_arr.append(sigma_1)

        nr_of_layers = len(self.nn_layer_list)

        sig_i = 0 #for indexing sigma_arr
        #get the sigma values for the rest of layers
        for i in range(nr_of_layers - 2, -1, -1):
            print("-------------")
            print("i (layer from last to first)", i)
            current_layer = self.nn_layer_list[i]
            f = current_layer.activation_function #get the act fun from this layer
            #layer right to current layer 
            layer_right = self.nn_layer_list[i+1]
            W = layer_right.get_weights(only_weights=True)
            print("W:\n",W)
            #get the sigma from the right layer
            sig_old = sigma_arr[sig_i]
            print("sig old: \n", sig_old)
            #derivative
            f_prim = f(current_layer._get_net_values(), derivative=True)
            #calculate the sigma for current layer
            sigma = f_prim * (W.T @ sig_old)
            sigma_arr.append(sigma)
            sig_i+=1
        
        #make the sigma in the right order
        sigma_arr = sigma_arr[::-1]
        return sigma_arr
        
    def backward(self, y_true: np.ndarray, lr: float = 0.1) -> float:
        """
        Perform backprop on the last forward pass and update weights.
        Returns the new loss.

        Parameters:
        y_true: the true values that NN should output.
        lr: learning rate

        Returns: The L2 loss after updating the layers
        """
        
        if len(self.nn_layer_list) < 2:
            raise ValueError("Cannot do backpropagation with only one layer.")

        #get the sigma values for every layer
        last_layer = self.nn_layer_list[-1]
        sigma_arr = self._get_sigma_values(y_true)
        sigma = sigma_arr[-1]
        #the outputs from the layer previous to the last one
        prev_layer_output = self.nn_layer_list[-2].layer_output

        
        #go trough every node
        for l_ix in range(len(sigma_arr)):
            #gradient for weights and biases
            grad_w = np.outer(sigma_arr[l_ix], self._a[l_ix])
            grad_b = sigma_arr[l_ix]
            #total gradient
            gradient = np.column_stack((grad_w, grad_b))

            #Now u can update every layer's weights!
            layer = self.nn_layer_list[l_ix]
            old_weights = layer.get_weights()
            new_weights = old_weights - lr*gradient
            layer._update_weights(new_weights)

        return self.error(y_true)


    