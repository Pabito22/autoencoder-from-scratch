import numpy as np

def relu(arr, derivative=False):
    if derivative:
        return np.where(arr > 0, 1, 0.01)
    return np.where(arr > 0, arr, arr * 0.01)


class Perceptron:
    """
    A class that is supposed to simulate a single node in NN.
    """

    def __init__(self, inpt_shape, activation_fun=relu):
        """
        This function is supposed just to create the Perceptron!
        Initializes the weights and adds the score function.
        
        input_shape: the size of the stuff that is fed into the neuron it has to be 1D array
        score_function: one of the posiible score functions
        """
        #initialise the weights randomly [weight + bias]
        self.weights = np.random.normal(size=inpt_shape+1) #+1 to account for bias!
        #add the score function
        self.activation_fun = activation_fun

    def predict(self, perc_input):
      """
      Outputs the Result of the Perceptron.
      perc_input the input that is fed into the neuron.
      Result:
      The output of perceptron.
      """  
      net_perc = np.sum(self.weights[:-1] * perc_input) + self.weights[-1]
      self.net_perc = net_perc
      self.perc_output = self.activation_fun(net_perc)
      return self.perc_output
          

class Layer:
    """
    Represents the single layer of NN.
    neuron_nets: the list of net sums of evry perceptron in the layer
    """
    def __init__(self,input_shape, nr_neurons, activation_function=relu):
        """
        Initialise the Layer. 
        input_shape - the shape of the input that is fed into the NN
        nr_neurons - nr of neurons (Perceptros) in the NN
        """
        layer_neurons_list = []
        for _ in range(nr_neurons):
            layer_neurons_list.append(Perceptron(input_shape, activation_fun=activation_function))
        self.layer_neurons_list = layer_neurons_list
        self.nr_neurons = nr_neurons
        self.activation_function = activation_function
        #an array to store the net sums of perceptrons that are there before being fed into nn
        self.neuron_nets = []

    def get_weights(self):
        """
        Returns the weights of the Layer.
        Dosen't return the bias!
        """
        weights = []
        for node in self.layer_neurons_list:
            weights.append(node.weights[:-1])
        return np.array(weights)

    def _get_net_values(self):
        """
        Returns the net values of every neuron in the layer.
        """
        nets = []
        for node in self.layer_neurons_list:
            nets.append(node.net_perc)
        return np.array(nets)
        
    def predict(self, layer_input):
        """
        Produce the output of the NN's layer and store the result.
        """
        results = np.zeros(self.nr_neurons)
        for i in range(self.nr_neurons):
            results[i] = self.layer_neurons_list[i].predict(layer_input)
            self.neuron_nets.append(self.layer_neurons_list[i].net_perc)

        self.neuron_nets = np.array(self.neuron_nets)
        self.layer_output = results
        return results
        

class NN:
    """This class represents the whole Neural Network."""

    def __init__(self, nn_input_size):
        """
        Initializes the NN

        nn_input_size: defines the input size fot the first layer
        """
        self.nn_input_size = nn_input_size
        #list for storing the layers
        self.nn_layer_list = []
        #counts how many layers there are currently
        self.layer_number = 0 #SET TO PRVATE LATER!!
    
    def add_layer(self, nr_neurons, activation_funct = relu):
        """
        Adds a new layer to the NN.
        Parameters:
        nr_neurons: nr of neurons in the new layer (also output size of the layer)
        activation_funct: activation function for all neurons in the layer
        """
        #if list with layers not empty -> take the last outpts!
        if self.nn_layer_list:
            #save the output size of the last layer and put as input to the new layer
            next_layer_input_size = self.nn_layer_list[self.layer_number-1].nr_neurons
            self.nn_layer_list.append(Layer(input_shape=next_layer_input_size,
                                            nr_neurons=nr_neurons,activation_function=activation_funct))
        else:
            #save the output size of the last layer and put as input to the new layer
            next_layer_input_size = self.nn_input_size
            self.nn_layer_list.append(Layer(input_shape=next_layer_input_size,
                                            nr_neurons=nr_neurons,activation_function=activation_funct))

        #increment the value of layers stored in NN
        self.layer_number += 1

    def predict(self, nn_input):
        """
        Gives the output from the NN.
        Params:
        nn_input: the input(e.g. image) fed into the nn
        """
        #calculate the result for the first layer and save it
        self.nn_layer_list[0].predict(nn_input)
        #update the rest of nn'a layesrs
        for i in range(1,self.layer_number):
            #get the output from last layer (input to the next one)
            output_from_last = self.nn_layer_list[i-1].layer_output
            #predict on next layer
            self.nn_layer_list[i].predict(output_from_last)

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
        [l1 sigma val, l2sigma val, ...., ln sigma val]
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

        sig_i = 0 #for indexing sigma_arr
        #get the sigma values for the rest of layers
        for i in range(self.layer_number - 2, -1, -1):
            current_layer = self.nn_layer_list[i]
            f = current_layer.activation_function #get the act fun from this layer
            #layer right to current layer 
            layer_right = self.nn_layer_list[i+1]
            W = layer_right.get_weights()
            #get the sigma from the right layer
            sig_old = sigma_arr[sig_i]
            #derivative
            f_prim = f(current_layer._get_net_values(), derivative=True)
            #calculate the sigma for current layer
            print("W:\n,", W)
            print(sig_old)
            sigma = f_prim * (W.T @ sig_old)
            sigma_arr.append(sigma)
            sig_i+=1
        return sigma_arr
        
    def backward_pass(self, exp_output, lbd=0.1):
        """
        Do a one backward pass for one training example.
        Use it after calling the self.predict method.

        exp_output (np.array) : expectef output for a training sample
        """
        if self.layer_number < 2:
            raise ValueError("Cannot do backpropagation with only one layer.")

        #get the sigma values for every layer
        last_layer = self.nn_layer_list[-1]
        last_layer_output = last_layer.layer_output
        sigma_arr = self._get_sigma_values(exp_output)
        
        sigma = sigma_arr[0]
        #the outputs from the layer previous to the last one
        prev_layer_output = self.nn_layer_list[-2].layer_output

        #LIST for storing new weights values for all neurons
        new_weights_list = []
        
        #get the derivatives for the weights for every neuron in the lasy layer
        for i in range(last_layer.nr_neurons):
            sigma_i = sigma[i]
            neuron_i = last_layer.layer_neurons_list[i]
            #depending on how many weights u have - this big this array will be
            grad_i = sigma_i*prev_layer_output
            #update the weights
            old_weights = neuron_i.weights[:-1]
            bias = neuron_i.weights[-1]
            new_weights = old_weights - lbd*grad_i
            new_bias = bias - lbd*sigma_i
            print(f"Neuron {i}: \n old weights and bias")
            print(old_weights, ' + b= ', bias)
            print(f"gradient: {grad_i} + del b = {sigma_i} ")
            print(f"New weights + bias: {new_weights} b_new = {new_bias}")
            #save the new weights
            new_weights_list.append(np.concat([new_weights, [new_bias]]))

        #NOW HIDDEN LAYERS
        #for layer in self.nn_layer_list[:-1]:
            #FIRSTLY U NEED SIGMA!!

        return ("Ala")

    