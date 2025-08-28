# Autoencoder from Scratch

This project implements an **autoencoder neural network** from scratch using **NumPy** to compress and reconstruct 8×8 grayscale digit images from the `sklearn` digits dataset. It demonstrates a complete training and evaluation pipeline without relying on deep learning libraries like TensorFlow or PyTorch.

---

## 📁 Project Structure

- `NeuralNet.py` — Core neural network implementation: supports forward propagation, backpropagation, and training.
- `Training of Autoencoder.ipynb` — Jupyter notebook for training the autoencoder.
- `Autoencoder in Action.ipynb` — Notebook for visualizing input digits, their compressed representations, and reconstructions.
- `layer_weights.npy` — (Optional) NumPy file containing trained layer weights for re-use or visualization.

---

##  Autoencoder Overview

The model is trained to minimize reconstruction loss between the input and the output image. The network consists of two main parts:

### Encoder
Reduces 64-dimensional input (8×8 image) to a compressed 32-dimensional representation (bottleneck).

### Decoder
Reconstructs the original input from the 32-dimensional bottleneck.

### Architecture

<img src="./Autoencoder Architecture.png" alt="Autoencoder Architecture" width="500"/>    

Input: 64       
Encoder: 50 → 40 → 36        
Decoder: 40 → 50 → 64      
Output: 64    

- Activation function: `tanh` for hidden layers, `sigmoid` at output layer.
- Loss function: Mean Squared Error (L2 loss).
- Optimizer: Manual implementation of Stochastic Gradient Descent.


---

##  Dataset

Uses the `load_digits()` dataset from `sklearn.datasets`:

- 8×8 grayscale digit images (flattened to 64 features).
- Total samples: 1797
- No labels are used (unsupervised learning).

---

## Training

Autoencoder is trained using custom forward and backward propagation loops.

## Future Improvements

    Adam optimizer

    Transition to full MNIST dataset (28×28)

    Better weights initialization


    
