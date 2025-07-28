# XOR Neural Network in C

A simple neural network written in C that learns the XOR logic gate using backpropagation and sigmoid activation.

This project was developed as a way to **understand how a neural network works under the hood**, implemented entirely from scratch in **C**, with no external libraries.

To learn the fundamental concepts behind artificial neural networks, including:
- How **artificial neurons** process data
- Manual implementation of **weights** and **biases**
- The use of the **sigmoid activation function** and its derivative
- The **backpropagation** algorithm for adjusting parameters
- Training a simple model to learn the logical **XOR** problem

## 🧠 Network Architecture
- **Input layer:** 2 neurons (representing the XOR inputs)
- **Hidden layer:** 2 neurons
- **Output layer:** 1 neuron
- **Activation function:** sigmoid
- **Training epochs:** 2,000,000
- **Learning rate:** 0.01

## 📈 Sample Output
```txt
Epoch 0 - Mean Square Error: 0.258299
...
Epoch 1900000 - Mean Square Error: 0.000067

Test after training:
Input: 0 XOR 0  ->  Predicted Output: 0.00803117  (Expected: 0)
Input: 0 XOR 1  ->  Predicted Output: 0.99248431  (Expected: 1)
Input: 1 XOR 0  ->  Predicted Output: 0.99085477  (Expected: 1)
Input: 1 XOR 1  ->  Predicted Output: 0.00708782  (Expected: 0)
```

## ⚙️ Compile and Run

```bash
gcc xor_neural_network.c -o xor_neural_network -lm
./xor_neural_network
