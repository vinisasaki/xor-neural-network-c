#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// neural network architecture
#define INPUT_LAYER 2   // 2 neurons in the input layer
#define HIDDEN_LAYER 2  // 2 neurons in the hidden layer
#define OUTPUT_LAYER 1  // 1 neuron in the output layer

// hyperparameters
#define EPOCHS 2000000  // epochs define how many times the data is processed by the network
#define LEARNING_RATE 0.01 // learning rate controls how quickly the weights are updated

// activation function, in this case, I used sigmoid
// this function transforms any real value into a number between 0 and 1, with an S-shaped curve, used to represent probabilities.
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
// The derivative of the sigmoid function shows the impact of each weight on the final error
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// this function calculates the initial weights and biases to initialize the network
void initial_weights_and_biases(double hidden_weights[INPUT_LAYER][HIDDEN_LAYER],
                                double hidden_bias[HIDDEN_LAYER],
                                double output_weights[HIDDEN_LAYER][OUTPUT_LAYER],
                                double output_bias[OUTPUT_LAYER]) {
    for (int i = 0; i < INPUT_LAYER; i++) {
        for (int j = 0; j < HIDDEN_LAYER; j++) {
            hidden_weights[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // to generate a random value between -1 and 1
        }
    }

    for (int i = 0; i < HIDDEN_LAYER; i++) {
        hidden_bias[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        for (int j = 0; j < OUTPUT_LAYER; j++) {
            output_weights[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }

    for (int i = 0; i < OUTPUT_LAYER; i++) {
        output_bias[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

int main() {
    srand(42);  // this seed is used to replicate the results, 42 because it's the answer to everything :D

    double X[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}; // real inputs and outputs of the XOR gate
    double y[4][1] = {{0}, {1}, {1}, {0}};

    double hidden_weights[INPUT_LAYER][HIDDEN_LAYER];
    double hidden_bias[HIDDEN_LAYER];
    double output_weights[HIDDEN_LAYER][OUTPUT_LAYER];
    double output_bias[OUTPUT_LAYER];

    initial_weights_and_biases(hidden_weights, hidden_bias, output_weights, output_bias);

    // for each epoch, the network performs a series of calculations to adjust the weights and improve its prediction
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double loss = 0.0; // loss variable represents the network's error and leads the learning process

        for (int i = 0; i < 4; i++) {

            // --------------------------- forward propagation ---------------------------
            // this process calculates the output of the network with the current weights and biases
            double hidden_layer_input[HIDDEN_LAYER] = {0};
            double hidden_layer_output[HIDDEN_LAYER];

            for (int j = 0; j < HIDDEN_LAYER; j++) {
                hidden_layer_input[j] = hidden_bias[j];
                for (int k = 0; k < INPUT_LAYER; k++) {
                    hidden_layer_input[j] += X[i][k] * hidden_weights[k][j]; // the equation is: z = X[i] · hidden_weights + hidden_bias
                }
                hidden_layer_output[j] = sigmoid(hidden_layer_input[j]); // Then, the sigmoid is applied: a = sigmoid(z)
            }

            double output_layer_input[OUTPUT_LAYER] = {0};
            double predicted_output_value[OUTPUT_LAYER];

            for (int j = 0; j < OUTPUT_LAYER; j++) {
                output_layer_input[j] = output_bias[j];
                for (int k = 0; k < HIDDEN_LAYER; k++) {
                    output_layer_input[j] += hidden_layer_output[k] * output_weights[k][j]; // the same for output layer
                }
                predicted_output_value[j] = sigmoid(output_layer_input[j]);
            }
            // ---------------------------------------------------------------------------
            // --------------------------- backward propagation --------------------------

            // it is the process by which the neural network adjusts the weights and biases based on the error made during forward propagation
            double output_error[OUTPUT_LAYER];
            double output_delta[OUTPUT_LAYER];

            for (int j = 0; j < OUTPUT_LAYER; j++) {
                output_error[j] = y[i][j] - predicted_output_value[j]; // difference between expected and predicted
                loss += pow(output_error[j], 2);   // accumulating the squared error
                output_delta[j] = output_error[j] * sigmoid_derivative(predicted_output_value[j]);
            }

            double hidden_error[HIDDEN_LAYER] = {0};
            double hidden_delta[HIDDEN_LAYER];
            
            // calculate error and delta for the hidden layer (based on output delta and weights)
            for (int j = 0; j < HIDDEN_LAYER; j++) {
                for (int k = 0; k < OUTPUT_LAYER; k++) {
                    hidden_error[j] += output_delta[k] * output_weights[j][k];
                }
                hidden_delta[j] = hidden_error[j] * sigmoid_derivative(hidden_layer_output[j]);
            }

            // updating weights and biases of hidden layer
            for (int j = 0; j < OUTPUT_LAYER; j++) {
                output_bias[j] += output_delta[j] * LEARNING_RATE;
                for (int k = 0; k < HIDDEN_LAYER; k++) {
                    output_weights[k][j] += hidden_layer_output[k] * output_delta[j] * LEARNING_RATE;
                }
            }

            // updating weights and biases of output layer
            for (int j = 0; j < HIDDEN_LAYER; j++) {
                hidden_bias[j] += hidden_delta[j] * LEARNING_RATE;
                for (int k = 0; k < INPUT_LAYER; k++) {
                    hidden_weights[k][j] += X[i][k] * hidden_delta[j] * LEARNING_RATE;
                }
            }
        }

        // every 100000 epochs, it will print the MSE
        if (epoch % 100000 == 0) {
            printf("Epoch %d - Mean Square Error: %.6f\n", epoch, loss / 4.0);
        }
    }

    // testing the network after training
    printf("\nTest after training:\n");
    for (int i = 0; i < 4; i++) {
        double hidden_layer_input[HIDDEN_LAYER] = {0};
        double hidden_layer_output[HIDDEN_LAYER];
        for (int j = 0; j < HIDDEN_LAYER; j++) {
            hidden_layer_input[j] = hidden_bias[j];
            for (int k = 0; k < INPUT_LAYER; k++) {
                hidden_layer_input[j] += X[i][k] * hidden_weights[k][j];
            }
            hidden_layer_output[j] = sigmoid(hidden_layer_input[j]);
        }

        double output_layer_input[OUTPUT_LAYER] = {0};
        double predicted_output_value[OUTPUT_LAYER];
        for (int j = 0; j < OUTPUT_LAYER; j++) {
            output_layer_input[j] = output_bias[j];
            for (int k = 0; k < HIDDEN_LAYER; k++) {
                output_layer_input[j] += hidden_layer_output[k] * output_weights[k][j];
            }
            predicted_output_value[j] = sigmoid(output_layer_input[j]);
        }

        printf("Input: %d XOR %d  ->  Predicted Output: %.8f  (Expected: %d)\n",
               (int)X[i][0], (int)X[i][1],
               predicted_output_value[0], (int)y[i][0]);
    }

    return 0;
}
