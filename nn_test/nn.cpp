// Simple test of NeuralNetwork.hpp to train a network to determine which of two numbers is larger.

/*
 * Compile and run:
 * g++ nn.cpp -o nn.exe
 * nn.exe 100 2 3 0.01
 */


#include "../NeuralNetwork.hpp"
#include <iostream>
#include <random>
#include <ctime>



const float _EPSILON = 1e-5;
const int INPUT_COUNT = 2;
const int OUTPUT_COUNT = 2;


// Get random value within range
inline float randomRange(const float min = 0, const float max = 1){
    return ((double) std::rand() / RAND_MAX)*(max-min) + min;
}


int main(int argc, char **argv){
    // Get input arguments
    int c = 100;
    int hidden_layer_size = 2;
    int hidden_layer_count = 3;
    float learning_rate = 0.01f;
    if(argc > 1){
        c = std::stoi(argv[1]);
        if(argc > 3){
            hidden_layer_size = std::stoi(argv[2]);
            hidden_layer_count = std::stoi(argv[3]);

            if(argc > 4){
                learning_rate = std::stof(argv[4]);
            }
        }
    }

    // Initialize network
    NeuralNetwork network(  INPUT_COUNT, OUTPUT_COUNT, hidden_layer_size, hidden_layer_count,
                            NeuralNetwork::RELU, NeuralNetwork::ACTIVATIONS(NeuralNetwork::SIGMOID, OUTPUT_COUNT));
    network.kaimingInit();

    // Train network for a fixed number of steps
    int err_c = 0;
    std::srand(std::time(0));
    for(int i = 0; i < c; i++){
        // Get two random numbers
        float in[INPUT_COUNT];
        in[0] = randomRange(-10, 10);
        in[1] = randomRange(-10, 10);
        const bool lessThan = in[0] < in[1];

        // Propagate through network with inputs and store propagated values
        const int node_count = network.getPropagateNodeCount();
        float prev_nodes[node_count];
        float post_nodes[node_count];
        network.propagateStore(in, prev_nodes, post_nodes);
        
        // Determine actual output values (1 for larger number, 0 for smaller number)
        float actual_values[OUTPUT_COUNT];
        actual_values[0] = !lessThan ?  1 : 0;
        actual_values[1] =  lessThan ?  1 : 0;
        
        // Calculate loss gradient (derivative of squared difference)
        float loss_grad[OUTPUT_COUNT];
        loss_grad[0] = -2*(actual_values[0] - post_nodes[node_count-2]);
        loss_grad[1] = -2*(actual_values[1] - post_nodes[node_count-1]);
        
        // Backpropagate to get the gradients for the weights and bias'
        const int weight_count = network.getWeightCount();
        const int bias_count = network.getBiasCount();
        float w_grad[weight_count];
        float b_grad[bias_count];
        network.backpropagateStore(in, prev_nodes, post_nodes, loss_grad, w_grad, b_grad);

        // Apply gradients
        network.applyGradients(w_grad, b_grad, learning_rate);

        // Count the number of errors
        if(lessThan != (post_nodes[node_count-2] < post_nodes[node_count-1]))
            err_c++;
    }

    // Display number of errors
    std::cout << "Done: " << err_c << " errors\n";
    std::cout << (double)err_c/c*100 << "% wrong\n";
    // Print network, if desired
    //std::cout << network;
}