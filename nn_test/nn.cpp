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
    jai::NeuralNetwork network( INPUT_COUNT, OUTPUT_COUNT, hidden_layer_size, hidden_layer_count,
                                jai::ELU, 
                                jai::SOFTMAX(INPUT_COUNT));
                                //jai::ACTIVATION(jai::SIGMOID, INPUT_COUNT));
    network.kaimingInit();
    const jai::LossFunction sqrd_diff = jai::SQUARED_DIFF(INPUT_COUNT);


    // Train network for a fixed number of steps
    std::cout << "Start Training: " << c << " datapoints\n";
    float total_loss = 0;
    std::srand(std::time(0));
    for(int i = 0; i < c; i++){
        // Get two random numbers
        float in[INPUT_COUNT];
        in[0] = randomRange(-10, 10);
        in[1] = randomRange(-10, 10);
        const bool lessThan = in[0] < in[1];
        
        // Determine actual output values (1 for larger number, 0 for smaller number)
        float actual_values[OUTPUT_COUNT];
        actual_values[0] = !lessThan ?  1 : 0;
        actual_values[1] =  lessThan ?  1 : 0;
        
        // Train network
        float loss = network.train(in, actual_values, sqrd_diff, learning_rate);
        total_loss += loss;
    }

    // Display average loss
    const double avg_loss = (double)(total_loss/c);
    std::cout << "Done Training: " << avg_loss << "\n\n";


    // Test finished network
    std::cout << "Start Testing: " << c << " datapoints\n";
    int err_c = 0;
    std::srand(std::time(0));
    for(int i = 0; i < c; i++){
        // Get two random numbers
        float in[INPUT_COUNT];
        in[0] = randomRange(-10, 10);
        in[1] = randomRange(-10, 10);
        const bool lessThan = in[0] < in[1];

        float out[OUTPUT_COUNT];
        network.propagate(in, out);
        
        // Determine actual output values (1 for larger number, 0 for smaller number)
        float actual_values[OUTPUT_COUNT];
        actual_values[0] = !lessThan ?  1 : 0;
        actual_values[1] =  lessThan ?  1 : 0;
        
        // Count the number of errors
        if(lessThan != (out[0] < out[1]))
            err_c++;
    }

    // Display number of errors
    std::cout << "Done Testing: " << err_c << " errors\n";
    std::cout << (double)err_c/c*100 << "% wrong\n\n";

    // Print network, if desired
    //std::cout << network << '\n';
}