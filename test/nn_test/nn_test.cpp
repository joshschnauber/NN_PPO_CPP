/* nn_test.cpp */

/*
 * Test of different parts of NeuralNetwork.hpp.
 * Tests the correctness of the activatio functions, and then does a simple test by 
 * training a NeuralNetwork to determine which of two numbers is larger.
 *
 * g++ nn_test.cpp -o nn_test.exe
 * nn_test.exe 100 2 3 0.01
 */

#include "../../NeuralNetwork.hpp"
#include "../unit_test.hpp"
#include <random>
#include <ctime>



const float _EPSILON = 1e-5;
const int INPUT_COUNT = 2;
const int OUTPUT_COUNT = 2;



// Get random value within range
inline float randomRange(const float min = 0, const float max = 1){
    return ((double) std::rand() / RAND_MAX)*(max-min) + min;
}



/**
 * Runs tests on each Activation to verify correctness
 */
void test_activation() {
    START_TESTING("Activation")


    UNIT_TEST("LinearActivation")

        jai::LinearActivation linear_a = jai::LinearActivation();
        
        test_true( linear_a.verify() );

    END_UNIT_TEST


    UNIT_TEST("ReLUActivation")

        jai::ReLUActivation relu_a = jai::ReLUActivation();
        
        test_true( relu_a.verify() );

    END_UNIT_TEST


    UNIT_TEST("ELUActivation")

        jai::ELUActivation elu_a = jai::ELUActivation();
        
        test_true( elu_a.verify() );

    END_UNIT_TEST


    UNIT_TEST("SoftplusActivation")

        jai::SoftplusActivation softplus_a = jai::SoftplusActivation();
        
        test_true( softplus_a.verify() );

    END_UNIT_TEST


    UNIT_TEST("SigmoidActivation")

        jai::SigmoidActivation sigmoid_a = jai::SigmoidActivation();
        
        test_true( sigmoid_a.verify() );

    END_UNIT_TEST


    UNIT_TEST("AugSigmoidActivation")

        jai::AugSigmoidActivation augsigmoid_a_1 = jai::AugSigmoidActivation(-1, 1);
        jai::AugSigmoidActivation augsigmoid_a_2 = jai::AugSigmoidActivation(2, 10);
        jai::AugSigmoidActivation augsigmoid_a_3 = jai::AugSigmoidActivation(-8, -3);
        
        test_true( augsigmoid_a_1.verify() );
        test_true( augsigmoid_a_2.verify() );
        test_true( augsigmoid_a_3.verify() );

        test_throws( jai::AugSigmoidActivation augsigmoid_a_4 = jai::AugSigmoidActivation(2, 1) );
        test_throws( jai::AugSigmoidActivation augsigmoid_a_5 = jai::AugSigmoidActivation(1, -1) );
        test_throws( jai::AugSigmoidActivation augsigmoid_a_6 = jai::AugSigmoidActivation(-2, -2.1) );

        test_not_throws( jai::AugSigmoidActivation augsigmoid_a_7 = jai::AugSigmoidActivation(10, 10) );
        test_not_throws( jai::AugSigmoidActivation augsigmoid_a_8 = jai::AugSigmoidActivation(-10, -10) );

    END_UNIT_TEST


    UNIT_TEST("TanhActivation")

        jai::TanhActivation tanh_a = jai::TanhActivation();
        
        test_true( tanh_a.verify() );

    END_UNIT_TEST


    UNIT_TEST("ExpActivation")

        jai::ExpActivation exp_a = jai::ExpActivation();
        
        test_true( exp_a.verify() );

    END_UNIT_TEST


    UNIT_TEST("AugExpActivation")

        jai::AugExpActivation augexp_a_1 = jai::AugExpActivation(0);
        jai::AugExpActivation augexp_a_2 = jai::AugExpActivation(0.5);
        jai::AugExpActivation augexp_a_3 = jai::AugExpActivation(1);
        jai::AugExpActivation augexp_a_4 = jai::AugExpActivation(10);
        
        test_true( augexp_a_1.verify() );
        test_true( augexp_a_2.verify() );
        test_true( augexp_a_3.verify() );
        test_true( augexp_a_4.verify() );

        test_throws( jai::AugExpActivation augexp_a_5 = jai::AugExpActivation(-1) );
        test_throws( jai::AugExpActivation augexp_a_6 = jai::AugExpActivation(-10) );

    END_UNIT_TEST


    UNIT_TEST("PowerActivation")

        jai::PowerActivation augexp_a_1 = jai::PowerActivation(0);
        jai::PowerActivation augexp_a_2 = jai::PowerActivation(1);
        jai::PowerActivation augexp_a_3 = jai::PowerActivation(-1);
        jai::PowerActivation augexp_a_4 = jai::PowerActivation(10);
        jai::PowerActivation augexp_a_5 = jai::PowerActivation(-8);
        
        test_true( augexp_a_1.verify() );
        test_true( augexp_a_2.verify() );
        test_true( augexp_a_3.verify() );
        test_true( augexp_a_4.verify() );
        test_true( augexp_a_5.verify() );

    END_UNIT_TEST


    END_TESTING
}



/**
 * Runs tests on each LayerActivation to verify correctness
 */
void test_layer_activation() {
    START_TESTING("LayerActivation")

    
    /* TODO: Tests for LayerActivation */


    END_TESTING
}


/**
 * Runs simple greater than or less than training test on NeuralNetwork
 */
void test_neural_network() {
    START_TESTING("NeuralNetwork")


    // Get input arguments
    int c = 100;
    int hidden_layer_size = 2;
    int hidden_layer_count = 3;
    float learning_rate = 0.01f;
    //if(argc > 1){
    //    c = std::stoi(argv[1]);
    //    if(argc > 3){
    //        hidden_layer_size = std::stoi(argv[2]);
    //        hidden_layer_count = std::stoi(argv[3]);
    //
    //        if(argc > 4){
    //            learning_rate = std::stof(argv[4]);
    //        }
    //    }
    //}


    // Initialize network
    jai::NeuralNetwork network( INPUT_COUNT, OUTPUT_COUNT, hidden_layer_size, hidden_layer_count,
                                jai::ELUActivation(), 
                                jai::SoftmaxLayerActivation());
                                //jai::UniformLayerActivation(jai::SigmoidActivation()));
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


    END_TESTING
}



int main(int argc, char **argv){
    
    /* Run each set of tests */
    test_activation();

    test_layer_activation();

    test_neural_network();

    return EXIT_SUCCESS;
}