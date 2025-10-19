/* nn_test.cpp */

/*
 * Test of different parts of NeuralNetwork.hpp.
 * Tests the correctness of the activation functions, and then does a simple test by 
 * training a NeuralNetwork to determine which of two numbers is larger.
 *
 * g++ -std=c++20 -g -Wextra -Wall nn_test.cpp -o nn_test.exe
 * nn_test.exe 
 * g++ -std=c++20 -g -Wextra -Wall nn_test.cpp -o nn_test.out
 * ./nn_test.out
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

        test_throws( jai::AugSigmoidActivation _ = jai::AugSigmoidActivation(2, 1) );
        test_throws( jai::AugSigmoidActivation _ = jai::AugSigmoidActivation(1, -1) );
        test_throws( jai::AugSigmoidActivation _ = jai::AugSigmoidActivation(-2, -2.1) );

        test_not_throws( jai::AugSigmoidActivation _ = jai::AugSigmoidActivation(10, 10) );
        test_not_throws( jai::AugSigmoidActivation _ = jai::AugSigmoidActivation(-10, -10) );

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

        test_throws( jai::AugExpActivation _ = jai::AugExpActivation(-1) );
        test_throws( jai::AugExpActivation _ = jai::AugExpActivation(-10) );

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
 * Runs simple functionality tests on the NeuralNetwork
 */
void test_neural_network() {
    START_TESTING("NeuralNetwork")


    UNIT_TEST("Network Size Constructors") {

        jai::NeuralNetwork nn1({2, 3, 1});

        test_equals( nn1.getInputLayerSize(), 2 );
        test_equals( nn1.getOutputLayerSize(), 1 );
        test_equals( nn1.getLayerCount(), 3 );
        test_equals( nn1.getLayerSize(0), 2 );
        test_equals( nn1.getLayerSize(1), 3 );
        test_equals( nn1.getLayerSize(2), 1 );

        jai::NeuralNetwork nn2({4, 5, 7, 3}, jai::SigmoidActivation(), jai::SoftmaxLayerActivation());

        test_equals( nn2.getInputLayerSize(), 4 );
        test_equals( nn2.getOutputLayerSize(), 3 );
        test_equals( nn2.getLayerCount(), 4 );
        test_equals( nn2.getLayerSize(0), 4 );
        test_equals( nn2.getLayerSize(1), 5 );
        test_equals( nn2.getLayerSize(2), 7 );
        test_equals( nn2.getLayerSize(3), 3 );

        jai::NeuralNetwork nn3(2, 1, 4, 1);

        test_equals( nn3.getInputLayerSize(), 2 );
        test_equals( nn3.getOutputLayerSize(), 1 );
        test_equals( nn3.getLayerCount(), 3 );
        test_equals( nn3.getLayerSize(0), 2 );
        test_equals( nn3.getLayerSize(1), 4 );
        test_equals( nn3.getLayerSize(2), 1 );

        jai::NeuralNetwork nn4(10, 5, 8, 3);

        test_equals( nn4.getInputLayerSize(), 10 );
        test_equals( nn4.getOutputLayerSize(), 5 );
        test_equals( nn4.getLayerCount(), 5 );
        test_equals( nn4.getLayerSize(0), 10 );
        test_equals( nn4.getLayerSize(1), 8 );
        test_equals( nn4.getLayerSize(2), 8 );
        test_equals( nn4.getLayerSize(3), 8 );
        test_equals( nn4.getLayerSize(4), 5 );

    } END_UNIT_TEST


    UNIT_TEST("XOR Training") {

        jai::NeuralNetwork xor_nn(
            2, 
            1, 
            2, 
            1, 
            jai::ReLUActivation(), 
            jai::UniformLayerActivation(jai::SigmoidActivation())
        );
        jai::NeuralNetwork::Hyperparameters hp;
        hp.max_epochs = 1000;

        const jai::Matrix X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        const jai::Matrix Y = {{0}, {1}, {1}, {0}};

        xor_nn.train(
            X,
            Y,
            jai::SquaredDiffLossFunction(),
            hp
        );

        for( size_t i = 0; i < X.size(0); ++i ) {
            const float out = xor_nn.propagate(X[0])[0];
            const float y_p = (out > 0.5) ? 1 : 0;
            std::cout << y_p << "\n";
            test_equals( y_p, Y[i][0] );
        }

    } END_UNIT_TEST


    UNIT_TEST("Larger Number Training With DataStream") {

        // Create data stream to retrieve two random numbers, and
        // whether or not one is larger than the other
        const long DATA_SEED = 100;
        const size_t MAX_DATAPOINTS = 10000;
        class RandomNumberDataStream : public jai::SimpleDataStream {
            bool retrieveDatapoint( 
                jai::BaseVector& training_input,
                jai::BaseVector& training_expected_output
            ) override {
                training_input[0] = dst( rd_gen );
                training_input[1] = dst( rd_gen );

                if( training_input[0] < training_input[1] ) {
                    training_expected_output[0] = 0;
                } else {
                    training_expected_output[0] = 1;
                }

                return (++datapoints_retrieved) < MAX_DATAPOINTS;
            }
            
            size_t inputSize() const override {
                return 2;
            }
            size_t outputSize() const override {
                return 1;
            }

            private:
            std::mt19937 rd_gen = std::mt19937(DATA_SEED);
            std::uniform_real_distribution<float> dst = std::uniform_real_distribution(-100.0f, 100.0f);
            size_t datapoints_retrieved = 0;
        };


    } END_UNIT_TEST


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

        /*
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
    */


    END_TESTING
}



int main(int argc, char **argv){
    
    /* Run each set of tests */
    test_activation();

    test_layer_activation();

    test_neural_network();

    return EXIT_SUCCESS;
}