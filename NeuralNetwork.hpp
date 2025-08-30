/* NeuralNetwork.hpp */

#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "Tensor.hpp"
#include <cmath>
#include <vector>
#include <functional>
#include <iostream>
#include <algorithm>
#include <random>

const float N_EPSILON = 1e-7f;
const float N_INFINITY = 1.0f/0.0f;



/* Declaration */
namespace jai {
    /**
     * This represents an activation to be applied in a neural network.
     * It contains a function for the activation and its derivative.
     */
    class Activation {
        float fn( const float in ) const; 
        float fn_D( const float in ) const; 
    };
    
    // Activation functions
    extern const Activation LINEAR;
    extern const Activation RELU;
    extern const Activation ELU;
    extern const Activation SOFTPLUS;
    extern const Activation SIGMOID;        // Sigmoid from 0 to 1
    extern const Activation AUG_SIGMOID;    // Sigmoid from -1 to 1
    extern const Activation TANH;
    extern const Activation EXP;
    extern const Activation POW_1P1;        // Exponential function with base 1.1

    // Struct the stores the activations for an entire layer
    struct LayerActivation {
        size_t layer_size;
        std::function<void(const float*, float*)> fn;
        std::function<void(const float*, const float*, float*)> fn_d;
    };

    // Output activation functions
    extern const LayerActivation ACTIVATION( const Activation& activation, const size_t layer_size );
    extern const LayerActivation ACTIVATIONS( const std::vector<Activation>& activations );
    extern const LayerActivation SOFTMAX( const size_t layer_size );
    extern const LayerActivation SPLIT_SOFTMAX( const std::vector<int>& softmax_sizes );                // Creates multiple softmaxes in one output, each with the specified sizes
    extern const LayerActivation MIXED_SOFTMAX( );                                                      // Creates a mixed output of softmaxes and other activations

    // Struct that contains the loss function and it's derivative
    // Only calculates loss using a network's outputted values and the actual values
    struct LossFunction {
        size_t output_size;
        std::function<float(const float*, const float*)> fn;
        std::function<void(const float*, const float*, float*)> fn_d;
    };

    // Loss functions
    extern const LossFunction SQUARED_DIFF( const size_t output_size );
    extern const LossFunction ABS_DIFF( const size_t output_size );

    class NeuralNetwork {
        public:
        struct NetworkSize {
            size_t hidden_layer_size;
            size_t hidden_layer_count;
        };
        // Struct containing hyperparameters for training
        struct Hyperparameters {
            size_t epochs =                 1;
            size_t batch_size =             SIZE_MAX;
            float regularization_strength = 1e-5f;
            float momentum_decay =          0.900f;
            float sqr_momentum_decay =      0.999f;
            float learning_rate =           1e-2f;
        };
        
        public:
        // Constructors
        NeuralNetwork();
        NeuralNetwork(  size_t input_layer_size, size_t output_layer_size,
                        const Activation& hidden_activation = RELU, const LayerActivation& output_activations = {0, nullptr, nullptr} );
        NeuralNetwork(  size_t input_layer_size, size_t output_layer_size, size_t hidden_layer_size, size_t hidden_layer_count,
                        const Activation& hidden_activation = RELU, const LayerActivation& output_activations = {0, nullptr, nullptr} );

        // Sets random weights between min and max
        void randomInit(const float min = -1, const float max = 1);
        // Sets random weights based on Kaiming initialization
        void kaimingInit();

        // Propagate the signals from the inputs into the outputs, and store each value at each node, before and after the activation function
        // Returns a pointer to the outputs, which is stored internally
        const float* propagateStore(const float* inputs);
        // Propagate the signals from the inputs into the outputs
        void propagate(const float* inputs, float* outputs) const;
        
        // Find the gradients with the node values obtained from propagateStore() and store the gradients
        // The loss_d is the derivative of the loss function in terms of the outputs
        void backpropagateStore( const float* inputs, const float* loss_d );
        // Find the gradients with the node values obtained from propagateStore() and update the bias and weights
        void backpropagate( const float* inputs, const float* loss_d, const float regularization_strength = 1e-5f, const float learning_rate = 1e-2f );

        // Apply L2 regularization to gradients
        void applyRegularization( const float regularization_strength);
        // Update weights with gradients
        // Subtracts weights for gradient descent
        void applyGradients( const float learning_rate);
        
        // Updates the network with backpropagation to bring the output closer to the actual output from the given input
        // Returns the loss (or average loss)
        float train( const float* inputs, const float* actual_outputs, const float learning_rate = 1e-2f, const float regularization_strength = 1e-5f );
        float train( const float* inputs, const float* actual_outputs, const LossFunction& loss_fn, const float learning_rate = 1e-2f, const float regularization_strength = 1e-5f );
        // Updates the network like train(), but trains all of the datapoints at once, with the given batch size.
        std::vector<float> batchTrain( const std::vector<float*>& inputs, const std::vector<float*>& actual_outputs, 
                                       const size_t batch_size = SIZE_MAX, const size_t epochs = 1, const float learning_rate = 1e-2f, 
                                       const float regularization_strength = 1e-5f, const float momentum_decay = 0.9f, const float sqr_momentum_decay = 0.999f );
        std::vector<float> batchTrain( const std::vector<float*>& inputs, const std::vector<float*>& actual_outputs, const LossFunction& loss_fn, 
                                       const size_t batch_size = SIZE_MAX, const size_t epochs = 1, const float learning_rate = 1e-2f, 
                                       const float regularization_strength = 1e-5f, const float momentum_decay = 0.9f, const float sqr_momentum_decay = 0.999f );


        // Finds the start pointer for the weights in a given layer
        // Layer 0 contains edges between the input and first hidden layer
        inline int getWeightLayerIndex(const int layer) const;
        // Finds the start pointer for the bias' in a given layer
        // Layer 0 is the first hidden layer and the layer (hidden_layer_count) is the output layer (because there is (hidden_layer_count+1) total layers)
        inline int getBiasLayerIndex(const int layer) const;
        // Gets the start pointer for the weights pointing to the node at the given index
        inline int getNodeWeightsIndex(const int node_index) const;
        // Gets the number of weight edges pointing into the given bias node
        inline int getNodeWeightCount(const int node_index) const;

        // Gets the pointer to the weights and bias'
        inline const float* getWeightData() const   { return weights.data(); }
        inline const float* getBiasData() const     { return bias.data(); }
        // Get sizes for initializing external arrays
        inline size_t getInputLayerSize() const     { return input_layer_size; }
        inline size_t getOutputLayerSize() const    { return output_layer_size; }
        inline size_t getPropagateNodeCount() const { return bias.size(); }
        inline size_t getWeightCount() const        { return weights.size(); }
        inline size_t getBiasCount() const          { return bias.size(); }
        inline size_t getHiddenLayerCount() const   { return hidden_layer_count; }
        inline size_t getHiddenLayerSize() const    { return hidden_layer_size; }

        // Gets difference between the two networks
        float getWeightDiff(const NeuralNetwork& other) const;
        float getBiasDiff(const NeuralNetwork& other) const;
        // For printing the network
        friend std::ostream& operator<< (std::ostream& fs, const NeuralNetwork& n);


        private:
        // Recalculates needed values upon direct network size changes
        // Sets all weights and bias' to 0
        void recalculate();


        private:
        // Network values
        size_t input_layer_size;
        size_t output_layer_size;
        size_t hidden_layer_size;
        size_t hidden_layer_count;
        std::vector<float> weights;
        std::vector<float> bias;
        Activation hidden_activation;
        LayerActivation output_activations;
        // Cached values from propagateStore() or backpropagateStore()
        std::vector<float> propagate_vals_cache;
        std::vector<float> gradient_vals_cache;
    };
}



/* Implementation */
namespace jai {

    // ACTIVATIONS
    constexpr Activation LINEAR = {
        [](const float v){
            return v;
        },
        [](const float v){
            return 1.0f;
        }
    };
    constexpr Activation RELU = {
        [](const float v){
            return std::max(0.0f, v);
        },
        [](const float v){
            return (float) !std::signbit(v);
        }
    };
    constexpr Activation ELU = {
        [](const float v){
            return ( v >=0 ) ?  v : std::exp(v) - 1;
        },
        [](const float v){
            return ( v >=0 ) ?  1.0f : std::exp(v);
        }
    };
    constexpr Activation SOFTPLUS = {
        [](const float v){
            return std::log( 1 + std::exp(v) );
        },
        [](const float v){
            return 1 / (1 + std::exp(-v));
        }
    };
    constexpr Activation SIGMOID = {
        [](const float v){
            return 1.0f/(1+std::exp(-v));
        },
        [](const float v){
            const float sigmoid = 1.0f/(1+std::exp(-v));
            return sigmoid*(1-sigmoid);
        }
    };
    constexpr Activation AUG_SIGMOID = {
        [](const float v){
            return 2.0f/(1+std::exp(-v)) - 1.0f;
        },
        [](const float v){
            const float aug_sigmoid = 2.0f/(1+std::exp(-v)) - 1.0f;
            return (aug_sigmoid+1)*(1-aug_sigmoid)/2;
        }
    };
    constexpr Activation TANH = {
        [](const float v){
            return std::tanh(v);
        },
        [](const float v){
            const float tanh = std::tanh(v);
            return 1 - tanh*tanh;
        }
    };
    constexpr Activation EXP = {
        [](const float v){
            return std::exp(v);
        },
        [](const float v){
            return std::exp(v);
        }
    };
    constexpr Activation POW_1P1 = {
        [](const float v){
            return std::pow(1.1f, v);
        },
        [](const float v){
            return 0.09531018f * std::pow(1.1f, v);
        }
    };
        
    // OUTPUT ACTIVATIONS
    const LayerActivation ACTIVATION(const Activation& activation, const size_t layer_size) {
        return {
            layer_size,
            [activation, layer_size](const float* p_v, float* v){
                for(size_t i = 0; i < layer_size; ++i){
                    v[i] = activation.fn(p_v[i]);
                }
            },
            [activation, layer_size](const float* p_v, const float* post_d, float* v){
                for(size_t i = 0; i < layer_size; ++i){
                    v[i] = activation.fn_d(p_v[i]) * post_d[i];
                }
            }
        };
    }
    const LayerActivation ACTIVATIONS(const std::vector<Activation>& activations) {
        return {
            activations.size(),
            [activations](const float* p_v, float* v){
                for(size_t i = 0; i < activations.size(); ++i){
                    v[i] = activations[i].fn(p_v[i]);
                }
            },
            [activations](const float* p_v, const float* post_d, float* v){
                for(size_t i = 0; i < activations.size(); ++i){
                    v[i] = activations[i].fn_d(p_v[i]) * post_d[i];
                }
            }
        };
    }
    const LayerActivation SOFTMAX(const size_t layer_size) { 
        return {
            layer_size,
            [layer_size](const float* p_v, float* v){
                float sum = 0;
                for(size_t i = 0; i < layer_size; ++i){
                    v[i] = std::exp(p_v[i]);
                    sum += v[i];
                }
                for(size_t i = 0; i < layer_size; ++i){
                    v[i] = v[i] / sum;
                }
            },
            [layer_size](const float* p_v, const float* post_d, float* v){
                // Find softmax first
                float softmax[layer_size];
                float e_sum = 0;
                for(size_t i = 0; i < layer_size; ++i){
                    softmax[i] = std::exp(p_v[i]);
                    e_sum += softmax[i];
                }
                for(size_t i = 0; i < layer_size; ++i){
                    softmax[i] = softmax[i] / e_sum;
                }
                // Calculate derivative using softmax
                float sum = 0;
                for( size_t i = 0; i < layer_size; ++i ) {
                    sum += softmax[i] * post_d[i];
                }
                for( size_t i = 0; i < layer_size; ++i ) {
                    v[i] += softmax[i] * (post_d[i] - sum);
                }
            }
        };
    }
    const LayerActivation SPLIT_SOFTMAX( const std::vector<size_t>& softmax_sizes ) {
        // Count total size of the layer
        size_t total_layer_size = 0;
        for( const size_t size : softmax_sizes ) 
            total_layer_size += size;
        
        return {
            total_layer_size,
            [softmax_sizes](const float* p_v, float* v){
                
            },
            [softmax_sizes](const float* p_v, const float* post_d, float* v){
                
            }
        };
    }
    const LayerActivation MIXED_SOFTMAX( ) {
        
        return {
            0,
            [](const float* p_v, float* v){
                
            },
            [](const float* p_v, const float* post_d, float* v){
                
            }
        };
    }

    // LOSS FUNCTIONS
    const LossFunction SQUARED_DIFF( const size_t output_size ) {
        return {
            output_size,
            [output_size](const float* out, const float* actual_out) {
                float sqrd_sum = 0;

                for( size_t i = 0; i < output_size; ++i ) {
                    float diff = actual_out[i] - out[i];
                    sqrd_sum += diff*diff;
                }

                return sqrd_sum / output_size;
            },
            [output_size](const float* out, const float* actual_out, float* d) {
                for( size_t i = 0; i < output_size; ++i ) {
                    d[i] = -2 * (actual_out[i] - out[i]);
                }
            }
        };
    };
    const LossFunction ABS_DIFF( const size_t output_size ) {
        return {
            output_size,
            [output_size](const float* out, const float* actual_out) {
                float abs_sum = 0;

                for( int i = 0; i < output_size; ++i ) {
                    abs_sum += std::abs(actual_out[i] - out[i]);
                }

                return abs_sum;
            },
            [output_size](const float* out, const float* actual_out, float* d) {
                for( int i = 0; i < output_size; ++i ) {
                    const float diff = actual_out[i] - out[i];
                    d[i] = (diff > 0)  ?  -1.0  :  1.0;
                }
            }
        };
    };

    // NEWTORK CONSTRUCTORS
    NeuralNetwork::NeuralNetwork( ) { 
        // Set sizes to 0
        this->input_layer_size = 0;
        this->output_layer_size = 0;
        this->hidden_layer_size = 0;
        this->hidden_layer_count = 0;
    }
    NeuralNetwork::NeuralNetwork(   size_t input_layer_size, size_t output_layer_size, size_t hidden_layer_size, size_t hidden_layer_count,
                                    const Activation& hidden_activation, const LayerActivation& output_activations ) {
        // Check if network sizes are invalid
        if(input_layer_size < 1  ||  output_layer_size < 1){
            throw std::invalid_argument("Cannot have input or output layer with size 0.");
        }
        if(hidden_layer_size < 1  ||  hidden_layer_count < 1){
            throw std::invalid_argument("Cannot have hidden layer with size 0.");
        }
        if(output_activations.layer_size != output_layer_size) {
            throw std::invalid_argument("Network output layer size does not match output activations size.");
        }
        
        // Set sizes
        this->input_layer_size = input_layer_size;
        this->output_layer_size = output_layer_size;
        this->hidden_layer_size = hidden_layer_size;
        this->hidden_layer_count = hidden_layer_count;
        // Set activation functions
        this->hidden_activation = hidden_activation;
        if( output_activations.fn  &&  output_activations.fn_d )
            this->output_activations = output_activations;
        else
            this->output_activations = ACTIVATION(SIGMOID, output_layer_size);
        // Calculate total nodes and edges
        recalculate();
    }

    // INITIALIZATION
    void NeuralNetwork::randomInit( const float min, const float max ){
        for(int i = 0; i < weights.size(); i++){
            weights[i] = ((double) std::rand() / RAND_MAX)*(max-min) + min;
        }
        for(int i = 0; i < bias.size(); i++){
            bias[i] = 0.0f;
        }
    }
    void NeuralNetwork::kaimingInit(){
        std::random_device rd;
        std::mt19937 rd_gen(rd());
        std::normal_distribution dst(0.0f, 1.0f);

        const int input_weight_end_index = input_layer_size*hidden_layer_size;
        for(int i = 0; i < weights.size(); i++){
            int n_in = hidden_layer_size;
            if(i < input_weight_end_index)
                n_in = input_layer_size;

            weights[i] = dst(rd_gen) * std::sqrt(2.0f/n_in);
        }
        for(int i = 0; i < bias.size(); i++){
            bias[i] = 0;
        }
    }

    // NETWORK INTERFACE
    const float* NeuralNetwork::propagateStore( const float* inputs ) {          
        // Initialize internal cache of values, if not initialized yet
        const size_t node_count = this->getBiasCount();
        if( propagate_vals_cache.size() == 0 ) {
            propagate_vals_cache.resize( node_count*2 );
        }
        float* prev_node_vals = propagate_vals_cache.data();
        float* post_node_vals = prev_node_vals + node_count;

        // The starting index of the weights pointing into a given node
        int weight_start_index = 0;

        // Propagate from input layer
        {const int this_layer_start_index = this->getBiasLayerIndex(0);
        for(int i = 0; i < hidden_layer_size; i++){
            const int node_index = this_layer_start_index + i;
            // Value at node before activation
            prev_node_vals[node_index] = bias[node_index];
            for(int j = 0; j < input_layer_size; j++){
                prev_node_vals[node_index] += inputs[j] * weights[weight_start_index + j];
            }
            // Value at node after activation
            post_node_vals[node_index] = hidden_activation.fn(prev_node_vals[node_index]);

            weight_start_index += input_layer_size;
        }}

        // Propagate between hidden layers
        for(int k = 1; k < hidden_layer_count; k++){
            
            const int this_layer_start_index = this->getBiasLayerIndex(k);
            const int prev_layer_start_index = this->getBiasLayerIndex(k-1);
            for(int i = 0; i < hidden_layer_size; i++){
                const int node_index = this_layer_start_index + i;
                // Value at node before activation
                prev_node_vals[node_index] = bias[node_index];
                for(int j = 0; j < hidden_layer_size; j++){
                    prev_node_vals[node_index] += post_node_vals[prev_layer_start_index + j] * weights[weight_start_index + j];
                }
                // Value at node after activation
                post_node_vals[node_index] = hidden_activation.fn(prev_node_vals[node_index]);

                weight_start_index += hidden_layer_size;
            }
        }

        // Propagate to output layer
        const int output_layer_start_index = this->getBiasLayerIndex(hidden_layer_count);
        const int prev_layer_start_index = this->getBiasLayerIndex(hidden_layer_count-1);
        for(int i = 0; i < output_layer_size; i++){
            const int node_index = output_layer_start_index + i;
            // Value at node before activation
            prev_node_vals[node_index] = bias[node_index];
            for(int j = 0; j < hidden_layer_size; j++){
                prev_node_vals[node_index] += post_node_vals[prev_layer_start_index + j] * weights[weight_start_index + j];
            }
            // Value at node after activation
            weight_start_index += hidden_layer_size;
        }
        // Apply output activations
        output_activations.fn( prev_node_vals+output_layer_start_index, post_node_vals+output_layer_start_index );

        // Return pointer to outputs
        return (post_node_vals + output_layer_start_index);
    }
    void NeuralNetwork::propagate(const float* inputs, float* outputs) const {
        // Arrays to hold the current layers computed values
        float hidden_node_val[hidden_layer_size];
        float prev_hidden_node_val[hidden_layer_size];

        // The starting index of the weights pointing into a given node
        int weight_start_index = 0;

        // Propagate from input layer
        {const int this_layer_start_index = this->getBiasLayerIndex(0);
        for(int i = 0; i < hidden_layer_size; ++i){
            const int node_index = this_layer_start_index + i;

            hidden_node_val[i] = bias[node_index];
            for(int j = 0; j < input_layer_size; j++){
                hidden_node_val[i] += inputs[j] * weights[weight_start_index + j];
            }

            hidden_node_val[i] = hidden_activation.fn(hidden_node_val[i]);

            weight_start_index += input_layer_size;
        }}

        // Propagate between hidden layers
        for(int k = 1; k < hidden_layer_count; ++k){
            // Save previous hidden node values before setting new ones
            for(int i = 0; i < hidden_layer_size; ++i){
                prev_hidden_node_val[i] = hidden_node_val[i];
            }

            const int this_layer_start_index = this->getBiasLayerIndex(k);
            const int prev_layer_start_index = this->getBiasLayerIndex(k-1);
            for(int i = 0; i < hidden_layer_size; ++i){
                const int node_index = this_layer_start_index + i;

                hidden_node_val[i] = bias[node_index];
                for(int j = 0; j < hidden_layer_size; ++j){
                    hidden_node_val[i] += prev_hidden_node_val[j] * weights[weight_start_index + j];
                }

                hidden_node_val[i] = hidden_activation.fn(hidden_node_val[i]);

                weight_start_index += hidden_layer_size;
            }
        }

        // Propagate to output layer
        {float prev_outputs[output_layer_size];
        const int this_layer_start_index = this->getBiasLayerIndex(hidden_layer_count);
        for(int i = 0; i < output_layer_size; i++){
            const int node_index = this_layer_start_index + i;

            prev_outputs[i] = bias[node_index];
            for(int j = 0; j < hidden_layer_size; j++){
                prev_outputs[i] += hidden_node_val[j] * weights[weight_start_index + j];
            }
            //outputs[i] = output_activations[i].fn( outputs[i] );

            weight_start_index += hidden_layer_size;
        }
        // Apply output activations
        output_activations.fn( prev_outputs, outputs );
        }
    }

    void NeuralNetwork::backpropagateStore( const float* inputs, const float* loss_d ) {
        // Check if propagated node values have been cached
        if( propagate_vals_cache.size() == 0 ) {
            return;
        }
        const float* prev_node_vals = propagate_vals_cache.data();
        const float* post_node_vals = prev_node_vals + this->getBiasCount();

        // Initialize internal cache of gradient values, if not initialized yet
        if( gradient_vals_cache.size() == 0 ) {
            gradient_vals_cache.resize(this->getWeightCount() + this->getBiasCount(), 0.0f);
        }
        float* weight_grad = gradient_vals_cache.data();
        float* bias_grad = weight_grad + this->getWeightCount();

        // Sum of weights pointing back into each vertice
        float weight_sums[hidden_layer_size];
        float prev_weight_sums[hidden_layer_size];
        for(int j = 0; j < hidden_layer_size; j++)
            weight_sums[j] = 0;
        
        // The starting index of the weights pointing into a given node
        int weight_start_index = this->getNodeWeightsIndex(bias.size()-1);

        // Find grads for output nodes
        {
        const int this_layer_start_index = this->getBiasLayerIndex(hidden_layer_count);
        const int prev_layer_start_index = this->getBiasLayerIndex(hidden_layer_count-1);
        // Calculate output derviatives
        float prev_outputs_d[output_layer_size];
        output_activations.fn_d(prev_node_vals + this_layer_start_index, loss_d, prev_outputs_d);
        for(int i = output_layer_size-1; i >= 0; --i){
            const int node_index = this_layer_start_index + i;
            // Bias
                                        //output_activations[i].fn_d(prev_node_vals[node_index])
            bias_grad[node_index] = (1) * prev_outputs_d[i];
            
            // Weights
            for(int j = hidden_layer_size-1; j >= 0; --j){
                weight_grad[weight_start_index + j] = post_node_vals[prev_layer_start_index + j] * bias_grad[node_index]; 
                weight_sums[j] += weights[weight_start_index + j] * bias_grad[node_index];
                // Average the derivatives here, because this is where the output derivatives "meet up"
                weight_sums[j] /= output_layer_size;
            }

            weight_start_index -= hidden_layer_size;
        }}

        // Find grads between hidden nodes
        for(int k = hidden_layer_count-1; k >= 1; --k){
            for(int j = 0; j < hidden_layer_size; j++){
                prev_weight_sums[j] = weight_sums[j];
                weight_sums[j] = 0;
            }

            const int this_layer_start_index = this->getBiasLayerIndex(k);
            const int prev_layer_start_index = this->getBiasLayerIndex(k-1);
            for(int i = hidden_layer_size-1; i >= 0; --i){
                const int node_index = this_layer_start_index + i;
                // Bias
                bias_grad[node_index] = (1) * hidden_activation.fn_d(prev_node_vals[node_index]) * prev_weight_sums[i];

                // Weights
                for(int j = hidden_layer_size-1; j >= 0; --j){
                    weight_grad[weight_start_index + j] = post_node_vals[prev_layer_start_index+j] * bias_grad[node_index]; 
                    weight_sums[j] += weights[weight_start_index + j] * bias_grad[node_index];
                }
                
                weight_start_index -= hidden_layer_size;
            }
        }
        // Ensure start index accounts for difference in edges with the nodes in the first hidden layer
        weight_start_index += hidden_layer_size;
        weight_start_index -= input_layer_size;
        
        // Find grads for input nodes
        {const int this_layer_start_index = this->getBiasLayerIndex(0);
        for(int i = hidden_layer_size-1; i >= 0; --i){
            const int node_index = this_layer_start_index + i;
            // Bias
            bias_grad[node_index] = (1) * hidden_activation.fn_d(prev_node_vals[node_index]) * (weight_sums[i]);
            
            // Weights
            for(int j = input_layer_size-1; j >= 0; --j){
                weight_grad[weight_start_index + j] = inputs[j] * bias_grad[node_index]; 
            }
            
            weight_start_index -= input_layer_size;
        }}
    }
    void NeuralNetwork::backpropagate( const float* inputs, const float* loss_d, const float regularization_strength, const float learning_rate ) {        
        backpropagateStore(inputs, loss_d);

        applyRegularization(regularization_strength);
        applyGradients(learning_rate);
    }

    void NeuralNetwork::applyRegularization( const float regularization_strength ) {
        // Check if gradient values have been cached
        if( gradient_vals_cache.size() == 0 ) {
            return;
        }
        // Add regularization to gradient
        const size_t size = this->getWeightCount() + this->getBiasCount();
        for( int i = 0; i < this->getWeightCount(); ++i ){
            gradient_vals_cache[i] += 2 * weights[i] * regularization_strength;
        }
        for( int i = this->getWeightCount(); i < size; ++i ){
            gradient_vals_cache[i] += 2 * bias[i] * regularization_strength;
        }
    }
    void NeuralNetwork::applyGradients( const float learning_rate ) {
        // Check if gradient values have been cached
        if( gradient_vals_cache.size() == 0 ) {
            return;
        }
        // Update weights and bias' with gradient
        const size_t size = this->getWeightCount() + this->getBiasCount();
        for( int i = 0; i < this->getWeightCount(); ++i ) {
            weights[i] -= gradient_vals_cache[i] * learning_rate;
        }
        for( int i = this->getWeightCount(); i < size; ++i ) {
            bias[i] -= gradient_vals_cache[i] * learning_rate;
        }

        // Clear propagate value cache and gradient
        propagate_vals_cache.clear();
        gradient_vals_cache.clear();
    }

    float NeuralNetwork::train( const float* inputs, const float* actual_outputs, const float learning_rate, const float regularization_strength ) {
        return train(inputs, actual_outputs, SQUARED_DIFF(output_layer_size), regularization_strength, learning_rate);
    }
    float NeuralNetwork::train( const float* inputs, const float* actual_outputs, const LossFunction& loss_fn, const float learning_rate, const float regularization_strength ) {
        // Check if the loss function has the correct size
        if(loss_fn.output_size != output_layer_size) {
            throw std::invalid_argument("Network output layer size does not match output activations size.");
        }

        // Propagate through network to get 
        const float* outputs = this->propagateStore(inputs);

        // Calculate loss derivative from expected (actual) output
        float loss_d[output_layer_size];
        loss_fn.fn_d(outputs, actual_outputs, loss_d);

        // Backpropagate to train network
        this->backpropagate(inputs, loss_d, learning_rate);

        // Return loss
        return loss_fn.fn(outputs, actual_outputs);
    }
    // NOT IMPLEMENTED
    std::vector<float> NeuralNetwork::batchTrain( const std::vector<float*>& inputs, const std::vector<float*>& actual_outputs, const LossFunction& loss_fn, 
                                       const size_t batch_size, const size_t epochs, const float learning_rate, 
                                       const float regularization_strength, const float momentum_decay, const float sqr_momentum_decay ) {
        return batchTrain(inputs, actual_outputs, SQUARED_DIFF(output_layer_size), batch_size, epochs, learning_rate, regularization_strength, momentum_decay, sqr_momentum_decay);
    }
    std::vector<float> NeuralNetwork::batchTrain( const std::vector<float*>& inputs, const std::vector<float*>& actual_outputs, const LossFunction& loss_fn, 
                                       const size_t batch_size, const size_t epochs, const float learning_rate, 
                                       const float regularization_strength, const float momentum_decay, const float sqr_momentum_decay ) {        
        throw "NOT IMPLEMENTED";
        
        // Check if the loss function has the correct size
        if(loss_fn.output_size != output_layer_size) {
            throw std::invalid_argument("Network output layer size does not match output activations size.");
        }
        // Check if number of inputs and actual outputs are the same
        const size_t n = inputs.size();
        if( actual_outputs.size() != n) {
            throw std::invalid_argument("Number of inputs does not match the number of actual outputs.");
        }

        // Initialize vector that stores average loss after each batch
        std::vector<float> losses;
        losses.reserve( epochs * (1 + n/batch_size) );

        // Create vector with datapoint indexes
        std::vector<int> datapoint_indexes = std::vector<int>(n);
        for(int l = 0; l < n; ++l)
            datapoint_indexes[l] = l;
        // Create random number generator to shuffle indexes
        std::random_device rd;
        std::mt19937 rd_gen(rd());

        // Repeat training over the dataset for the number of epochs
        for( size_t k = 0; k < epochs; ++k ) {

            size_t i = 0;
            while( i < n ) {
                // Shuffle datapoints
                std::shuffle(datapoint_indexes.begin(), datapoint_indexes.end(), rd_gen);

                // Calculate total gradient of batch
                float total_loss = 0;
                size_t j = 0;
                while( j < batch_size  &&  i < n ) {



                    ++j; ++i;
                }
                losses.push_back( total_loss / j);
                

                // Perform regularization and momentum

                // 
            }


        }
        

        return losses;
    }

    // GETTERS
    inline int NeuralNetwork::getWeightLayerIndex(const int layer) const {
        const int first_layer_edge_count = input_layer_size*hidden_layer_size;
        const int hidden_layer_edge_count = hidden_layer_size*hidden_layer_size;
        
        return (layer==0 ? 0 : first_layer_edge_count + hidden_layer_edge_count*(layer-1));
    }
    inline int NeuralNetwork::getBiasLayerIndex(const int layer) const {
        return layer*hidden_layer_size;
    }
    inline int NeuralNetwork::getNodeWeightsIndex(const int node_index) const {
        // Between input and first hidden layer
        if(node_index < hidden_layer_size)
            return node_index*input_layer_size;
        else
            return (hidden_layer_size*input_layer_size) + (node_index-hidden_layer_size)*hidden_layer_size;
    }
    inline int NeuralNetwork::getNodeWeightCount(const int node_index) const {
        // Between input and first hidden layer
        if(node_index < hidden_layer_size)
            return input_layer_size;
        else
            return hidden_layer_size;
    }

    // DEBUGGING
    float NeuralNetwork::getWeightDiff(const NeuralNetwork& other) const {
        if(weights.size() != other.weights.size()){
            throw std::invalid_argument("Networks are not the same size!");
        }

        float sqrd_diff_sum;
        for(int i = 0; i < weights.size(); ++i){
            float diff = weights[i] - other.weights[i];
            sqrd_diff_sum += diff*diff;
        }
        return std::sqrt(sqrd_diff_sum);
    }
    float NeuralNetwork::getBiasDiff(const NeuralNetwork& other) const {
        if(bias.size() != other.bias.size()){
            throw std::invalid_argument("Networks are not the same size!");
        }

        float sqrd_diff_sum;
        for(int i = 0; i < bias.size(); ++i){
            float diff = bias[i] - other.bias[i];
            sqrd_diff_sum += diff*diff;
        }
        return std::sqrt(sqrd_diff_sum);
    }
    std::ostream& operator<< (std::ostream& fs, const NeuralNetwork& n){
        // Layer sizes
        fs << "Layers: " << n.input_layer_size << ' ';
        for(int i = 0; i < n.hidden_layer_count; i++)
            fs << n.hidden_layer_size << ' ';
        fs << n.output_layer_size;
        // Bias and weight values
        fs << "\nBias': ";
        for(int i = 0; i < n.bias.size(); i++){
            fs << n.bias[i] << ' ';
        }
        fs << "\nWeights: ";
        for(int i = 0; i < n.weights.size(); i++){
            fs << n.weights[i] << ' ';
        }

        return fs;
    }

    // INTERNAL FUNCTIONS
    void NeuralNetwork::recalculate(){
        const int weight_count = input_layer_size*hidden_layer_size  +  hidden_layer_size*hidden_layer_size*(hidden_layer_count-1)  +  output_layer_size*hidden_layer_size;
        const int bias_count = hidden_layer_size*hidden_layer_count + output_layer_size;

        // Initialize all weights at zero
        weights = std::vector<float>(weight_count, 0);
        bias = std::vector<float>(bias_count, 0);
    }
}



#endif