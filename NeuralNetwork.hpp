#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <cmath>
#include <vector>
#include <functional>
#include <iostream>
#include <random>

const float N_EPSILON = 1e-7f;
const float N_INFINITY = 1.0f/0.0f;



class NeuralNetwork {
    public:
    // Struct that stores an activation function and it's derivative
    struct Activation {
        float (*fn) (const float);
        float (*fn_d) (const float);
    };
    // Struct the stores the activations for the entire output layer
    struct OutputActivations {
        std::function<void(const float*, float*)> fn;
        std::function<void(const float*, float*)> fn_d;
    };

    // Activation functions
    const static Activation LINEAR;
    const static Activation RELU;
    const static Activation SIGMOID; // Sigmoid from 0 to 1
    const static Activation AUG_SIGMOID; // Sigmoid from -1 to 1
    const static Activation TANH;
    const static Activation EXP;
    const static Activation POW_1P1;
    // Output activation functions
    const static OutputActivations ACTIVATIONS(const Activation& activation, const size_t layer_size);
    const static OutputActivations ACTIVATIONS(const std::vector<Activation>& activations);
    const static OutputActivations SOFTMAX(const size_t layer_size);


    public:
    // Constructors
    NeuralNetwork();
    NeuralNetwork(  size_t input_layer_size, size_t output_layer_size, size_t hidden_layer_size, size_t hidden_layer_count,
                    const Activation& hidden_activation = RELU, const OutputActivations& output_activations = {std::function<void(const float*, float*)>(), std::function<void(const float*, float*)>()} );

    // Sets random weights between min and max
    void randomInit(const float min = -1, const float max = 1);
    // Sets random weights based on Kaiming initialization
    void kaimingInit();

    // Propagate the signals from the inputs into the outputs, and store each value at each node, before and after the activation function
    void propagateStore(const float* inputs, float* prev_node_vals, float* post_node_vals) const;
    // Propagate the signals from the inputs into the outputs
    void propagate(const float* inputs, float* outputs) const;
    
    // Find the gradients with the squared loss and node values obtained from propagateStore() and store the gradients
    // The output_D is essentially the derivative of the bias' for the output nodes
    void backpropagateStore(const float* inputs, const float* prev_node_vals, const float* post_node_vals, const float* output_D,
                            float* weight_grad, float* bias_grad) const;
    // Find the gradients with the squared loss and node values obtained from propagateStore() and update the bias and weights
    void backpropagate(const float* inputs, const float* prev_node_vals, const float* post_node_vals, const float* output_D, const float learning_rate);

    // Update weights with gradients
    // Subtracts weights for gradient descent
    void applyGradients(const float* weight_grad, const float* bias_grad, const float learning_rate);
    // Apply L2 regularization to gradients
    void addRegularization(float* weight_grad, float* bias_grad, const float regularization_strength) const;

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
    size_t input_layer_size;
    size_t output_layer_size;
    size_t hidden_layer_size;
    size_t hidden_layer_count;
    std::vector<float> weights;
    std::vector<float> bias;

    Activation hidden_activation;
    OutputActivations output_activations;
};



// ACTIVATIONS
constexpr NeuralNetwork::Activation NeuralNetwork::LINEAR = {
    [](const float v){
        return v;
    },
    [](const float v){
        return 1.0f;
    }
};
constexpr NeuralNetwork::Activation NeuralNetwork::RELU = {
    [](const float v){
        return std::max(0.0f, v);
    },
    [](const float v){
        return (float) !std::signbit(v);
    }
};
constexpr NeuralNetwork::Activation NeuralNetwork::SIGMOID = {
    [](const float v){
        return 1.0f/(1+std::exp(-v));
    },
    [](const float v){
        const float sigmoid = 1.0f/(1+std::exp(-v));
        return sigmoid*(1-sigmoid);
    }
};
constexpr NeuralNetwork::Activation NeuralNetwork::AUG_SIGMOID = {
    [](const float v){
        return 2.0f/(1+std::exp(-v)) - 1.0f;
    },
    [](const float v){
        const float aug_sigmoid = 2.0f/(1+std::exp(-v)) - 1.0f;
        return (aug_sigmoid+1)*(1-aug_sigmoid)/2;
    }
};
constexpr NeuralNetwork::Activation NeuralNetwork::TANH = {
    [](const float v){
        return std::tanh(v);
    },
    [](const float v){
        const float tanh = std::tanh(v);
        return 1 - tanh*tanh;
    }
};
constexpr NeuralNetwork::Activation NeuralNetwork::EXP = {
    [](const float v){
        return std::exp(v);
    },
    [](const float v){
        return std::exp(v);
    }
};
constexpr NeuralNetwork::Activation NeuralNetwork::POW_1P1 = {
    [](const float v){
        return std::pow(1.1f, v);
    },
    [](const float v){
        return 0.09531018f * std::pow(1.1f, v);
    }
};
    
// OUTPUT ACTIVATIONS
const NeuralNetwork::OutputActivations NeuralNetwork::ACTIVATIONS(const Activation& activation, const size_t layer_size){
    return {
        [activation, layer_size](const float* p_v, float* v){
            for(size_t i = 0; i < layer_size; ++i){
                v[i] = activation.fn(p_v[i]);
            }
        },
        [activation, layer_size](const float* p_v, float* v){
            for(size_t i = 0; i < layer_size; ++i){
                v[i] = activation.fn_d(p_v[i]);
            }
        }
    };
}
const NeuralNetwork::OutputActivations NeuralNetwork::ACTIVATIONS(const std::vector<Activation>& activations){
    return {
        [activations](const float* p_v, float* v){
            for(size_t i = 0; i < activations.size(); ++i){
                v[i] = activations[i].fn(p_v[i]);
            }
        },
        [activations](const float* p_v, float* v){
            for(size_t i = 0; i < activations.size(); ++i){
                v[i] = activations[i].fn_d(p_v[i]);
            }
        }
    };
}
const NeuralNetwork::OutputActivations NeuralNetwork::SOFTMAX(const size_t layer_size){ // Finds derivative of just softmax, but ideally the derivative of the loss should purely be (actual-softmax)
    return {
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
        [layer_size](const float* p_v, float* v){
            // Find softmax first
            float softmax[layer_size];
            float sum = 0;
            for(size_t i = 0; i < layer_size; ++i){
                softmax[i] = std::exp(p_v[i]);
                sum += softmax[i];
            }
            for(size_t i = 0; i < layer_size; ++i){
                softmax[i] = softmax[i] / sum;
            }
            // Use softmax to find derivative
            for(int i = 0; i < layer_size; ++i){ // Derivative of each value
                v[i] = 0;
                for(int j = 0; j < layer_size; ++j){ // Sum the derivatives with respect to each other value
                    if(i == j)
                        v[i] += softmax[i]*(1-softmax[i]);
                    else
                        v[i] += -softmax[i]*softmax[j];
                }
            }
        }
    };
}

// CONSTRUCTORS
NeuralNetwork::NeuralNetwork( ) { 
    // Set sizes to 0
    this->input_layer_size = 0;
    this->output_layer_size = 0;
    this->hidden_layer_size = 0;
    this->hidden_layer_count = 0;
}
NeuralNetwork::NeuralNetwork(   size_t input_layer_size, size_t output_layer_size, size_t hidden_layer_size, size_t hidden_layer_count,
                                const Activation& hidden_activation, const OutputActivations& output_activations ) {
    // Check if network sizes are invalid
    if(input_layer_size < 1  ||  output_layer_size < 1){
        throw std::invalid_argument("Cannot have input or output layer with size 0.");
    }
    if(hidden_layer_size < 1  ||  hidden_layer_count < 1){
        throw std::invalid_argument("Cannot have hidden layer with size 0.");
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
        this->output_activations = ACTIVATIONS(SIGMOID, output_layer_size);
    // Calculate total nodes and edges
    recalculate();
}

// INITIALIZATION
void NeuralNetwork::randomInit(const float min, const float max){
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
void NeuralNetwork::propagateStore(const float* inputs, float* prev_node_vals, float* post_node_vals) const {          
    // The starting index of the weights pointing into a given node
    int weight_start_index = 0;

    // Propagate from input layer
    {const int this_layer_start_index = this->getBiasLayerIndex(0);
    for(int i = 0; i < hidden_layer_size; i++){
        const int node_index = this_layer_start_index + i;

        prev_node_vals[node_index] = bias[node_index];
        for(int j = 0; j < input_layer_size; j++){
            prev_node_vals[node_index] += inputs[j] * weights[weight_start_index + j];
        }

        post_node_vals[node_index] = hidden_activation.fn(prev_node_vals[node_index]);

        weight_start_index += input_layer_size;
    }}

    // Propagate between hidden layers
    for(int k = 1; k < hidden_layer_count; k++){
        
        const int this_layer_start_index = this->getBiasLayerIndex(k);
        const int prev_layer_start_index = this->getBiasLayerIndex(k-1);
        for(int i = 0; i < hidden_layer_size; i++){
            const int node_index = this_layer_start_index + i;

            prev_node_vals[node_index] = bias[node_index];
            for(int j = 0; j < hidden_layer_size; j++){
                prev_node_vals[node_index] += post_node_vals[prev_layer_start_index + j] * weights[weight_start_index + j];
            }

            post_node_vals[node_index] = hidden_activation.fn(prev_node_vals[node_index]);

            weight_start_index += hidden_layer_size;
        }
    }

    // Propagate to output layer
    {const int this_layer_start_index = this->getBiasLayerIndex(hidden_layer_count);
    const int prev_layer_start_index = this->getBiasLayerIndex(hidden_layer_count-1);
    for(int i = 0; i < output_layer_size; i++){
        const int node_index = this_layer_start_index + i;

        prev_node_vals[node_index] = bias[node_index];
        for(int j = 0; j < hidden_layer_size; j++){
            prev_node_vals[node_index] += post_node_vals[prev_layer_start_index + j] * weights[weight_start_index + j];
        }
        //post_node_vals[node_index] = output_activations[i].fn( prev_node_vals[node_index] );

        weight_start_index += hidden_layer_size;
    }
    // Apply output activations
    output_activations.fn( prev_node_vals+this_layer_start_index, post_node_vals+this_layer_start_index );
    }
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

void NeuralNetwork::backpropagateStore( const float* inputs, const float* prev_node_vals, const float* post_node_vals, const float* output_D,
                                        float* weight_grad, float* bias_grad) const {
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
    output_activations.fn_d(prev_node_vals + this_layer_start_index, prev_outputs_d);
    for(int i = output_layer_size-1; i >= 0; --i){
        const int node_index = this_layer_start_index + i;
        // Bias
                                    //output_activations[i].fn_d(prev_node_vals[node_index])
        bias_grad[node_index] = (1) * prev_outputs_d[i] * (output_D[i]);
        
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
void NeuralNetwork::backpropagate(const float* inputs, const float* prev_node_vals, const float* post_node_vals, const float* output_D, const float learning_rate){
    float weight_grad[weights.size()];
    float bias_grad[bias.size()];
    
    backpropagateStore(inputs, prev_node_vals, post_node_vals, output_D, weight_grad, bias_grad);

    applyGradients(weight_grad, bias_grad, learning_rate);
}

void NeuralNetwork::applyGradients(const float* weight_grad, const float* bias_grad, const float learning_rate){
    for(int i = 0; i < weights.size(); i++){
        weights[i] -= weight_grad[i] * learning_rate;
    }
    for(int i = 0; i < bias.size(); i++){
        bias[i] -= bias_grad[i] * learning_rate;
    }
}
void NeuralNetwork::addRegularization(float* weight_grad, float* bias_grad, const float regularization_strength) const {
    for(int i = 0; i < getWeightCount(); ++i){
        weight_grad[i] += 2 * weights[i] * regularization_strength;
    }
    for(int i = 0; i < getBiasCount(); ++i){
        bias_grad[i] += 2 * bias[i] * regularization_strength;
    }
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



#endif