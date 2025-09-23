/* NeuralNetwork.hpp */

#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "Tensor.hpp"
#include <cmath>
#include <vector>
#include <memory>
#include <functional>
#include <iostream>
#include <algorithm>
#include <random>

const float NN_EPSILON = 1e-7f;
const float NN_INFINITY = 1.0f/0.0f;



/* Declaration */
namespace jai {
    /**
     * This represents an activation to be applied in a neural network.
     * It contains a function for the activation and its derivative.
     * This is an abstract class intended to be overridden for specific functionality
     */
    class Activation {
        public:
        /**
         * The activation function on the value `x`.
         */
        virtual float fn( const float x ) const = 0;
        /**
         * The derivative of the activation function on the value `x`.
         */
        virtual float fn_D( const float x ) const = 0;
        /**
         * Creates a `std::unique_ptr` that manages a new copy of `this` activation.
         */
        virtual std::unique_ptr<Activation> clone() const = 0;
        /**
         * Verifies that `fn_D` is the derivative of `fn`, by testing values 
         * between `min` and `max`, with the given `step`.
         * Returns true if it is, and false if not.
         */
        bool verify( const float min = -10.0f, const float max = 10.0f, const float step = 0.5f ) const;
    };
    /**
     * Linear activation
     */
    class LinearActivation : public Activation {
        public:
        float fn( const float x ) const override;
        float fn_D( const float x ) const override;
        std::unique_ptr<Activation> clone() const override;
    };
    /**
     * ReLU activation
     */
    class ReLUActivation : public Activation {
        public:
        float fn( const float x ) const override;
        float fn_D( const float x ) const override;
        std::unique_ptr<Activation> clone() const override;
    };
    /**
     * ELU activation
     */
    class ELUActivation : public Activation {
        public:
        float fn( const float x ) const override;
        float fn_D( const float x ) const override;
        std::unique_ptr<Activation> clone() const override;
    };
    /**
     * Softplus activation
     */
    class SoftplusActivation : public Activation {
        public:
        float fn( const float x ) const override;
        float fn_D( const float x ) const override;
        std::unique_ptr<Activation> clone() const override;
    };
    /**
     * Sigmoid activation.
     */
    class SigmoidActivation : public Activation {
        public:
        float fn( const float x ) const override;
        float fn_D( const float x ) const override;
        std::unique_ptr<Activation> clone() const override;
    };
    /**
     * Augmented sigmoid activation with custom lower and upper bounds.
     */
    class AugSigmoidActivation : public Activation {
        public:
        /**
         * Constructs a sigmoid activation with a lower bound `l` and
         * upper bound `u`.
         * If `l` > `u`, this throws an `std::invalid_argument`.
         */
        AugSigmoidActivation( const float l, const float u );

        float fn( const float x ) const override;
        float fn_D( const float x ) const override;
        std::unique_ptr<Activation> clone() const override;

        private:
        float lower_bound;
        float range;
    };
    /**
     * Tanh activation
     */
    class TanhActivation : public Activation {
        public:
        float fn( const float x ) const override;
        float fn_D( const float x ) const override;
        std::unique_ptr<Activation> clone() const override;
    };
    /**
     * Exponential function activation.
     * Represents an exponential function of the form y = e^x.
     */
    class ExpActivation : public Activation {
        public:
        float fn( const float x ) const override;
        float fn_D( const float x ) const override;
        std::unique_ptr<Activation> clone() const override;
    };
    /**
     * Augmented exponential function activation with custom base.
     * Represents an exponential function of the form y = b^x.
     */
    class AugExpActivation : public Activation {
        public:
        /**
         * Constructs a exponential function activation with base `b`.
         * If `b` < 0, this throws an `std::invalid_argument`.
         */
        AugExpActivation( const float b );

        float fn( const float x ) const override;
        float fn_D( const float x ) const override;
        std::unique_ptr<Activation> clone() const override;

        private:
        float base;
        float ln_of_base;
    };
    /**
     * Power function activation
     * Represents a power function of the form y = x^p.
     */
    class PowerActivation : public Activation {
        public:
        /**
         * Constructs a power function activation with the power `p`.
         */
        PowerActivation( const float p );

        float fn( const float x ) const override;
        float fn_D( const float x ) const override;
        std::unique_ptr<Activation> clone() const override;

        private:
        float power;
    };

    /**
     * This represents an activation to be applied to an entire layer in a neural network.
     * It contains a function for the activation and its derivative.
     * This is an abstract class intended to be overridden for specific functionality.
     */
    class LayerActivation {
        public:
        /**
         * The activation function on Vector `x`.
         * Places the result in Vector `y`.
         */
        virtual void fn( const BaseVector& x, BaseVector& y ) const = 0;
        /**
         * The derivative of the activation function on Vector `x` using the .
         * Places the result in Vector `y`.
         */
        virtual void fn_D( const BaseVector& x, const BaseVector& ___, BaseVector& y_D ) const = 0;
        /**
         * Creates a `std::unique_ptr` that manages a new copy of `this` layer activation.
         */
        virtual std::unique_ptr<LayerActivation> clone() const = 0;
        /**
         * Checks if the `layer_size` is valid for this layer activation.
         */
        virtual bool isValidLayerSize( size_t layer_size ) const = 0;
        /**
         * Verifies that `fn_D` is the derivative of `fn`.
         * Returns true if it is, and false if not.
         * Tests `test_count` Vectors of size `layer_size`, with random values bounded
         * by `min` and `max`. The values are generated using the given `seed`.
         */
        bool verify( 
            const size_t layer_size, const float min = -10.0f, const float max = 10.0f, 
            const size_t test_count = 20, const int seed = 0
        ) const;
    };
    /**
     * Uniform layer activation.
     * Applies the same independent activation to each node in the layer.
     */
    class UniformLayerActivation : public LayerActivation {
        public:
        /**
         * Constructs a uniform layer activation using `activation` for each node.
         */
        UniformLayerActivation( const Activation& activation );

        void fn( const BaseVector& x, BaseVector& y ) const override;
        void fn_D( const BaseVector& x, const BaseVector& ___, BaseVector& y_D ) const override;
        std::unique_ptr<LayerActivation> clone() const override;
        bool isValidLayerSize( size_t layer_size ) const override;

        private:
        std::unique_ptr<Activation> activation;
    };
    /**
     * Non Uniform layer activation.
     * Applies a different independent activation to each node in the layer.
     * TODO: The construction of this is not very user friendly.
     */
    class NonUniformLayerActivation : public LayerActivation {
        public:
        /**
         * Constructs a non uniform layer activation using the `activations`, such
         * that the i'th activation corresponds to the i'th node in the layer.
         * This layers activation only supports layers such that the layer size is the
         * same as the size of `activations`.
         * The activations correspoinding to the passed `Activation*` should be allocated
         * via the `new` keyword. They will then be managed by the class.
         */
        NonUniformLayerActivation( const std::vector<Activation*>& activations );

        void fn( const BaseVector& x, BaseVector& y ) const override;
        void fn_D( const BaseVector& x, const BaseVector& ___, BaseVector& y_D ) const override;
        std::unique_ptr<LayerActivation> clone() const override;
        bool isValidLayerSize( size_t layer_size ) const override;
 
        private:
        std::vector<std::unique_ptr<Activation>> activations;
    };
    /**
     * Softmax layer activation.
     */
    class SoftmaxLayerActivation : public LayerActivation {
        public:
        void fn( const BaseVector& x, BaseVector& y ) const override;
        void fn_D( const BaseVector& x, const BaseVector& ___, BaseVector& y_D ) const override;
        std::unique_ptr<LayerActivation> clone() const override;
        bool isValidLayerSize( size_t layer_size ) const override;
    };
    // TODO: Do implementation
    /**
     * Split softmax layer activation.
     * Consists of multiple independent softmaxes across the layer's nodes.
     */
    class SplitSoftmaxLayerActivation : public LayerActivation {
        public:
        /**
         * Constructs a split softmax layer activation.
         * `softmax_sizes` specifies that size of each independent softmax.
         * This means the total size of the layer should be the sum of each element in
         * `softmax_sizes`.
         */
        SplitSoftmaxLayerActivation( const std::vector<int> softmax_sizes );

        void fn( const BaseVector& x, BaseVector& y ) const override;
        void fn_D( const BaseVector& x, const BaseVector& ___, BaseVector& y_D ) const override;
        std::unique_ptr<LayerActivation> clone() const override;
        bool isValidLayerSize( size_t layer_size ) const override;

        private:
        std::vector<int> softmax_sizes;
    };
    // TODO: Do implementation
    /**
     * Mixed softmax layer activation.
     * Consists of different independent activations for the nodes at the start of the
     * layer, and then multiple independent softmax layers after those.
     */
    class MixedSoftmaxLayerActivation : public LayerActivation {
        public:
        /**
         * Constructs a mixed softmax layer activation.
         * The i'th activation corresponds to the i'th node in the layer, and 
         * `softmax_sizes` specifies that size of each independent softmax after that.
         * This means the total size of the layer should be the sum of the size of
         * `activations` and each element in `softmax_sizes`.
         */
        MixedSoftmaxLayerActivation( 
            const std::vector<Activation*>& activations, 
            const std::vector<int> softmax_sizes 
        );

        void fn( const BaseVector& x, BaseVector& y ) const override;
        void fn_D( const BaseVector& x, const BaseVector& ___, BaseVector& y_D ) const override;
        std::unique_ptr<LayerActivation> clone() const override;
        bool isValidLayerSize( size_t layer_size ) const override;

        private:
        std::vector<std::unique_ptr<Activation>> activations;
        std::vector<int> softmax_sizes;
    };

    /**
     * This represents a loss function to be applied to the output of a neural network,
     * using the expected output to calculate the loss.
     * It contains a function for the loss and its derivative.
     * Called 'Simple' because it compares only the actual vs expected outputs, without
     * considering other variables.
     * This is an abstract class intended to be overridden for specific functionality.
     */
    class SimpleLossFunction {
        public:
        /**
         * The loss function on Vector `x`.
         * Places the result in Vector `y`.
         */
        virtual float fn( const BaseVector& x, const BaseVector& expected_x ) const = 0;
        /**
         * The derivative of the activation function on Vector `x` using the .
         * Places the result in Vector `y`.
         */
        virtual void fn_D( const BaseVector& x, const BaseVector& expected_x, BaseVector& y_D ) const = 0;
        /**
         * Creates a `std::unique_ptr` that manages a new copy of `this` layer activation.
         */
        virtual std::unique_ptr<SimpleLossFunction> clone() const = 0;
        /**
         * Checks if the `layer_size` is valid for this layer activation.
         */
        virtual bool isValidLayerSize( size_t layer_size ) const = 0;
        /**
         * Verifies that `fn_D` is the derivative of `fn`.
         * Returns true if it is, and false if not.
         * Tests `test_count` Vectors of size `layer_size`, with random values bounded
         * by `min` and `max`. The values are generated using the given `seed`.
         */
        bool verify( 
            const size_t layer_size, const float min = -10.0f, const float max = 10.0f, 
            const size_t test_count = 20, const int seed = 0
        ) const;
    };
    /**
     * Squared difference loss function.
     * The value of the loss is the average of squared differences between the expected and
     * actual values.
     */
    class SquaredDiffLossFunction : public SimpleLossFunction {
        public:
        float fn( const BaseVector& x, const BaseVector& expected_x ) const override;
        void fn_D( const BaseVector& x, const BaseVector& expected_x, BaseVector& y_D ) const override;
        std::unique_ptr<SimpleLossFunction> clone() const override;
        bool isValidLayerSize( size_t layer_size ) const override;
    };
    /**
     * Absolute difference loss function.
     * The value of the loss is the average of absolute differences between the expected and
     * actual values.
     */
    class AbsoluteDiffLossFunction : public SimpleLossFunction {
        public:
        float fn( const BaseVector& x, const BaseVector& expected_x ) const override;
        void fn_D( const BaseVector& x, const BaseVector& expected_x, BaseVector& y_D ) const override;
        std::unique_ptr<SimpleLossFunction> clone() const override;
        bool isValidLayerSize( size_t layer_size ) const override;
    };
    /**
     * Cross entropy loss function.
     * The inputs are expected to be probabilities that all add to 1.
     */
    class CrossEntropyLossFunction : public SimpleLossFunction {
        public:
        float fn( const BaseVector& x, const BaseVector& expected_x ) const override;
        void fn_D( const BaseVector& x, const BaseVector& expected_x, BaseVector& y_D ) const override;
        std::unique_ptr<SimpleLossFunction> clone() const override;
        bool isValidLayerSize( size_t layer_size ) const override;
    };

    /**
     * Abstract class for retreiving data for training a neural network over time.
     * Called 'Simple' because it retrieves only an input and the expected output,
     * with no other context.
     */
    class SimpleDataStream {
        public:
        /**
         * Gets an input to propagate through a neural network and places it into
         * `training_input`, and gets the expected output for that input and places it
         * into `training_expected_output`.
         */
        virtual Vector retrieveDatapoint( 
            BaseVector& training_input,
            BaseVector& training_expected_output
        ) = 0;
        /**
         * Returns the size of the datapoint training input
         */
        virtual size_t inputSize() = 0;
        /**
         * Returns the size of the datapoint training output
         */
        virtual size_t outputSize() = 0;
    };

    /**
     * Class...
     */
    class NeuralNetwork {
        /* Inner Structs */
        public:

        /**
         * Struct representing the hyperparameters needed for training a neural network
         */
        struct Hyperparameters {
            /**
             * The maximum error before training will stop.
             */
            float error_tolerance =         0.1f;
            /**
             * The maximum number of passes over the entire training set.
             * Training will stop even if the desired error tolerance is not achieved.
             */
            size_t epochs =                 10;  
            /**
             *  The maximum size of each batch to train on.
             */  
            size_t batch_size =             1000;
            /**
             * The regularization strength applied to the gradients to encourage smaller
             * network weights and bias'
             */
            float regularization_strength = 1e-5f;
            /**
             * The momentum decay applied to the gradients
             */
            float momentum_decay =          0.900f;
            /**
             * The square momentum decay applied to the gradients
             */
            float sqr_momentum_decay =      0.999f;
            /**
             * The learning rate applied to the gradients when updating the network.
             */
            float learning_rate =           1e-2f;
        };
        
        /* Constructors */
        public:

        /**
         * Contructs an empty NeuralNetwork with no layers.
         */
        NeuralNetwork();
        /**
         * Constructs a NeuralNetwork with the given sizes and activations, with no
         * weights or bias' set.
         */
        NeuralNetwork(  
            const size_t input_layer_size, 
            const size_t output_layer_size, 
            const size_t hidden_layer_count,
            const size_t hidden_layer_size,
            const Activation& hidden_activation = ReLUActivation(), 
            const LayerActivation& output_layer_activation = UniformLayerActivation(SigmoidActivation())
        );
        /**
         * Constructs a NeuralNetwork with `layer_count` layers, where each layer is 
         * given the corresponding size from `layer_sizes`, as well as the activations, 
         * with no weights or bias' set.
         */
        NeuralNetwork(
            const size_t layer_count, 
            const size_t layer_sizes[], 
            const Activation& hidden_activation = ReLUActivation(), 
            const LayerActivation& output_layer_activation = UniformLayerActivation(SigmoidActivation())
        );
        /**
         * Constructs a NeuralNetwork using already existing weights and bias'.
         */
        NeuralNetwork(
            const RaggedTensor<3>& weights,
            const RaggedMatrix& bias,
            const Activation& hidden_activation,
            const LayerActivation& output_layer_activation
        );

        /* Accessors */
        public:

        /**
         * Propagates through the network using the `inputs` Vector, and returns the
         * final result.
         */
        Vector propagate( const BaseVector& inputs ) const;
        /**
         * Propagates through the network with each row in the `inputs` Matrix, and
         * returns the final results in each corrsponding row in the returned matrix.
         */
        Matrix propagate( const BaseMatrix& inputs ) const;
        /**
         * Propagates through the network using the `inputs` Vector, and stores all of
         * the propagated values internally in `propagated_vals`. The stored values
         * consist of the values before and after the activation for each node.
         * Returns a reference to the final results, which is contained at the end of 
         * `propagated_vals`.
         */
        VVector propagate( 
            const BaseVector& inputs,
            RaggedTensor<3>& propagated_vals
        ) const;

        /**
         * Backpropagates through the network using the `inputs` and `loss_D` to find the 
         * gradients for the weights and bias'.
         * Uses `inputs` to propagate through the network to get the propagated values
         * for each node.
         * `loss_D` is the gradient of the loss function with respect to the output nodes.
         * The gradients for the weights and bias' are placed in `weight_gradients` and
         * `bias_gradients` respectively.
         */
        void backpropagate(
            const BaseVector& inputs, 
            const BaseVector& loss_D, 
            RaggedTensor<3>& weight_gradients,
            RaggedMatrix& bias_gradients
        ) const;
        /**
         * Backpropagates through the network using the precomputed `propagated_vals`
         * and `loss_D` to find the gradients for the weights and bias'.
         * `loss_D` is the gradient of the loss function with respect to the output nodes.
         * The gradients for the weights and bias' are placed in `weight_gradients` and
         * `bias_gradients` respectively.
         */
        void backpropagate(
            const RaggedTensor<3>& propagated_vals, 
            const BaseVector& loss_D, 
            RaggedTensor<3>& weight_gradients,
            RaggedMatrix& bias_gradients
        ) const;

        /**
         * Applies L2 regularization to `weight_gradients` and `bias_gradients` using the
         * `regularization_strength` and `this` NeuralNetwork's current weights and bias'.
         */
        void applyRegularization( 
            RaggedTensor<3>& weight_gradients, 
            RaggedMatrix& bias_gradients, 
            const float regularization_strength
        ) const;

        /* Mutators */
        public:

        /** 
         * Sets random network weights between min and max, and bias' to 0.
         */
        void randomInit( const float min = -1, const float max = 1 );
        /** 
         * Sets random network weights and bias' based on Kaiming initialization
         */
        void kaimingInit();
        /** 
         * Sets random network weights and bias' based on Xavier initialization
         */
        void xavierInit();

        /**
         * Trains `this` NeuralNetwork using the `training_inputs` and
         * `training_expected_outputs`.
         * Uses the `loss_function` and `training_hyperparameters` during training.
         * Returns a Vector containing the calculated losses throughout training.
         */
        Vector train( 
            const BaseMatrix& training_inputs,
            const BaseMatrix& training_expected_outputs,
            const SimpleLossFunction& loss_function = SquaredDiffLossFunction(),
            const Hyperparameters& training_hyperparameters = Hyperparameters()
        );
        /**
         * Trains `this` NeuralNetwork using the the inputs and expected outputs from the
         * `training_data_stream`.
         * Uses the `loss_function` and `training_hyperparameters` during training.
         * Returns a Vector containing the calculated losses throughout training.
         */
        Vector train(
            SimpleDataStream& training_data_stream,
            const SimpleLossFunction& loss_function = SquaredDiffLossFunction(),
            const Hyperparameters& training_hyperparameters = Hyperparameters()
        );

        /* Getters */
        public:
        
        /**
         * Returns the size of the input layer.
         */
        size_t getInputLayerSize() const;
        /**
         * Returns the size of the output layer.
         */
        size_t getOutputLayerSize() const;
        /**
         * Returns the size of the layer at index `layer_index`, where the first layer is
         * the input layer, and the final layer is the output layer.
         */
        size_t getLayerSize( const size_t layer_index ) const;
        /**
         * Returns the number of layers in `this` NeuralNetwork, including
         * the input and output layers.
         */
        size_t getLayerCount() const;
        /**
         * Returns the weights in `this` NeuralNetwork
         */
        const RaggedTensor<3>& getWeights() const;
        /**
         * Returns the bias' in `this` NeuralNetwork
         */
        const RaggedMatrix& getBias() const;
        /**
         * Returns an empty RaggedTensor which is sized to store all propagated values
         * for the propagate() function.
         */
        RaggedTensor<3> getEmptyPropagationTensor() const;

        /**
         * Writes the size of the NeuralNetwork `nn` and its weights and bias' to the
         * stream `fs`.
         */
        friend std::ostream& operator << ( std::ostream& fs, const NeuralNetwork& nn );

        /* NeuralNetwork member variables */
        private:

        /**
         * 
         */
        size_t input_layer_size;
        /**
         * 
         */
        size_t output_layer_size;
        /**
         * 
         */
        jai::RaggedTensor<3> weights;
        /**
         * 
         */
        jai::RaggedMatrix bias;
        /**
         * 
         */
        std::vector<std::unique_ptr<LayerActivation>> layer_activations;
    };
}



/* Implementation */
namespace jai {
    /* Activations Implementation */

    bool Activation::verify( const float min, const float max, const float step ) const {
        // The distance from x used when estimating the slope between x and another point
        const float poll_distance = 1e-2;
        // The tolerance for how far the predicted and expected values can be
        const float tol = 1e-3;

        float x = min;
        while( x <= max ) {
            // Get actual values from functions
            const float y = this->fn(x);
            const float y_D = this->fn_D(x);

            // Check the values of y around x
            const float x_n = x - poll_distance;
            const float x_p = x + poll_distance;
            const float y_n = this->fn(x_n);
            const float y_p = this->fn(x_p);

            // Calculate average slope
            const float predicted_y_D_n = (y - y_n) / (poll_distance);
            const float predicted_y_D_p = (y_p - y) / (poll_distance);
            const float predicted_y_D = (predicted_y_D_n + predicted_y_D_p) / 2.0f;

            // Return false if the predicted and actual values are too far
            if( predicted_y_D - tol > y_D || predicted_y_D + tol < y_D) {
                return false;
            }

            x += step;
        }
        
        return true;
    }

    float LinearActivation::fn( const float x ) const {
        return x;
    }
    float LinearActivation::fn_D( const float x ) const {
        return 1.0f;
    }
    std::unique_ptr<Activation> LinearActivation::clone() const {
        return std::make_unique<Activation>(new LinearActivation(*this));
    }

    float ReLUActivation::fn( const float x ) const {
        return std::max(0.0f, x);
    }
    float ReLUActivation::fn_D( const float x ) const {
        return (float) !std::signbit(x);
    }
    std::unique_ptr<Activation> ReLUActivation::clone() const {
        return std::make_unique<Activation>(new ReLUActivation(*this));
    }

    float ELUActivation::fn( const float x ) const {
        return ( x >=0 ) ?  x : std::exp(x) - 1;
    }
    float ELUActivation::fn_D( const float x ) const {
        return ( x >=0 ) ?  1.0f : std::exp(x);
    }
    std::unique_ptr<Activation> ELUActivation::clone() const {
        return std::make_unique<Activation>(new ELUActivation(*this));
    }

    float SoftplusActivation::fn( const float x ) const {
        return std::log( 1 + std::exp(x) );
    }
    float SoftplusActivation::fn_D( const float x ) const {
        return 1 / (1 + std::exp(-x));
    }
    std::unique_ptr<Activation> SoftplusActivation::clone() const {
        return std::make_unique<Activation>(new SoftplusActivation(*this));
    }

    float SigmoidActivation::fn( const float x ) const {
        return 1.0f / (1 + std::exp(-x));
    }
    float SigmoidActivation::fn_D( const float x ) const {
        const float sigmoid = 1.0f / (1 + std::exp(-x));
        return sigmoid * (1 - sigmoid);
    }
    std::unique_ptr<Activation> SigmoidActivation::clone() const {
        return std::make_unique<Activation>(new SigmoidActivation(*this));
    }

    AugSigmoidActivation::AugSigmoidActivation( const float l, const float u ) {
        if( l > u ) {
            throw std::invalid_argument("The lower bound cannot be greater than the upper bound");
        }
        this->lower_bound = l;
        this->range = u - l;
    }
    float AugSigmoidActivation::fn( const float x ) const {
        return this->range / (1 + std::exp(-x)) + this->lower_bound;
    }
    float AugSigmoidActivation::fn_D( const float x ) const {
        const float sigmoid = 1.0f / (1 + std::exp(-x));
        return this->range * sigmoid * (1 - sigmoid);
    }
    std::unique_ptr<Activation> AugSigmoidActivation::clone() const {
        return std::make_unique<Activation>(new AugSigmoidActivation(*this));
    }

    float TanhActivation::fn( const float x ) const {
        return std::tanh(x);
    }
    float TanhActivation::fn_D( const float x ) const {
        const float tanh = std::tanh(x);
        return 1 - tanh*tanh;
    }
    std::unique_ptr<Activation> TanhActivation::clone() const {
        return std::make_unique<Activation>(new TanhActivation(*this));
    }

    float ExpActivation::fn( const float x ) const {
        return std::exp(x);
    }
    float ExpActivation::fn_D( const float x ) const {
        return std::exp(x);
    }
    std::unique_ptr<Activation> ExpActivation::clone() const {
        return std::make_unique<Activation>(new ExpActivation(*this));
    }

    AugExpActivation::AugExpActivation( const float b ) {
        if( b < 0 ) {
            throw std::invalid_argument("The base of an exponential function cannot be negative");
        }
        this->base = b;
        this->ln_of_base = std::log(b);
    }
    float AugExpActivation::fn( const float x ) const {
        return std::pow(this->base, x);
    }
    float AugExpActivation::fn_D( const float x ) const {
        return this->ln_of_base * std::pow(this->base, x);
    }
    std::unique_ptr<Activation> AugExpActivation::clone() const {
        return std::make_unique<Activation>(new AugExpActivation(*this));
    }

    PowerActivation::PowerActivation( const float p ) {
        this->power = p;
    }
    float PowerActivation::fn( const float x ) const {
        return std::pow(x, this->power);
    }
    float PowerActivation::fn_D( const float x ) const {
        return this->power * std::pow(x, this->power-1);
    }
    std::unique_ptr<Activation> PowerActivation::clone() const {
        return std::make_unique<Activation>(new PowerActivation(*this));
    }
    
    
    /* LayerActivations Implementation */

    bool LayerActivation::verify( 
        const size_t layer_size, const float min, const float max,
        const size_t test_count, const int seed 
    ) const {
        // The distance from x used when estimating the slope between x and another point
        const float poll_distance = 1e-2;
        // The tolerance for how far the predicted and expected values can be
        const float tol = 1e-3;

        Vector x_diff(layer_size, poll_distance);

        const Vector one_vec(layer_size, 1.0f);
        // Run test on a number of Vectors
        for( size_t i = 0; i < test_count; ++i ) {
            // Generate Vector with random values
            jai::Vector x(layer_size);
            std::srand(seed);
            const float range = max - min;
            for( size_t i = 0; i < x.size(); ++i ) {
                x[i] = ((double) std::rand() / RAND_MAX) * range + min;
            }

            // Calculate actual values
            Vector y(layer_size);
            Vector y_D(layer_size);
            this->fn(x, y);
            this->fn_D(x, one_vec, y_D);

            // Check some values around x
            Vector x_n = x - x_diff;
            Vector x_p = x + x_diff;
            Vector y_n;
            Vector y_p;
            this->fn(x_n, y_n);
            this->fn(x_p, y_p);

            // Calculate predicted slopes
            Vector predicted_y_D_n = (y - y_n) / x_diff;
            Vector predicted_y_D_p = (y - y_p) / x_diff;
            Vector predicted_y_D = (predicted_y_D_n + predicted_y_D_p) / 2.0f;

            // Check if the slopes are close enough
            for( size_t i = 0; i < layer_size; ++i ) {
                // Return false if the predicted and actual values are too far
                if( predicted_y_D[i] - tol > y_D[i] || predicted_y_D[i] + tol < y_D[i] ) {
                    return false;
                }
            }
        }

        return true;
    }

    UniformLayerActivation::UniformLayerActivation( const Activation& activation )
        : activation(activation.clone()) { }
    void UniformLayerActivation::fn( const BaseVector& x, BaseVector& y ) const {
        for( size_t i = 0; i < x.size(); ++i ) {
            y[i] = this->activation.get()->fn(x[i]);
        }  
    }
    void UniformLayerActivation::fn_D( const BaseVector& x, const BaseVector& ___, BaseVector& y_D ) const {
        for( size_t i = 0; i < x.size(); ++i ) {
            y_D[i] = this->activation.get()->fn_D(x[i]) * ___[i];
        }
    }
    std::unique_ptr<LayerActivation> UniformLayerActivation::clone() const {
        return std::make_unique<LayerActivation>(
            new UniformLayerActivation(*this->activation.get())
        );
    }
    bool UniformLayerActivation::isValidLayerSize( size_t layer_size ) const {
        return true;
    }

    NonUniformLayerActivation::NonUniformLayerActivation( const std::vector<Activation*>& activations ) {
        for( size_t i = 0; i < activations.size(); ++i ) {
            this->activations.push_back(
                std::make_unique<Activation>(activations[i])
            );
        }
     }
    void NonUniformLayerActivation::fn( const BaseVector& x, BaseVector& y ) const {
        for( size_t i = 0; i < x.size(); ++i ) {
            y[i] = this->activations[i].get()->fn(x[i]);
        }
    }
    void NonUniformLayerActivation::fn_D( const BaseVector& x, const BaseVector& ___, BaseVector& y_D ) const {
        for( size_t i = 0; i < x.size(); ++i ) {
            y_D[i] = this->activations[i].get()->fn_D(x[i]) * ___[i];
        }
    }
    std::unique_ptr<LayerActivation> NonUniformLayerActivation::clone() const {
        // Create copies of activations
        std::vector<std::unique_ptr<Activation>> activation_cpy;
        std::vector<Activation*> activation_cpy_ptrs(this->activations.size());
        for( size_t i = 0; i < this->activations.size(); ++i ) {
            activation_cpy.push_back(this->activations[i]->clone());
            activation_cpy_ptrs[i] = activation_cpy[i].get();
        }

        return std::make_unique<LayerActivation>(
            new NonUniformLayerActivation(activation_cpy_ptrs)
        );
    }
    bool NonUniformLayerActivation::isValidLayerSize( size_t layer_size ) const {
        return layer_size == this->activations.size();
    }

    void SoftmaxLayerActivation::fn( const BaseVector& x, BaseVector& y ) const {
        const size_t layer_size = x.size();
        float sum = 0;
        for(size_t i = 0; i < layer_size; ++i){
            y[i] = std::exp(x[i]);
            sum += y[i];
        }
        for(size_t i = 0; i < layer_size; ++i){
            y[i] = x[i] / sum;
        }
    }
    // TODO: This is wrong
    void SoftmaxLayerActivation::fn_D( const BaseVector& x, const BaseVector& ___, BaseVector& y_D ) const {
        const size_t layer_size = x.size();
        // Find softmax first
        float softmax[layer_size];
        float e_sum = 0;
        for(size_t i = 0; i < layer_size; ++i){
            softmax[i] = std::exp(x[i]);
            e_sum += softmax[i];
        }
        for(size_t i = 0; i < layer_size; ++i){
            softmax[i] = softmax[i] / e_sum;
        }
        // Calculate derivative using softmax
        float sum = 0;
        for( size_t i = 0; i < layer_size; ++i ) {
            sum += softmax[i] * ___[i];
        }
        for( size_t i = 0; i < layer_size; ++i ) {
            y_D[i] += softmax[i] * (___[i] - sum);
        }
    }
    std::unique_ptr<LayerActivation> SoftmaxLayerActivation::clone() const {
        return std::make_unique<LayerActivation>(new SoftmaxLayerActivation());
    }
    bool SoftmaxLayerActivation::isValidLayerSize( size_t layer_size ) const {
        return true;
    }


    /* SimpleLossFunction Implementation */

    float SquaredDiffLossFunction::fn( const BaseVector& x, const BaseVector& expected_x ) const {
        const size_t size = x.size();
        
        float squared_sum = 0;
        for( size_t i = 0; i < size; ++i ) {
            float diff = expected_x[i] - x[i];
            squared_sum += diff * diff;
        }

        return squared_sum / size;
    }
    void SquaredDiffLossFunction::fn_D( const BaseVector& x, const BaseVector& expected_x, BaseVector& y_D ) const {
        const size_t size = x.size();
        for( size_t i = 0; i < size; ++i ) {
            y_D[i] = -2 * (expected_x[i] - x[i]) / size;
        }
    }
    std::unique_ptr<SimpleLossFunction> SquaredDiffLossFunction::clone() const {
        return std::make_unique<SimpleLossFunction>(new SquaredDiffLossFunction());
    }
    bool SquaredDiffLossFunction::isValidLayerSize( size_t layer_size ) const {
        return true;
    }

    float AbsoluteDiffLossFunction::fn( const BaseVector& x, const BaseVector& expected_x ) const {
        const size_t size = x.size();
        
        float absolute_sum = 0;
        for( size_t i = 0; i < size; ++i ) {
            absolute_sum += std::abs(expected_x[i] - x[i]);
        }

        return absolute_sum / size;
    }
    void AbsoluteDiffLossFunction::fn_D( const BaseVector& x, const BaseVector& expected_x, BaseVector& y_D ) const {
        const size_t size = x.size();
        for( size_t i = 0; i < size; ++i ) {
            const float diff = (expected_x[i] - x[i]);
            y_D[i] = (diff > 0)  ?  -1.0  :  1.0;
        }
    }
    std::unique_ptr<SimpleLossFunction> AbsoluteDiffLossFunction::clone() const {
        return std::make_unique<SimpleLossFunction>(new AbsoluteDiffLossFunction());
    }
    bool AbsoluteDiffLossFunction::isValidLayerSize( size_t layer_size ) const {
        return true;
    }

    float CrossEntropyLossFunction::fn( const BaseVector& x, const BaseVector& expected_x ) const {
        const size_t size = x.size();
        
        float sum = 0;
        for( size_t i = 0; i < size; ++i ) {
            sum += expected_x[i] * std::log(x[i]);
        }

        return -sum;
    }
    void CrossEntropyLossFunction::fn_D( const BaseVector& x, const BaseVector& expected_x, BaseVector& y_D ) const {
        const float EPSILON = 0.01f;
        
        const size_t size = x.size();
        for( size_t i = 0; i < size; ++i ) {
            float div = x[i];
            if( div < EPSILON ) div = EPSILON;
            y_D[i] = -expected_x[i] / x[i];
        }
    }
    std::unique_ptr<SimpleLossFunction> CrossEntropyLossFunction::clone() const {
        return std::make_unique<SimpleLossFunction>(new CrossEntropyLossFunction());
    }
    bool CrossEntropyLossFunction::isValidLayerSize( size_t layer_size ) const {
        return true;
    }


    /* NeuralNetwork Helper Functions */
    
    namespace {
        std::vector<size_t> constructLayerSizes( 
            const size_t input_layer_size,
            const size_t output_layer_size,
            const size_t hidden_layer_size, 
            const size_t hidden_layer_count
        ) {
            const size_t layer_count = 2 + hidden_layer_count;
            std::vector<size_t> layer_sizes(layer_count);
            layer_sizes[0] = input_layer_size;
            for( size_t i = 1; i < layer_count - 1; ++i ) {
                layer_sizes[i] = hidden_layer_size;
            }
            layer_sizes[layer_count - 1] = output_layer_size;
        }
    }


    /* NeuralNetwork Implementation */

    NeuralNetwork::NeuralNetwork() { 
        this->input_layer_size = 0;
        this->output_layer_size = 0;
    }
    NeuralNetwork::NeuralNetwork(   
        const size_t input_layer_size, 
        const size_t output_layer_size, 
        const size_t hidden_layer_size, 
        const size_t hidden_layer_count,
        const Activation& hidden_activation, 
        const LayerActivation& output_layer_activation 
    ) : NeuralNetwork(
        2 + hidden_layer_count, 
        constructLayerSizes(
            input_layer_size,
            output_layer_size,
            hidden_layer_size,
            hidden_layer_count
        ).data(),
        hidden_activation,
        output_layer_activation
    ) { }
    NeuralNetwork::NeuralNetwork(
        const size_t layer_count,
        const size_t layer_sizes[],
        const Activation& hidden_activation, 
        const LayerActivation& output_layer_activation
    ) {
        // Check if network sizes are invalid
        if( layer_count < 2 ) {
            throw std::invalid_argument("Must have at least 2 layers.");
        }
        for( size_t i = 0; i < layer_count; ++i ) {
            if( layer_sizes[i] < 1 ) {
                throw std::invalid_argument("All layers must have a size greater than 0. ");
            }
        }
        if( !output_layer_activation.isValidLayerSize(output_layer_size) ) {
            throw std::invalid_argument("Network output layer size does not match output layer activation size.");
        }
        
        // Set sizes
        this->input_layer_size = layer_sizes[0];
        this->output_layer_size = layer_sizes[layer_count - 1];

        // Create weights tensor
        std::vector<size_t[2]> weights_inner_tensor_dims(layer_count - 1);
        for( size_t i = 0; i < layer_count - 1; ++i ) {
            weights_inner_tensor_dims[i][0] = layer_sizes[i + 1];
            weights_inner_tensor_dims[i][1] = layer_sizes[i];
        }
        this->weights = RaggedTensor<3>(layer_count - 1, weights_inner_tensor_dims.data());

        // Create bias' tensor
        std::vector<size_t> bias_inner_tensor_dims(layer_count - 1);
        for( size_t i = 0; i < layer_count - 1; ++i ) {
            bias_inner_tensor_dims[i] = layer_sizes[i + 1];
        }
        this->bias = RaggedMatrix(layer_count - 1, bias_inner_tensor_dims.data());

        // Set activation functions
        for( size_t i = 0; i < layer_count - 2; ++i ) {
            this->layer_activations.push_back(
                std::make_unique<UniformLayerActivation>(UniformLayerActivation(hidden_activation))
            );
        }
        this->layer_activations.push_back(
            output_layer_activation.clone()
        );
    }
    NeuralNetwork::NeuralNetwork(
        const RaggedTensor<3>& weights,
        const RaggedMatrix& bias,
        const Activation& hidden_activation,
        const LayerActivation& output_layer_activation
    ) {
        // Check that the sizes of the weights and bias' match
        if( weights.dim1Size() != bias.dim1Size() ) {
            throw std::invalid_argument("The weights and bias' Tensors must correspond to the same number of layers.");
        }
        for( size_t i = 0; i < bias.dim1Size(); ++i ) {
            if( bias[i].size() != weights[i].size(1) ) {
                throw std::invalid_argument("The bias' and weights Tensor must match.");
            }
        }
        for( size_t i = 0; i < bias.dim1Size() - 1; ++i ) {
            if( bias[i].size() != weights[i + 1].size(0) ) {
                throw std::invalid_argument("The bias' and weights Tensor must match");
            }
        }
        // Check if network sizes are invalid
        if( bias.dim1Size() < 1 ) {
            throw std::invalid_argument("Must have at least 2 layers.");
        }
        for( size_t i = 0; i < bias.dim1Size(); ++i ) {
            if( bias[i].size() < 1 ) {
                throw std::invalid_argument("All layers must have a size greater than 0. ");
            }
        }
        if( !output_layer_activation.isValidLayerSize(output_layer_size) ) {
            throw std::invalid_argument("Network output layer size does not match output layer activation size.");
        }

        // Set sizes
        this->input_layer_size = weights[0].size(0);
        this->output_layer_size = weights[weights.dim1Size() - 1].size(1);

        // Set weights and bias'
        this->weights = weights;
        this->bias = bias;

        // Set activation functions
        const size_t layer_count = weights.dim1Size() + 1;
        for( size_t i = 0; i < layer_count - 2; ++i ) {
            this->layer_activations.push_back(
                std::make_unique<UniformLayerActivation>(UniformLayerActivation(hidden_activation))
            );
        }
        this->layer_activations.push_back(
            output_layer_activation.clone()
        );
    }

    Vector NeuralNetwork::propagate( const BaseVector& inputs ) const {
        Vector y = inputs;
        for( size_t i = 0; i < this->getLayerCount() - 1; ++i ) {
            // Propagate through network
            Vector propagated_x = this->weights[i].mul(y) + this->bias[i];
            // Apply activations
            this->layer_activations[i]->fn(
                propagated_x, 
                y
            );
        }

        return y;
    }
    Matrix NeuralNetwork::propagate( const BaseMatrix& inputs ) const  {
        Matrix outputs({inputs.size(0), this->output_layer_size});
        
        for( size_t i = 0; i < inputs.size(0); ++i ) {
            outputs[i].set( 
                this->propagate(inputs[i]) 
            );
        }

        return outputs;
    }
    VVector NeuralNetwork::propagate(
        const BaseVector& inputs,
        RaggedTensor<3>& propagated_vals
    ) const  {
        // Set the inputs values as the stored values for both sides of the input layer
        propagated_vals[0][0].set(inputs);
        propagated_vals[0][1].set(inputs);

        VVector y = propagated_vals[0][1];
        for( size_t i = 0; i < this->getLayerCount() - 1; ++i ) {
            VMatrix layer_values = propagated_vals[i + 1];
            // Propagate through network
            VVector pre_activation = layer_values[0];
            pre_activation.set(
                this->weights[i].mul(y) + this->bias[i]
            );
            // Apply activations
            VVector post_activation = layer_values[1];
            this->layer_activations[i]->fn(
                pre_activation, 
                post_activation
            );

            y = post_activation;
        }

        return y;
    }

    void NeuralNetwork::backpropagate(
        const BaseVector& inputs, 
        const BaseVector& loss_D, 
        RaggedTensor<3>& weight_gradients,
        RaggedMatrix& bias_gradients
    ) const {
        // Get propagated values
        RaggedTensor<3> propagated_vals = this->getEmptyPropagationTensor();
        this->propagate(inputs, propagated_vals);

        // Do backpropagation
        this->backpropagate(
            propagated_vals,
            loss_D,
            weight_gradients,
            bias_gradients
        );
    }
    void NeuralNetwork::backpropagate(
        const RaggedTensor<3>& propagated_vals, 
        const BaseVector& loss_D, 
        RaggedTensor<3>& weight_gradients,
        RaggedMatrix& bias_gradients
    ) const {
        const size_t layer_count = this->getLayerCount();

        // Calculate gradients for hidden layers
        for( size_t i = layer_count - 2; i > 0; --i ) {
            // If on last layer, use loss_D for post_D
            Vector post_D;
            if (i == layer_count - 2) {
                post_D = loss_D;
            } else {
                VVector delta_i_po = bias_gradients[i + 1];
                post_D = this->weights[i + 1].transpose().mul(delta_i_po);
            }
            VVector delta_i = bias_gradients[i];

            // Calculate layer bias gradients
            const VVector x_i = propagated_vals[i + 1][0];
            this->layer_activations[i]->fn_D(
                x_i,
                post_D,
                // Assign delta_i, which assigns gradient for bias layer
                delta_i
            );

            // Calculate layer weight gradients
            const VVector y_i_mo = propagated_vals[(i + 1) - 1][1];
            weight_gradients[i].set(
                delta_i.mul(y_i_mo.transpose())
            );
        }
    }

    void NeuralNetwork::applyRegularization( 
        RaggedTensor<3>& weight_gradients, 
        RaggedMatrix& bias_gradients, 
        const float regularization_strength
    ) const {
        weight_gradients.addTo( (2 * regularization_strength) * this->weights );
        bias_gradients.addTo( (2 * regularization_strength) * this->bias );
    }

    void NeuralNetwork::randomInit( const float min, const float max ){
        std::random_device rd;
        std::mt19937 rd_gen(rd());
        std::uniform_real_distribution dst(0.0, 1.0);
        
        // Assign weights
        const float range = max - min;
        for( size_t i = 0; i < this->getLayerCount() - 1; ++i ) {
            this->weights[i].transform(
                [min, range, &dst, &rd_gen](const size_t[2], const float) {
                    return (dst(rd_gen) * range) + min;
                }
            );
        }
        // Set bias' to 0
        this->bias.fill(0);
    }
    void NeuralNetwork::kaimingInit(){
        std::random_device rd;
        std::mt19937 rd_gen(rd());
        std::normal_distribution dst(0.0f, 1.0f);

        // Assign weights
        for( size_t i = 0; i < this->getLayerCount() - 1; ++i ) {
            const float bound = std::sqrt(2.0f / this->getLayerSize(i));
            this->weights[i].transform(
                [bound, &dst, &rd_gen](const size_t[2], const float) {
                    return dst(rd_gen) * bound;
                }
            );
        }
        // Set bias' to 0
        this->bias.fill(0);
    }
    void NeuralNetwork::xavierInit() {
        std::random_device rd;
        std::mt19937 rd_gen(rd());
        std::uniform_real_distribution dst(-1.0, 1.0);

        // Assign weights
        for( size_t i = 0; i < this->getLayerCount() - 1; ++i ) {
            const size_t layer_size_sum = this->getLayerSize(i) + this->getLayerSize(i + 1);
            const float bound = 2.44948 / std::sqrt(layer_size_sum);
            this->weights[i].transform(
                [bound, &dst, &rd_gen](const size_t[2], const float) {
                    return dst(rd_gen) * bound; 
                }
            );
        }
        // Set bias' to 0
        this->bias.fill(0);
    }

    Vector NeuralNetwork::train( 
        const BaseMatrix& training_inputs,
        const BaseMatrix& training_expected_outputs,
        const SimpleLossFunction& loss_function = SquaredDiffLossFunction(),
        const Hyperparameters& training_hyperparameters = Hyperparameters()
    ) {
        // Check that the size of the data is valid
        if( training_inputs.size(0) != training_expected_outputs.size(0) ) {
            throw std::invalid_argument("The number of training inputs must be the same as the number of training outputs");
        }
        if( training_inputs.size(0) == 0 ) {
            throw std::invalid_argument("There must be at least one data point");
        }
        // Check that the sizes of the given inputs and outputs match the network
        if( training_inputs.size(1) != this->input_layer_size ) {
            throw std::invalid_argument("The given training inputs must the network input layer size");
        }
        if( training_expected_outputs.size(1) != this->output_layer_size ) {
            throw std::invalid_argument("The given training outputs must the network output layer size");
        }
        // Check that the loss function is the correct size
        if( !loss_function.isValidLayerSize(this->output_layer_size) ) {
            throw std::invalid_argument("The loss function input size must match the network output layer size");
        }

        const Hyperparameters& hp = training_hyperparameters;

        // Store losses in std::vector
        std::vector<float> losses;



        // Create vector with datapoint indexes
        std::vector<int> datapoint_indexes = std::vector<int>(n);
        for(int l = 0; l < n; ++l)
            datapoint_indexes[l] = l;
        // Create random number generator to shuffle indexes
        std::random_device rd;
        std::mt19937 rd_gen(rd());





        return Vector(losses.size(), losses.data());
    }
    Vector NeuralNetwork::train(
        SimpleDataStream& training_data_stream,
        const SimpleLossFunction& loss_function = SquaredDiffLossFunction(),
        const Hyperparameters& training_hyperparameters = Hyperparameters()
    ) {
        // Check that the sizes of the inputs and outputs returned by the data stream match the network
        if( training_data_stream.inputSize() != this->input_layer_size ) {
            throw std::invalid_argument("The given inputs do not match the network input layer size");
        }
        if( training_data_stream.outputSize() != this->output_layer_size ) {
            throw std::invalid_argument("The given outputs do not match the network output layer size");
        }
        // Check that the loss function is the correct size
        if( !loss_function.isValidLayerSize(this->output_layer_size) ) {
            throw std::invalid_argument("The loss function input size does not match the network output layer size");
        }
    }

    size_t NeuralNetwork::getInputLayerSize() const {
        return this->input_layer_size;
    }
    size_t NeuralNetwork::getOutputLayerSize() const {
        return this->output_layer_size;
    }
    size_t NeuralNetwork::getLayerSize( const size_t layer_index ) const {
        return (layer_index == 0) ?
                this->input_layer_size :
                this->bias[layer_index - 1].size();
    }
    size_t NeuralNetwork::getLayerCount() const {
        return 1 + this->bias.dim1Size();
    }
    const RaggedTensor<3>& NeuralNetwork::getWeights() const {
        return this->weights;
    }
    const RaggedMatrix& NeuralNetwork::getBias() const {
        return this->bias;
    }
    RaggedTensor<3> NeuralNetwork::getEmptyPropagationTensor() const {
        const size_t layer_count = this->getLayerCount();
        size_t inner_matrix_sizes[layer_count][2];
        // Set input layer size
        inner_matrix_sizes[0][0] = input_layer_size;
        inner_matrix_sizes[0][1] = 2;
        // Set every layer after
        for( size_t i = 1; i < layer_count; ++i ) {
            inner_matrix_sizes[i][0] = this->bias[i].size();
            inner_matrix_sizes[i][1] = 2;
        }

        return RaggedTensor<3>(layer_count, inner_matrix_sizes);
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
        output_layer_activation.fn_d(prev_node_vals + this_layer_start_index, loss_d, prev_outputs_d);
        for(int i = output_layer_size-1; i >= 0; --i){
            const int node_index = this_layer_start_index + i;
            // Bias
                                        //output_layer_activation[i].fn_d(prev_node_vals[node_index])
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
    jai::Vector NeuralNetwork::batchTrain( const std::vector<float*>& inputs, const std::vector<float*>& actual_outputs, const LossFunction& loss_fn, 
                                       const size_t batch_size, const size_t epochs, const float learning_rate, 
                                       const float regularization_strength, const float momentum_decay, const float sqr_momentum_decay ) {
        return batchTrain(inputs, actual_outputs, SQUARED_DIFF(output_layer_size), batch_size, epochs, learning_rate, regularization_strength, momentum_decay, sqr_momentum_decay);
    }
    jai::Vector NeuralNetwork::batchTrain( const std::vector<float*>& inputs, const std::vector<float*>& actual_outputs, const LossFunction& loss_fn, 
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
        jai::Vector losses;
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

    std::ostream& operator << ( std::ostream& fs, const NeuralNetwork& nn ) {
        // Display layer sizes
        fs << "Layers: ";
        for(int i = 0; i < nn.getLayerCount(); i++) {
            fs << nn.getLayerSize(i) << ' ';
        }
        // Display weights and bias values
        fs << "\nWeights: ";
        for( size_t i = 0; i < nn.weights.totalSize(); ++i ) {
            fs << nn.weights.data()[i] << ' ';
        }
        fs << "\nBias': ";
        for( size_t i = 0; i < nn.bias.totalSize(); ++i ) {
            fs << nn.bias.data()[i] << ' ';
        }

        return fs;
    }
}



#endif