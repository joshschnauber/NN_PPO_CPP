/* TODO:
 * Implement discrete actions support
 * Allow discrete and continuous actions outputs in one network
 * Merge some separate arrays into one (prev_node_vals&post_node_vals, weights&bias, weight_grad&bias_grad,...)
 * Allow some way to accumulate gradients from backpropagation, to use and modify
 * Use NeuralNetwork hyperparameters in PPO hyperparams, and use them in NeuralNetwork class, maybe
 * Remove sizes from hyperparameters, and make them their own thing... maybe
 * Store loss function in class, so the size check doesn't have to be performed at each call to train() (NOT SURE, maybe setLossFunction() function?)
 * Ensure terminology remains constant, i.e. layer size, layer count, action count, choice, ...
 * Episode/Timestate class could be a lot more efficient
 * Remove need for CircularBuffer class, unnessasary extra file to copy around
 * Implement batchTrain() in NeuralNetwork, and potentially implement momentum and regularization in train(), backpropagate(), and/or batchTrain()
 * 
 */


#ifndef PPO_HPP
#define PPO_HPP

#include "CircularBuffer.hpp"
#include "NeuralNetwork.hpp"
#include <list>
#include <vector>
#include <algorithm>



// Declaration
namespace jai {

    // Class representing all the timestates for one agent.
    class Episode {
        public:
        // One single timestate.
        class Timestate {
            public:
            Timestate();
            private:
            Timestate(const size_t input_size, const size_t output_size);
            
            public:
            float* getState()               { return state.data(); }
            const float* getState() const   { return state.data(); }
            float* getActions()             { return actions.data(); }
            const float* getActions() const { return actions.data(); }
            float* getChoices()             { return choices.data(); }
            const float* getChoices() const { return choices.data(); }
            float& getReward()              { return reward; }
            float getReward() const         { return reward; }

            private:
            std::vector<float> state;   // Inputs to network
            std::vector<float> actions; // Outputs representing a normal distribution
            std::vector<float> choices; // Choices sampled from the distribution
            float reward;   // Raw reward at the timestate

            friend class Episode;
        };

        public:
        Episode(const size_t input_size, const size_t output_size, const size_t max_timestates);
        private:
        Episode(const std::vector<Episode>& episodes);
        
        public:
        // Appends a new timestate to the end of the episode and returns it. To be filled with data by caller.
        Timestate& appendTimestate();
        // Appends a new timestate to the end of the episode and fills it with the provided data.
        void appendTimestate(const float* state, const float* actions, const float* choices, const float reward);
        // Adds the reward to the last timestate. Intended to be used with action repeating.
        void addLastTimestateReward(const float reward);
        const Timestate& operator[](size_t index) const         { return timestates[index]; }
        const Timestate& back() const                           { return timestates.back(); }
        size_t size() const                                     { return timestates.size(); }

        private:
        static size_t sumEpisodeSizes(const std::vector<Episode>& episodes);

        private:
        CircularBuffer<Timestate> timestates;
        size_t input_size;
        size_t output_size;

        friend class PPO;
    };

    class PPO {
        public:
        // Struct containing the hyperparameters for PPO.
        struct Hyperparameters {
            // Network sizes
            size_t value_hidden_layer_size          = 10;
            size_t value_hidden_layer_count         = 1;
            size_t policy_hidden_layer_size         = 10;
            size_t policy_hidden_layer_count        = 1;
            // Reward decay constants
            float reward_reduction                  = 0.990f;
            float reward_smoothing_factor           = 0.950f;
            // Policy training constants
            float clip                              = 0.200f;
            float entropy_bonus                     = 1e-3f;
            // Network update constants
            //NeuralNetwork::Hyperparameters value_hp;
            //NeuralNetwork::Hyperparameters policy_hp;
            size_t value_epochs                     = 1;
            size_t policy_epochs                    = 1;
            size_t value_batch_size                 = SIZE_MAX;
            size_t policy_batch_size                = SIZE_MAX;
            float value_regularization_strength     = 1e-5f;
            float policy_regularization_strength    = 1e-5f;
            float value_momentum_decay              = 0.900f;
            float policy_momentum_decay             = 0.900f;
            float value_sqr_momentum_decay          = 0.999f;
            float policy_sqr_momentum_decay         = 0.999f;
            float value_learning_rate               = 1e-2f;
            float policy_learning_rate              = 1e-2f;
        };

        public:
        PPO();
        // Start PPO with the given inputs and number of actions (Number of outputs is double the action_count).
        // First half of outputs are the mean and the second half is the standard deviation.
        // This is for continuous action spaces.
        PPO( const size_t input_size, const size_t action_count, const Hyperparameters& hyperparams = Hyperparameters() );

        // Randomizes the value and policy network using Kaiming initialization
        void initialize();
        // Sets the hyperparameters for PPO
        void setHyperparameters(const Hyperparameters& hyperparams);
        // Sets the seed for the randomizer
        void seed(const unsigned int seed);
        
        // Trains the AI with the given episode
        void train(const Episode& episode);
        // Trains the AI with all of the given episodes
        void train(const std::vector<Episode>& episodes);

        // Computes the advantages for each timestep
        void computeAdvantages(const Episode& episode, float* advantages) const;
        // Computes the advantages for each timestep using the smoothing factor, and the targets for the value function
        void computeAdvantagesWithSmoothing(const Episode& episode, float* advantages, float* value_targets) const;
        // Compute the discounted reward for each timestep
        void computeDiscountedRewards(const Episode& episode, float* discounted_rewards) const;

        // Gets actions directly from policy, given the state
        void getPolicyOutputs(const float* states, float* actions) const;
        // Samples choices given the mean and stddev from the actions
        // The choice is just the value chosen directly from the mean and stddev.
        void sampleChoices(const float* actions, float* choices);
        // Samples choices given the mean and stddev from the actions, but smooths it based on the previous choice
        void sampleSmoothedChoices(const float* actions, const float* prev_choices, float* choices, const float choice_smoothing = 0.9f);
        // Transform sampled choice into a range between -1 and 1
        void transformSampledChoices(float* choices, float* aug_choices) const;

        // Gets output from policy network and samples choices.
        // Essentially calls getPolicyOutputs(), sampleChoices(), and transformSampledChoices(), in that order.
        // This is all you should really need in most cases.
        void getPolicyChoices(const float* state, float* actions, float* choices, float* aug_choices = nullptr);

        // Getters
        size_t getStateSize() const                     { return policy_network.getInputLayerSize(); }
        size_t getActionSize() const                    { return policy_network.getOutputLayerSize() / 2; }
        size_t getValueUpdateCount() const              { return value_updates; }
        size_t getPolicyUpdateCount() const             { return policy_updates; }
        const NeuralNetwork& getValueNetwork() const    { return value_network; }
        const NeuralNetwork& getPolicyNetwork() const   { return policy_network; }


        private:
        // Trains the value network
        void trainValueNetwork(const Episode& episode, const float* value_target);
        // Trains the policy network
        void trainPolicyNetwork(const Episode& episode, const float* advantages);

        // Returns the probability density at the given choice in a normal distribution
        static float probabilityDensity(const float mean, const float stdev, const float choice);


        public:
        // Store hyperparameters
        Hyperparameters hyperparams;

        private:
        // Network that estimates the reward
        NeuralNetwork value_network; 
        // Previous gradients for value network
        std::vector<float> value_weight_momentum;
        std::vector<float> value_bias_momentum;
        std::vector<float> value_weight_sqr_momentum;
        std::vector<float> value_bias_sqr_momentum;
        // Network that estimates the optimal movement
        NeuralNetwork policy_network;
        // Previous gradients for policy network
        std::vector<float> policy_weight_momentum;
        std::vector<float> policy_bias_momentum;
        std::vector<float> policy_weight_sqr_momentum;
        std::vector<float> policy_bias_sqr_momentum;
        // Count of the number of times each network has been updated
        size_t value_updates;
        size_t policy_updates;
        // Random generator for random sampling and timestep shuffling
        std::mt19937 rnd_gen;
    };

} // end jai


// Implementation
namespace jai {

    // NESTED CLASS DEFINITIONS
    Episode::Timestate::Timestate() { }
    Episode::Timestate::Timestate(const size_t input_size, const size_t output_size) : state(input_size), actions(output_size), choices(output_size/2), reward(0) { }
    Episode::Episode(const size_t input_size, const size_t output_size, const size_t max_timestates) : timestates(max_timestates) {
        this->input_size = input_size;
        this->output_size = output_size;
    }
    Episode::Episode(const std::vector<Episode>& episodes) : timestates(sumEpisodeSizes(episodes)) {
        // Collect all episodes into a larger episode
        const size_t input_size = episodes[0].input_size, output_size = episodes[0].output_size;
        for(int i = 0; i < episodes.size(); ++i){
            if( episodes[i].input_size != input_size  ||  episodes[i].output_size != output_size)
                throw std::invalid_argument("Episodes do not have the same input or output sizes.");

            for(int j = 0; j < episodes[i].size(); ++j){
                timestates.push_back(episodes[i].timestates[j]);
            }
        }
    }

    Episode::Timestate& Episode::appendTimestate(){
        timestates.push_back(Timestate(input_size, output_size));
        return timestates.back();
    }
    void Episode::appendTimestate(const float* state, const float* actions, const float* choices, const float reward){
        timestates.push_back(Timestate(input_size, output_size));
        Timestate& ts = timestates.back();
        // Fill state data
        for(int i = 0; i < input_size; ++i){
            ts.state[i] = state[i];
        }
        // Fill action data
        for(int i = 0; i < output_size; ++i){
            ts.actions[i] = actions[i];
        }
        // Fill choice data
        for(int i = 0; i < output_size; ++i){
            ts.choices[i] = choices[i];
        }
        // Fill reward data
        ts.reward = reward;
    }
    void Episode::addLastTimestateReward(const float reward){
        timestates.back().reward += reward;
    }
    size_t Episode::sumEpisodeSizes(const std::vector<Episode>& episodes){
        size_t sum = 0;
        for(int i = 0; i < episodes.size(); ++i){
            sum += episodes[i].size();
        }
        return sum;
    }

    // CONSTRUCTORS
    PPO::PPO() { }
    PPO::PPO( const size_t input_size, const size_t action_count, const Hyperparameters& hyperparams ) : hyperparams(hyperparams), rnd_gen(std::mt19937(std::random_device{}())){
        // Set LINEAR for mean outputs and EXP for log stddev outputs (to get normal stddev)
        std::vector<Activation> o_activations(action_count*2);
        for(size_t i = 0; i < action_count; ++i)
            o_activations[i] = LINEAR;
        for(size_t i = action_count; i < action_count*2; ++i)
            o_activations[i] = POW_1P1;
        const LayerActivation outputs_activation = ACTIVATIONS(o_activations);

        // Initialize networks
        value_network = NeuralNetwork( input_size, 1, hyperparams.value_hidden_layer_size, hyperparams.value_hidden_layer_count, RELU, ACTIVATION(LINEAR, 1) );
        policy_network = NeuralNetwork( input_size, action_count*2, hyperparams.policy_hidden_layer_size, hyperparams.policy_hidden_layer_count, RELU, outputs_activation);

        // Set previous momentum arrays all to 0
        value_weight_momentum  = std::vector<float>(value_network.getWeightCount()  , 0);
        value_bias_momentum    = std::vector<float>(value_network.getBiasCount()    , 0);
        policy_weight_momentum = std::vector<float>(policy_network.getWeightCount() , 0);
        policy_bias_momentum   = std::vector<float>(policy_network.getBiasCount()   , 0);
        value_weight_sqr_momentum  = std::vector<float>(value_network.getWeightCount()  , 0);
        value_bias_sqr_momentum    = std::vector<float>(value_network.getBiasCount()    , 0);
        policy_weight_sqr_momentum = std::vector<float>(policy_network.getWeightCount() , 0);
        policy_bias_sqr_momentum   = std::vector<float>(policy_network.getBiasCount()   , 0);

        // Set counts to zero
        value_updates = 0;
        policy_updates = 0;
    }

    // INITIALIZATION
    void PPO::initialize(){
        value_network.kaimingInit();
        policy_network.kaimingInit();
    }
    void PPO::setHyperparameters(const Hyperparameters& hyperparams){
        if( hyperparams.value_hidden_layer_size != this->hyperparams.value_hidden_layer_size    ||
            hyperparams.value_hidden_layer_count != this->hyperparams.value_hidden_layer_count  ||
            hyperparams.policy_hidden_layer_size != this->hyperparams.policy_hidden_layer_size  ||
            hyperparams.policy_hidden_layer_count != this->hyperparams.policy_hidden_layer_count ){
            throw std::invalid_argument("Cannot change network size.");
        }
        
        this->hyperparams = hyperparams;
    }
    void PPO::seed(const unsigned int seed){
        rnd_gen = std::mt19937(seed);
    }

    // PPO INTERFACE
    void PPO::train(const Episode& episode){
        if(policy_network.getInputLayerSize() != episode.input_size  ||  policy_network.getOutputLayerSize() != episode.output_size)
            throw std::invalid_argument("Episode input or output size doesn't match PPO input or output size.");

        const int episode_size = episode.size();
        // Find advantages
        float advantages[episode_size];
        float _value_targets[episode_size];
        computeAdvantagesWithSmoothing(episode, advantages, _value_targets);

        // Find discounted rewards
        float discounted_rewards[episode_size];
        computeDiscountedRewards(episode, discounted_rewards);

        // Train both networks
        trainValueNetwork(episode, discounted_rewards);
        trainPolicyNetwork(episode, advantages);
    }
    void PPO::train(const std::vector<Episode>& episodes){
        if(policy_network.getInputLayerSize() != episodes[0].input_size  ||  policy_network.getOutputLayerSize() != episodes[0].output_size)
            throw std::invalid_argument("Episode input or output size doesn't match PPO input or output size.");
        
        if(episodes.size() == 0) 
            return;
        // Just train on the 1 episode if only one exists
        if(episodes.size() == 1){
            this->train(episodes[0]);
            return;
        }
        // Else train on every episode at once, to avoid overprioritizing the first trajectory

        // Push all episodes into one longer episode
        const Episode all_episodes(episodes);

        // Calculate all advantages and discounted rewards for each episode, and place them into one buffer
        float all_advantages[all_episodes.size()];
        float all_discounted_rewards[all_episodes.size()];
        float _value_targets[all_episodes.size()];
        int s_index = 0;
        for(int i = 0; i < episodes.size(); ++i){
            computeAdvantagesWithSmoothing(episodes[i], all_advantages + s_index, _value_targets + s_index);
            computeDiscountedRewards(episodes[i], all_discounted_rewards + s_index);

            s_index += episodes[i].size();
        }

        // The indexes for the advantages and timesteps in all_episodes should correspond with each other at this point
        // Now train on the one big episode
        trainValueNetwork(all_episodes, all_discounted_rewards);
        trainPolicyNetwork(all_episodes, all_advantages);
    }

    void PPO::computeAdvantages(const Episode& episode, float* advantages) const {
        const int timestate_count = episode.size();

        // Store output value for the one output node
        float out[1];

        // Calculate augmented rewards for each timestate, and then find the squared difference in values to the total error
        float timestep_reward = 0;

        // Find value for last timetstep
        const int T = timestate_count-1;
        value_network.propagate(episode[T].getState(), out);
        const float value = (*out);
        advantages[T] = episode[T].getReward() - value;
        timestep_reward += value;

        for(int t = T-1; t >=0; --t){
            // Find reward guess by value network
            value_network.propagate(episode[t].getState(), out);
            const float value = (*out);
            
            // Find augmented reward for this timestep, and save it for previous step
            timestep_reward = episode[t].getReward() + timestep_reward*hyperparams.reward_reduction;
            advantages[t] = timestep_reward - value;
        }
    }
    void PPO::computeAdvantagesWithSmoothing(const Episode& episode, float* advantages, float* value_targets) const {
        const int timestate_count = episode.size();

        // Store values from one timestep ahead
        float future_value = 0;
        float future_advantage = 0;

        for(int t = timestate_count-1; t >=0; --t){
            // Find reward guess by value network
            float value;
            value_network.propagate(episode[t].getState(), &value);
            
            // Find augmented reward for this timestep, and save it for previous step
            const float delta = episode[t].getReward()  +  (hyperparams.reward_reduction*future_value)  -  value;
            advantages[t] = delta  +  (hyperparams.reward_reduction*hyperparams.reward_smoothing_factor)*future_advantage;
            
            // Add the value and advantages for the value target
            value_targets[t] = advantages[t] + value;

            // Set future values
            future_value = value;
            future_advantage = advantages[t];
        }
    }
    void PPO::computeDiscountedRewards(const Episode& episode, float* discounted_rewards) const {
        const int timestate_count = episode.size();

        float prev_reward = 0;
        for(int t = timestate_count-1; t >=0; --t){
            prev_reward = episode[t].getReward() + hyperparams.reward_reduction*prev_reward;
            discounted_rewards[t] = prev_reward;
        }
    }

    // GETTERS
    void PPO::getPolicyOutputs(const float* state, float* actions) const {
        policy_network.propagate(state, actions);
    }
    void PPO::sampleChoices(const float* actions, float* choices) {
        const int half_output_count = this->getActionSize();
        for(int i = 0; i < half_output_count; ++i){
            const float mean = actions[i];
            const float stddev = actions[half_output_count + i];
            // Sample choice from distribution
            std::normal_distribution<float> dst(mean, stddev);
            const float sample = dst(rnd_gen);
            choices[i] = sample;
        }
    }
    void PPO::sampleSmoothedChoices(const float* actions, const float* prev_choices, float* choices, const float choice_smoothing) {
        // Get raw choice
        sampleChoices(actions, choices);

        // Smooth choice from previous choice
        const int half_output_count = this->getActionSize();
        for(int i = 0; i < half_output_count; ++i){
            choices[i] = prev_choices[i]*choice_smoothing  +  (1-choice_smoothing)*choices[i];
        }
    }
    void PPO::transformSampledChoices(float* choices, float* aug_choices) const {
        const int half_output_count = this->getActionSize();
        for(int i = 0; i < half_output_count; ++i){
            // Transformation sample to fit within range
            aug_choices[i] = std::tanh(choices[i]);
        }
    }

    void PPO::getPolicyChoices(const float* state, float* actions, float* choices, float* aug_choices) {
        // Get action from policy
        getPolicyOutputs(state, actions);
        // Sample choice from distribution
        sampleChoices(actions, choices);
        // Transform choice to fit within range
        if(aug_choices != nullptr)
            transformSampledChoices(choices, aug_choices);
    }

    // INTERNAL FUNCTIONS
    void PPO::trainValueNetwork(const Episode& episode, const float* value_target){
        const int timestate_count = episode.size();

        // Store values for each node
        const int node_count = value_network.getPropagateNodeCount();
        float prev_node_vals[node_count];
        float post_node_vals[node_count];
        // Create gradients to increment for each timestep
        const int weight_count = value_network.getWeightCount();
        const int bias_count = value_network.getBiasCount();
        float total_weight_grad[weight_count];
        float total_bias_grad[bias_count];
        
        // Create vector with timestep indexes
        std::vector<int> timestep_indexes = std::vector<int>(timestate_count);
        for(int l = 0; l < timestate_count; ++l)
            timestep_indexes[l] = l;
        
        // Do gradient ascent for a few steps
        for(int k = 0; k < hyperparams.value_epochs; ++k){
            
            // Randomize timestate indexes
            std::shuffle(timestep_indexes.begin(), timestep_indexes.end(), rnd_gen);

            int l = 0;
            while(l < timestate_count){
            
                // Initialize total weights at 0
                for(int i = 0; i < weight_count; i++)
                    total_weight_grad[i] = 0;
                for(int i = 0; i < bias_count; i++)
                    total_bias_grad[i] = 0;

                int c = 0;
                for(; c < hyperparams.value_batch_size  &&  l < timestate_count; ++l, ++c){
                    const int t = timestep_indexes[l];
                    
                    // Find reward guess by value network
                    value_network.propagateStore(episode[t].getState(), prev_node_vals, post_node_vals);
                    const float value = post_node_vals[node_count-1];
                    // Calculate loss derivative
                    const float loss_D = -2 * (value_target[t] - value);
                    // Find weights for current values
                    float weight_grad[weight_count];
                    float bias_grad[bias_count];
                    value_network.backpropagateStore(episode[t].getState(), prev_node_vals, post_node_vals, &loss_D, weight_grad, bias_grad);

                    // Add weights to totals
                    for(int i = 0; i < weight_count; ++i)
                        total_weight_grad[i] += weight_grad[i];
                    for(int i = 0; i < bias_count; ++i)
                        total_bias_grad[i] += bias_grad[i];
                }

                // Average total weights and add momentum
                for(int i = 0; i < weight_count; ++i){
                    total_weight_grad[i] /= c;
                }
                for(int i = 0; i < bias_count; ++i){
                    total_bias_grad[i] /= c;
                }
                // Account for regularization in gradient
                if( hyperparams.value_regularization_strength > 0)
                    value_network.applyRegularization(total_weight_grad, total_bias_grad, hyperparams.value_regularization_strength); 
                // Add momentum and root mean square prop
                if( hyperparams.value_momentum_decay > 0  ||  hyperparams.value_sqr_momentum_decay > 0) {
                    for(int i = 0; i < weight_count; ++i){
                        value_weight_momentum[i] =      value_weight_momentum[i]*hyperparams.value_momentum_decay         +  total_weight_grad[i]*(1-hyperparams.value_momentum_decay);
                        value_weight_sqr_momentum[i] =  value_weight_sqr_momentum[i]*hyperparams.value_sqr_momentum_decay +  total_weight_grad[i]*total_weight_grad[i]*(1-hyperparams.value_sqr_momentum_decay);
                        float m_hat =       value_weight_momentum[i]        / (1 - std::pow(hyperparams.value_momentum_decay, value_updates+1));
                        float sqr_m_hat =   value_weight_sqr_momentum[i]    / (1 - std::pow(hyperparams.value_sqr_momentum_decay, value_updates+1));

                        total_weight_grad[i] = m_hat / (std::sqrt(sqr_m_hat) + N_EPSILON);
                    }
                    for(int i = 0; i < bias_count; ++i){
                        value_bias_momentum[i] =      value_bias_momentum[i]*hyperparams.value_momentum_decay         +  total_bias_grad[i]*(1-hyperparams.value_momentum_decay);
                        value_bias_sqr_momentum[i] =  value_bias_sqr_momentum[i]*hyperparams.value_sqr_momentum_decay +  total_bias_grad[i]*total_bias_grad[i]*(1-hyperparams.value_sqr_momentum_decay);
                        float m_hat =       value_bias_momentum[i]        / (1 - std::pow(hyperparams.value_momentum_decay, value_updates+1));
                        float sqr_m_hat =   value_bias_sqr_momentum[i]    / (1 - std::pow(hyperparams.value_sqr_momentum_decay, value_updates+1));

                        total_bias_grad[i] = m_hat / (std::sqrt(sqr_m_hat) + N_EPSILON);
                    }
                }
                // Apply gradients
                value_network.applyGradients(total_weight_grad, total_bias_grad, hyperparams.value_learning_rate);
                // Update count
                value_updates ++;
            }
        }
    }
    void PPO::trainPolicyNetwork(const Episode& episode, const float* advantages){
        const int timestate_count = episode.size();

        // Store starting index for outputs
        const int output_index = policy_network.getBiasLayerIndex(policy_network.getHiddenLayerCount());

        // Create gradients to increment for each timestep
        const int weight_count = policy_network.getWeightCount();
        const int bias_count = policy_network.getBiasCount();
        float total_weight_grad[weight_count];
        float total_bias_grad[bias_count];
        // Store weights
        const int node_count = policy_network.getPropagateNodeCount();
        float prev_node_vals[node_count];
        float post_node_vals[node_count];

        // Store probabilites for old policy
        const int half_output_count = this->getActionSize();
        float prob_old[timestate_count][half_output_count];
        for(int t = 0; t < timestate_count; ++t){
            for(int i = 0; i < half_output_count; ++i){
                prob_old[t][i] = probabilityDensity(episode[t].getActions()[i], episode[t].getActions()[half_output_count + i], episode[t].getChoices()[i]);
            }
        }

        // Create vector with timestep indexes
        std::vector<int> timestep_indexes = std::vector<int>(timestate_count);
        for(int l = 0; l < timestate_count; ++l)
            timestep_indexes[l] = l;


        // Do gradient ascent for a few steps
        for(int k = 0; k < hyperparams.policy_epochs; ++k){

            // Randomize timestate indexes
            std::shuffle(timestep_indexes.begin(), timestep_indexes.end(), rnd_gen);

            // For each time we iterate on the episode, update the gradient a few times
            int l = 0;
            while(l < timestate_count){
                
                // Initialize total weights at 0
                for(int i = 0; i < weight_count; i++)
                    total_weight_grad[i] = 0;
                for(int i = 0; i < bias_count; i++)
                    total_bias_grad[i] = 0;

                // Iterate through each timestep
                int c = 0;
                for(; c < hyperparams.policy_batch_size  &&  l < timestate_count; ++l, ++c){
                    const int t = timestep_indexes[l];

                    // Find weights for current values
                    float weight_grad[weight_count];
                    float bias_grad[bias_count];

                    // Get outputs for the new policy
                    policy_network.propagateStore(episode[t].getState(), prev_node_vals, post_node_vals);
                    float *out = post_node_vals + output_index;

                    // Find probability of this action with the updated policy
                    float prob_new[half_output_count];
                    float r[half_output_count];
                    for(int i = 0; i < half_output_count; ++i){
                        prob_new[i] = probabilityDensity(out[i], out[half_output_count + i], episode[t].getChoices()[i]);
                        r[i] = prob_new[i] / prob_old[t][i];
                    }

                    // Find the gradient of the clipped loss function
                    float loss_D[half_output_count*2];
                    for(int i = 0; i < half_output_count; ++i){ 
                        // Clip probability ratio
                        if( advantages[t] >= 0  &&      r[i] >= (1+hyperparams.clip) ){
                            loss_D[i] =                         0;// Mean
                            loss_D[half_output_count + i] =     0;// Standard deviation
                        }
                        else if( advantages[t] < 0  &&  r[i] <= (1-hyperparams.clip) ){
                            loss_D[i] =                         0; // Mean
                            loss_D[half_output_count + i] =     0; // Standard deviation
                        }
                        // If old ratio is nan or infinity, don't use it
                        else if( std::isnan(r[i])  ||  std::isinf(r[i]) ){
                            loss_D[i] =                         0; // Mean
                            loss_D[half_output_count + i] =     0; // Standard deviation
                        }

                        // If no clip, do normal derivative
                        else{
                            const float mean =  out[i];
                            const float stddev = out[half_output_count + i];
                            const float stddev_2 = stddev*stddev;
                            const float mean_diff = episode[t].getChoices()[i] - mean;

                            // Mean
                            loss_D[i] =                     (advantages[t]) * (r[i]) * (mean_diff / stddev_2);
                            // Standard deviation: Loss + entropy
                            loss_D[half_output_count + i] = (advantages[t]) * (r[i]) * (((mean_diff*mean_diff) / (stddev_2*stddev)) - (1/stddev))
                                                            + hyperparams.entropy_bonus * (1/stddev);
                        }
                    }

                    // Backpropagate with the given loss derivative
                    policy_network.backpropagateStore(episode[t].getState(), prev_node_vals, post_node_vals, loss_D, weight_grad, bias_grad);

                    // Add weights to totals
                    for(int i = 0; i < weight_count; ++i)
                        total_weight_grad[i] += weight_grad[i];
                    for(int i = 0; i < bias_count; ++i)
                        total_bias_grad[i] += bias_grad[i];
                
                }   // End minibatch

                // Average total weights and add momentum
                for(int i = 0; i < weight_count; ++i){
                    total_weight_grad[i] /= c;
                }
                for(int i = 0; i < bias_count; ++i){
                    total_bias_grad[i] /= c;
                }
                // Account for regularization in gradient
                if(hyperparams.policy_regularization_strength > 0)
                    policy_network.applyRegularization(total_weight_grad, total_bias_grad, -hyperparams.policy_regularization_strength); 
                // Add momentum and root mean square prop
                if( hyperparams.policy_momentum_decay > 0  ||  hyperparams.policy_sqr_momentum_decay > 0) {
                    for(int i = 0; i < weight_count; ++i){
                        policy_weight_momentum[i] =      policy_weight_momentum[i]*hyperparams.policy_momentum_decay         +  total_weight_grad[i]*(1-hyperparams.policy_momentum_decay);
                        policy_weight_sqr_momentum[i] =  policy_weight_sqr_momentum[i]*hyperparams.policy_sqr_momentum_decay +  total_weight_grad[i]*total_weight_grad[i]*(1-hyperparams.policy_sqr_momentum_decay);
                        float m_hat =       policy_weight_momentum[i]        / (1 - std::pow(hyperparams.policy_momentum_decay, policy_updates+1));
                        float sqr_m_hat =   policy_weight_sqr_momentum[i]    / (1 - std::pow(hyperparams.policy_sqr_momentum_decay, policy_updates+1));

                        total_weight_grad[i] = m_hat / (std::sqrt(sqr_m_hat) + N_EPSILON);
                    }
                    for(int i = 0; i < bias_count; ++i){
                        policy_bias_momentum[i] =      policy_bias_momentum[i]*hyperparams.policy_momentum_decay         +  total_bias_grad[i]*(1-hyperparams.policy_momentum_decay);
                        policy_bias_sqr_momentum[i] =  policy_bias_sqr_momentum[i]*hyperparams.policy_sqr_momentum_decay +  total_bias_grad[i]*total_bias_grad[i]*(1-hyperparams.policy_sqr_momentum_decay);
                        float m_hat =       policy_bias_momentum[i]        / (1 - std::pow(hyperparams.policy_momentum_decay, policy_updates+1));
                        float sqr_m_hat =   policy_bias_sqr_momentum[i]    / (1 - std::pow(hyperparams.policy_sqr_momentum_decay, policy_updates+1));

                        total_bias_grad[i] = m_hat / (std::sqrt(sqr_m_hat) + N_EPSILON);
                    }
                }
                // Apply gradients
                policy_network.applyGradients(total_weight_grad, total_bias_grad, -hyperparams.policy_learning_rate);
                // Update count
                policy_updates ++;
            }
        } // End episode
    }

    float PPO::probabilityDensity(const float mean, const float stdev, const float choice){
        const float cmm = choice - mean;
        // If the stdev is 0, then the probability function is 0, unless the choice equals the mean
        // This avoids Nan results, since inf*0 = nan
        if(stdev <= N_EPSILON){
            if(std::abs(cmm) <= N_EPSILON)
                return N_INFINITY; 
            return 0;
        }
            
        // 1/(stdev*sqrt(2*PI))
        const float i_std = (1.0f/2.5066283f) * (1.0f/stdev);
        // e^(-(choice - mean)^2/(2stdev^2))
        const float e_p = std::exp(-(cmm*cmm) / (2*stdev*stdev));
        
        return i_std * e_p;
    }

} // end jai


#endif