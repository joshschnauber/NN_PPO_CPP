// Simple test of PPO.hpp in the cartpole environment.

/*
 * Compile and run:
 * g++ cartpole.cpp -g -o cp.exe
 * g++ cartpole.cpp -g -Wall -Wextra -pedantic -o cp.exe
 * g++ cartpole.cpp -O3 -o cp.exe
 * cp.exe 10000
 */


#include <iostream>
#include <cmath>
#include <vector>
#include "../PPO.hpp"


const int TARGET = 1000;
const int INPUT_LAYER_SIZE = 4;
const int POLICY_OUTPUT_LAYER_SIZE = 1*(2);


class CartPole {
public:
    // Constructor: Initialize the CartPole environment
    CartPole() : gravity(9.8), cart_mass(1.0), pole_mass(0.1), pole_length(0.5),
                 force_mag(10.0), tau(0.02), theta_threshold(12.0 * M_PI / 180.0), x_threshold(2.4),
                 state({0.0, 0.0, 0.0, 0.0}), done(false) {}

    // Reset the environment to an initial state
    void reset() {
        state = {randomUniform(-0.05, 0.05), randomUniform(-0.05, 0.05),
                 randomUniform(-0.05, 0.05), randomUniform(-0.05, 0.05)};
        done = false;
    }

    // Perform an action (apply force) and update the environment
    void step(int action) {
        if (done) {
            std::cerr << "Environment is done. Reset to continue.\n";
            return;
        }

        float x = state[0];
        float x_dot = state[1];
        float theta = state[2];
        float theta_dot = state[3];

        float force = (action == 1) ? force_mag : -force_mag;
        float costheta = std::cos(theta);
        float sintheta = std::sin(theta);

        float total_mass = cart_mass + pole_mass;
        float pole_mass_length = pole_mass * pole_length;

        // Equations of motion
        float temp = (force + pole_mass_length * theta_dot * theta_dot * sintheta) / total_mass;
        float theta_acc = (gravity * sintheta - costheta * temp) /
                           (pole_length * (4.0 / 3.0 - pole_mass * costheta * costheta / total_mass));
        float x_acc = temp - pole_mass_length * theta_acc * costheta / total_mass;

        // Update state using Euler integration
        x += tau * x_dot;
        x_dot += tau * x_acc;
        theta += tau * theta_dot;
        theta_dot += tau * theta_acc;

        state = {x, x_dot, theta, theta_dot};

        // Check if episode is done
        if (std::abs(x) > x_threshold || std::abs(theta) > theta_threshold) {
            done = true;
        }
    }

    // Get the current state of the environment
    std::vector<float> getState() const {
        return state;
    }
    void getState(float* state){
        for(int i = 0; i < 4; ++i){
            state[i] = this->state[i];
        }
    }

    // Check if the episode is done
    bool isDone() const {
        return done;
    }

private:
    // Environment constants
    const float gravity;
    const float cart_mass;
    const float pole_mass;
    const float pole_length;
    const float force_mag;
    const float tau; // Time step
    const float theta_threshold; // Angle threshold in radians
    const float x_threshold; // Position threshold

    // State variables: [x, x_dot, theta, theta_dot]
    std::vector<float> state;

    // Whether the simulation has ended
    bool done;

    // Utility function: Generate a random number in the range [low, high]
    float randomUniform(float low, float high) {
        return low + static_cast<float>(rand()) / RAND_MAX * (high - low);
    }
};


// Train network in environment.
// Returns false if it fails, and true if it succeeds.
bool train(PPO ppo, const int max_itr, std::mt19937 rd_gen){
    std::cout << "Iterating " << max_itr << " times\n";
    // Initialize environment
    srand(rd_gen());
    CartPole env;
    
    // Run cartpole a fixed number of times
    int k = 0;
    while(++k < max_itr){
        env.reset();
        
        PPO::Episode this_ep(ppo.getStateSize(), ppo.getActionSize()*2, 1000);

        // Run cartpole until done
        int t = 0;
        while (!env.isDone()  &&  ++t <= TARGET) {
            PPO::Episode::Timestate& ts = this_ep.appendTimestate();
            float aug_choice[POLICY_OUTPUT_LAYER_SIZE/2];

            // Get action from policy
            env.getState(ts.getState());
            ppo.getPolicyChoices(ts.getState(), ts.getActions(), ts.getChoices(), aug_choice);

            // Stop training if choice is Nan
            if(std::isnan(aug_choice[0])){
                std::cout << "Nan choice " << k << ' ' << t << "\n";
                std::cout << "Vars " << ts.getChoices()[0] << ' ' << ts.getActions()[0] << ' ' << ts.getActions()[1] << "\n";
                std::cout << "FAULTY Policy:\n" << ppo.getPolicyNetwork() << '\n';
                std::cout << "FAULTY Value:\n" << ppo.getValueNetwork() << '\n';
                std::cout << '\n';
                return false;
            }

            // Move in environment
            int move = (aug_choice[0] < 0) ? 0 : 1;
            env.step(move);

            // Add constant reward
            ts.getReward() = 1;
        }        

        if(t > TARGET){
            std::cout << "Target "<<TARGET<<" achieved in " << k << " iterations\n";
            return true;
        }

        ppo.train(this_ep);
    }

    return true;
}


int main(int argc, char** argv) {
    // Get arguments
    int itr = 100;
    if(argc > 1){
        itr = std::stoi(argv[1]);
    }

    // Initialize hyperparameters
    PPO::Hyperparameters hyperparams;
    hyperparams.value_hidden_layer_count = 1;
    hyperparams.policy_hidden_layer_count = 1;
    hyperparams.value_hidden_layer_size = 8;
    hyperparams.policy_hidden_layer_size = 8;
    hyperparams.value_epochs = 3;
    hyperparams.policy_epochs = 3;

    // Create PPO
    PPO ppo(INPUT_LAYER_SIZE, POLICY_OUTPUT_LAYER_SIZE, hyperparams);
    ppo.initialize();

    // Seed for randomization
    std::random_device rd;
    const std::mt19937 rd_gen(rd());

    // Train policy
    if( !train(ppo, itr, rd_gen) ){

        std::cout << "Training failed; trying again:\n";
        for(int i = 0; i < 5; ++i){
            if( train(ppo, itr, rd_gen) ){
                std::cout << i << " succeeded\n";
            }
            else{
                std::cout << i << " failed\n";
            }
        }

        return EXIT_FAILURE;
    }
    else{
        std::cout << "Training succeeded!\n";
        return EXIT_SUCCESS;
    }
}