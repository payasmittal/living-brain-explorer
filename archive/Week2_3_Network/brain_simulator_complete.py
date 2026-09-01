"""
Living Brain Explorer: Complete Brain Simulator
A biologically-inspired neural network with Hebbian learning and dopamine modulation

Author: Your Name
Date: Week 3 of Living Brain Explorer Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# ============================================================================
# PART 1: NEURON CLASS
# ============================================================================

class Neuron:
    """
    A single neuron in the brain network.
    
    Simple model: 
    - Has an activation level (0 to 1)
    - Fires when activation exceeds threshold
    - Activation decays each timestep
    """
    
    def __init__(self, neuron_id, threshold=0.3, decay_rate=0.9):
        """
        Initialize a neuron.
        
        Parameters:
        - neuron_id: unique identifier (0, 1, 2, ...)
        - threshold: activation level needed to fire (~0.3)
        - decay_rate: how fast activation decreases (~0.9)
        """
        self.id = neuron_id
        self.activation = 0.0  # current activation (0 to 1)
        self.threshold = threshold
        self.decay_rate = decay_rate
        self.fired_last_step = False  # did this neuron fire last step?
        
    def receive_input(self, input_signal):
        """
        Add input to this neuron's activation.
        
        Parameters:
        - input_signal: how much input this neuron receives
        
        Example:
        - neuron.receive_input(0.5) → activation increases by 0.5
        """
        self.activation += input_signal
        # Clip to stay in valid range [0, 1]
        self.activation = np.clip(self.activation, 0, 1)
    
    def decay(self):
        """
        Activation naturally decreases over time.
        
        This models: neurons become less active if not stimulated
        """
        self.activation *= self.decay_rate
    
    def check_firing(self):
        """
        Check if neuron fires this timestep.
        
        Returns: True if activation > threshold, False otherwise
        
        Example:
        - If activation = 0.5 and threshold = 0.3 → fires (True)
        - If activation = 0.2 and threshold = 0.3 → doesn't fire (False)
        """
        self.fired_last_step = (self.activation > self.threshold)
        return self.fired_last_step
    
    def reset_activation(self):
        """Reset activation to 0 at start of new timestep."""
        self.activation = 0.0


# ============================================================================
# PART 2: SYNAPSE CLASS
# ============================================================================

class Synapse:
    """
    A connection between two neurons.
    
    Stores:
    - The two neurons it connects
    - The connection strength (weight)
    - History of weight changes (for learning)
    """
    
    def __init__(self, source_neuron, target_neuron, initial_weight=0.1):
        """
        Create a synapse between two neurons.
        
        Parameters:
        - source_neuron: neuron that sends signal
        - target_neuron: neuron that receives signal
        - initial_weight: starting connection strength (~0.1)
        """
        self.source = source_neuron
        self.target = target_neuron
        self.weight = initial_weight
        self.weight_history = []  # track how weight changes over time
        
    def transmit_signal(self):
        """
        Send signal through this synapse.
        
        If source neuron fired, pass signal to target neuron.
        Signal strength = source_activation × synapse_weight
        
        Returns: signal strength to send
        """
        if self.source.fired_last_step:
            # Signal is: source activity × connection strength
            signal = self.source.activation * self.weight
            return signal
        else:
            return 0.0
    
    def apply_hebbian_learning(self, dopamine_boost=1.0):
        """
        Apply Hebbian learning: strengthen if both neurons fired.
        
        Hebbian rule: "neurons that fire together, wire together"
        
        Parameters:
        - dopamine_boost: multiplier for learning rate (1.0 = normal, 2.0 = dopamine boost)
        
        Learning rule:
        if source_fired AND target_fired:
            weight += learning_rate × dopamine_boost
        """
        learning_rate = 0.01  # how much to strengthen per trial
        
        # Check if BOTH neurons fired
        if self.source.fired_last_step and self.target.fired_last_step:
            # Strengthen the synapse
            weight_change = learning_rate * dopamine_boost
            self.weight += weight_change
            
            # Clip weight to reasonable range (don't let it grow infinitely)
            self.weight = np.clip(self.weight, -1.0, 1.0)
        
        # Record weight for analysis
        self.weight_history.append(self.weight)


# ============================================================================
# PART 3: BRAIN CLASS (The Complete Network)
# ============================================================================

class Brain:
    """
    A network of neurons connected by synapses.
    
    Handles:
    - Creating neurons
    - Creating connections between them
    - Running simulation (forward pass)
    - Training (applying Hebbian learning)
    """
    
    def __init__(self, n_neurons=10, connectivity=0.5):
        """
        Create a brain network.
        
        Parameters:
        - n_neurons: how many neurons (10-50 typical)
        - connectivity: what fraction of possible connections exist (0.5 = 50%)
        """
        self.n_neurons = n_neurons
        self.neurons = [Neuron(i) for i in range(n_neurons)]
        self.synapses = []
        
        # Create random connections
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j and np.random.rand() < connectivity:
                    synapse = Synapse(self.neurons[i], self.neurons[j])
                    self.synapses.append(synapse)
        
        print(f"Created brain with {n_neurons} neurons and {len(self.synapses)} synapses")
    
    def step(self, input_signal=None):
        """
        Simulate ONE timestep of brain activity.
        
        Steps:
        1. Reset neuron activations
        2. Add external input (if any)
        3. Propagate signals through synapses
        4. Check which neurons fire
        5. Apply decay
        
        Parameters:
        - input_signal: array of length n_neurons, input to each neuron
        
        Returns:
        - which neurons fired (boolean array)
        """
        # Step 1: Reset activations
        for neuron in self.neurons:
            neuron.reset_activation()
        
        # Step 2: Add external input
        if input_signal is not None:
            for i, neuron in enumerate(self.neurons):
                neuron.receive_input(input_signal[i])
        
        # Step 3: Propagate signals through synapses
        # Each synapse sends signal from source to target
        for synapse in self.synapses:
            signal = synapse.transmit_signal()
            if signal != 0:
                self.neurons[synapse.target.id].receive_input(signal)
        
        # Step 4: Check which neurons fire
        fired = np.array([neuron.check_firing() for neuron in self.neurons])
        
        # Step 5: Apply decay (spontaneous decrease in activation)
        for neuron in self.neurons:
            neuron.decay()
        
        return fired
    
    def train(self, input_signal, expected_output, reward):
        """
        Train the brain on one example using Hebbian learning.
        
        Process:
        1. Run the network forward
        2. Check which neurons fired
        3. Apply Hebbian learning with dopamine boost
        
        Parameters:
        - input_signal: array of input activations
        - expected_output: what output we wanted (not used in basic Hebbian, but for tracking)
        - reward: reward signal (+1 for correct, -1 for wrong, 0 for neutral)
        
        Returns:
        - actual output from brain
        """
        # Run network forward
        output = self.step(input_signal)
        
        # Apply Hebbian learning with dopamine modulation
        # Dopamine boost: +1 for reward, normal for neutral, -0.5 for punishment
        if reward > 0:
            dopamine_boost = 2.0  # reward → amplify learning
        elif reward < 0:
            dopamine_boost = 0.5  # punishment → reduce learning
        else:
            dopamine_boost = 1.0  # neutral → normal learning
        
        # Apply Hebbian rule to all synapses
        for synapse in self.synapses:
            synapse.apply_hebbian_learning(dopamine_boost)
        
        return output
    
    def infer(self, input_signal, steps=10):
        """
        Run brain without learning (inference mode).
        Useful for testing learned behavior.
        
        Parameters:
        - input_signal: input to present
        - steps: how many timesteps to simulate
        
        Returns:
        - final output (which neurons fired)
        """
        for _ in range(steps):
            output = self.step(input_signal)
        return output
    
    def get_weights(self):
        """Get all current synapse weights as a matrix."""
        weights = np.zeros((self.n_neurons, self.n_neurons))
        for synapse in self.synapses:
            weights[synapse.source.id, synapse.target.id] = synapse.weight
        return weights
    
    def get_weight_history(self):
        """Get how synaptic weights changed over time."""
        histories = [synapse.weight_history for synapse in self.synapses]
        return np.array(histories)


# ============================================================================
# PART 4: SIMPLE DECISION TASK (Environment)
# ============================================================================

class DecisionTask:
    """
    Simple learning task: map stimulus to action.
    
    Task: Given one of 3 stimuli (A, B, C), choose correct action (0, 1, 2)
    Reward: +1 if correct, -1 if wrong
    """
    
    def __init__(self):
        """Define the task rules."""
        self.rules = {
            'A': 0,  # if stimulus A → output action 0
            'B': 1,  # if stimulus B → output action 1
            'C': 2   # if stimulus C → output action 2
        }
    
    def create_stimulus(self, stimulus_type):
        """
        Create input vector for stimulus.
        
        Stimulus A = [1, 0, 0]
        Stimulus B = [0, 1, 0]
        Stimulus C = [0, 0, 1]
        """
        stimulus_map = {
            'A': np.array([1.0, 0.0, 0.0]),
            'B': np.array([0.0, 1.0, 0.0]),
            'C': np.array([0.0, 0.0, 1.0])
        }
        return stimulus_map[stimulus_type]
    
    def get_reward(self, stimulus_type, brain_output):
        """
        Score brain's response.
        
        Returns:
        - +1 if correct action chosen
        - -1 if wrong action chosen
        """
        # Which neuron had highest activation? That's the "chosen action"
        if len(brain_output) > 0:
            chosen_action = np.argmax(brain_output)
        else:
            chosen_action = -1
        
        correct_action = self.rules[stimulus_type]
        
        if chosen_action == correct_action:
            return 1.0  # correct!
        else:
            return -1.0  # wrong


# ============================================================================
# PART 5: TRAINING LOOP & ANALYSIS
# ============================================================================

def train_brain_on_task(brain, task, n_trials=100, verbose=True):
    """
    Train brain on the decision task.
    
    Parameters:
    - brain: Brain object to train
    - task: DecisionTask object
    - n_trials: how many training examples
    - verbose: print progress?
    
    Returns:
    - accuracy_history: how accuracy improved over time
    - weight_history: how weights changed
    """
    accuracies = []
    correct_count = 0
    
    # Random sequence of stimuli
    stimuli = np.random.choice(['A', 'B', 'C'], size=n_trials)
    
    for trial, stimulus in enumerate(stimuli):
        # Create input and get correct answer
        input_signal = task.create_stimulus(stimulus)
        correct_action = task.rules[stimulus]
        
        # Train brain
        output = brain.train(input_signal, correct_action, reward=1.0)
        
        # Check if correct (for tracking)
        if len(output) > 0:
            chosen_action = np.argmax(output)
            if chosen_action == correct_action:
                correct_count += 1
        
        # Calculate running accuracy
        accuracy = (correct_count / (trial + 1)) * 100
        accuracies.append(accuracy)
        
        if verbose and (trial + 1) % 20 == 0:
            print(f"Trial {trial + 1}/{n_trials}: Accuracy = {accuracy:.1f}%")
    
    return np.array(accuracies), brain.get_weight_history()


# ============================================================================
# PART 6: VISUALIZATION
# ============================================================================

def plot_training_results(accuracies, weight_history):
    """Plot how brain learned over time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Learning curve
    ax = axes[0]
    ax.plot(accuracies, linewidth=2, color='blue')
    ax.axhline(y=33.3, color='red', linestyle='--', label='Random chance')
    ax.set_xlabel('Training Trial')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Brain Learning Over Time')
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Weight changes for a few synapses
    ax = axes[1]
    # Plot first 20 synapses
    for i in range(min(20, weight_history.shape[0])):
        ax.plot(weight_history[i], alpha=0.6, linewidth=1)
    ax.set_xlabel('Training Trial')
    ax.set_ylabel('Synapse Weight')
    ax.set_title('Individual Synapse Weight Changes (Sample)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# EXAMPLE: HOW TO USE
# ============================================================================

def run_example():
    """
    Complete example: create brain, train it, visualize results.
    
    Run this in Jupyter notebook to see everything work!
    """
    
    print("=" * 60)
    print("Living Brain Explorer: Neural Network Simulator")
    print("=" * 60)
    
    # Create brain
    print("\n1. Creating brain...")
    brain = Brain(n_neurons=10, connectivity=0.4)
    
    # Create task
    print("\n2. Creating task (stimulus-response)...")
    task = DecisionTask()
    
    # Train
    print("\n3. Training brain for 100 trials...")
    accuracies, weight_history = train_brain_on_task(
        brain, task, n_trials=100, verbose=True
    )
    
    # Analyze
    print("\n4. Results:")
    print(f"   - Initial accuracy: {accuracies[0]:.1f}%")
    print(f"   - Final accuracy: {accuracies[-1]:.1f}%")
    print(f"   - Improvement: {accuracies[-1] - accuracies[0]:.1f}%")
    
    # Visualize
    print("\n5. Plotting results...")
    fig = plot_training_results(accuracies, weight_history)
    plt.show()
    
    return brain, task, accuracies, weight_history


# ============================================================================
# END OF FILE
# ============================================================================

if __name__ == "__main__":
    # Run the example when you execute this script
    brain, task, accuracies, weight_history = run_example()
