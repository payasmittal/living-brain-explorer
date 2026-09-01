import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ===== CELL 1: Imports (already done above)

# ===== CELL 2: RealisticNeuron
class RealisticNeuron:
    def __init__(self, neuron_id, tau=0.7, bias=0.15):
        self.id = neuron_id
        self.activation = 0.0
        self.tau = tau
        self.bias = bias
        self.fired_last_step = False
        
    def receive_input(self, input_signal):
        self.activation = self.activation * (1 - self.tau) + (input_signal + self.bias) * self.tau
        self.activation = np.clip(self.activation, -1, 1)
    
    def get_output(self):
        return self.activation
    
    def check_firing(self, threshold=0.2):
        self.fired_last_step = (self.activation > threshold)
        return self.fired_last_step
    
    def reset(self):
        self.activation = 0.0
        self.fired_last_step = False

# ===== CELL 3: BiologicalSynapse
class BiologicalSynapse:
    def __init__(self, source, target, initial_weight=None):
        self.source = source
        self.target = target
        
        if initial_weight is None:
            self.weight = np.random.uniform(-0.5, 0.5)
        else:
            self.weight = initial_weight
        
        self.eligibility = 0.0
        self.eligibility_decay = 0.9
        self.eligibility_history = []
        self.weight_history = []
    
    def propagate(self):
        signal = self.source.get_output() * self.weight
        return signal
    
    def update_eligibility(self):
        presynaptic = self.source.get_output()
        postsynaptic = self.target.get_output()
        
        self.eligibility = self.eligibility * self.eligibility_decay
        self.eligibility += presynaptic * postsynaptic
        self.eligibility = np.clip(self.eligibility, -1, 1)
        self.eligibility_history.append(self.eligibility)
    
    def apply_three_factor_learning(self, dopamine, learning_rate=0.05):
        delta_w = learning_rate * self.eligibility * dopamine
        
        self.weight += delta_w
        self.weight = np.clip(self.weight, -1.5, 1.5)
        self.weight_history.append(self.weight)

# ===== CELL 4: BiologicalBrain
class BiologicalBrain:
    def __init__(self, input_size=3, hidden_size=8, output_size=3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.input_neurons = [RealisticNeuron(i, tau=0.7, bias=0.0) for i in range(input_size)]
        self.hidden_neurons = [RealisticNeuron(i + input_size, tau=0.7, bias=0.2) for i in range(hidden_size)]
        self.output_neurons = [RealisticNeuron(i + input_size + hidden_size, tau=0.7, bias=0.15) for i in range(output_size)]
        
        self.synapses = []
        
        for inp in self.input_neurons:
            for hidden in self.hidden_neurons:
                synapse = BiologicalSynapse(inp, hidden)
                self.synapses.append(synapse)
        
        for hidden in self.hidden_neurons:
            for output in self.output_neurons:
                synapse = BiologicalSynapse(hidden, output)
                self.synapses.append(synapse)
        
        for i, h_i in enumerate(self.hidden_neurons):
            for j, h_j in enumerate(self.hidden_neurons):
                if i != j and np.random.rand() < 0.4:
                    synapse = BiologicalSynapse(h_i, h_j)
                    self.synapses.append(synapse)
    
    def step(self, input_signal, iterations=10):
        for i, neuron in enumerate(self.input_neurons):
            neuron.reset()
            neuron.receive_input(input_signal[i])
        
        for neuron in self.hidden_neurons + self.output_neurons:
            neuron.reset()
        
        for iteration in range(iterations):
            hidden_inputs = np.zeros(self.hidden_size)
            output_inputs = np.zeros(self.output_size)
            
            for synapse in self.synapses:
                signal = synapse.propagate()
                
                if synapse.source in self.input_neurons and synapse.target in self.hidden_neurons:
                    idx = self.hidden_neurons.index(synapse.target)
                    hidden_inputs[idx] += signal
                elif synapse.source in self.hidden_neurons and synapse.target in self.output_neurons:
                    idx = self.output_neurons.index(synapse.target)
                    output_inputs[idx] += signal
                elif synapse.source in self.hidden_neurons and synapse.target in self.hidden_neurons:
                    idx = self.hidden_neurons.index(synapse.target)
                    hidden_inputs[idx] += signal
            
            for i, neuron in enumerate(self.hidden_neurons):
                neuron.receive_input(hidden_inputs[i])
                neuron.check_firing(threshold=0.2)
            
            for i, neuron in enumerate(self.output_neurons):
                neuron.receive_input(output_inputs[i])
                neuron.check_firing(threshold=0.2)
        
        for synapse in self.synapses:
            synapse.update_eligibility()
        
        output = np.array([n.get_output() for n in self.output_neurons])
        return output
    
    def train(self, input_signal, correct_action, iterations=10):
        output = self.step(input_signal, iterations=iterations)
        chosen_action = np.argmax(output)
        is_correct = (chosen_action == correct_action)
        
        dopamine = 1.5 if is_correct else -0.3
        
        for synapse in self.synapses:
            synapse.apply_three_factor_learning(dopamine, learning_rate=0.05)
        
        return output, is_correct

# ===== CELL 5: Task
class Task:
    def __init__(self):
        self.rules = {'A': 0, 'B': 1, 'C': 2}
    
    def get_stimulus(self, stimulus_type):
        stim = np.zeros(3)
        if stimulus_type == 'A':
            stim[0] = 1.0
        elif stimulus_type == 'B':
            stim[1] = 1.0
        elif stimulus_type == 'C':
            stim[2] = 1.0
        return stim
    
    def get_correct_action(self, stimulus_type):
        return self.rules[stimulus_type]

# ===== STREAMLIT APP
st.set_page_config(page_title="Living Brain Explorer", layout="wide")

st.title("🧠 Living Brain Explorer")
st.write("### Interactive Neural Network with Three-Factor Learning")

st.sidebar.header("⚙️ Parameters")

hidden_size = st.sidebar.slider("Hidden Neurons", 4, 16, 8)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.15, 0.05, 0.01)
eligibility_decay = st.sidebar.slider("Eligibility Decay", 0.8, 0.99, 0.9, 0.01)
n_trials = st.sidebar.slider("Training Trials", 100, 500, 300)

if st.sidebar.button("🚀 Train Network", key="train"):
    st.write("Training neural network...")
    
    progress_bar = st.progress(0)
    
    brain = BiologicalBrain(input_size=3, hidden_size=hidden_size, output_size=3)
    task = Task()
    
    accuracies = []
    correct_count = 0
    
    for trial in range(n_trials):
        stimulus_type = np.random.choice(['A', 'B', 'C'])
        stimulus = task.get_stimulus(stimulus_type)
        correct_action = task.get_correct_action(stimulus_type)
        
        output, is_correct = brain.train(stimulus, correct_action, iterations=10)
        
        if is_correct:
            correct_count += 1
        
        accuracy = (correct_count / (trial + 1)) * 100
        accuracies.append(accuracy)
        
        progress_bar.progress((trial + 1) / n_trials)
    
    st.success("✅ Training Complete!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Initial Accuracy", f"{accuracies[0]:.1f}%")
    with col2:
        st.metric("Final Accuracy", f"{accuracies[-1]:.1f}%")
    with col3:
        st.metric("Improvement", f"{accuracies[-1] - accuracies[0]:+.1f}%")
    
    st.subheader("📈 Learning Curve")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(accuracies, linewidth=2.5, color='darkblue', label='Network Learning')
    ax.axhline(y=33.3, color='red', linestyle='--', linewidth=2, label='Random Chance (33%)')
    ax.fill_between(range(len(accuracies)), 33.3, accuracies, alpha=0.2)
    ax.set_xlabel('Trial', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Three-Factor Learning: Hebbian + Eligibility Traces + Dopamine', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.subheader("⚡ Synaptic Weights")
    weights = np.array([s.weight for s in brain.synapses])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Weight", f"{np.mean(weights):.3f}")
    with col2:
        st.metric("Weight Range", f"[{np.min(weights):.2f}, {np.max(weights):.2f}]")
    with col3:
        st.metric("Total Synapses", len(brain.synapses))
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(weights, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Synapse Weight', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Synaptic Weights After Training', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    st.pyplot(fig)

st.markdown("---")
st.subheader("ℹ️ About This Simulator")
st.write("""
This neural network learns using three-factor learning rule:
- **Hebbian Plasticity**: Fire together → wire together
- **Eligibility Traces**: Remember recently active synapses
- **Dopamine Reward**: Strengthen paths when correct, weaken when wrong
""")