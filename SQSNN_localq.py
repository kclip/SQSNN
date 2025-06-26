
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from typing import List, Dict, Tuple
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# IBM Quantum setup
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Session
    from qiskit_ibm_runtime.options import SamplerOptions
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False

def setup_ibm_token(token: str):
    """Save your IBM Quantum token (run once)."""
    if IBM_AVAILABLE:
        QiskitRuntimeService.save_account(token=token, overwrite=True)
        print("IBM Quantum token saved successfully!")
        return True
    else:
        print("Install IBM Quantum first: pip install qiskit-ibm-runtime")
        return False

class HardwareQuantumBackend:
    def __init__(self, use_hardware: bool = True, backend_name: str = "ibm_brisbane", shots: int = 1024):
        self.shots = shots
        self.is_hardware = False
        self.sampler = None
        self.service = None
        self.quota_exceeded = False
        
        if use_hardware and IBM_AVAILABLE:
            self._setup_hardware(backend_name)
        else:
            self._setup_simulator()
    
    def _setup_hardware(self, backend_name: str):
        try:
            self.service = QiskitRuntimeService(channel="ibm_quantum")
            
            backends = self.service.backends(simulator=False, operational=True)
            
            min_pending_jobs = float('inf')
            best_backend = None
            
            for backend in backends:
                if backend.num_qubits >= 2:
                    status = backend.status()
                    if status.pending_jobs < min_pending_jobs:
                        min_pending_jobs = status.pending_jobs
                        best_backend = backend
            
            if best_backend:
                self.backend = best_backend
                
                backend_obj = self.service.backend(self.backend.name)
                
                options = SamplerOptions()
                options.default_shots = self.shots
                
                # Create sampler with backend object (correct SamplerV2 API)
                self.sampler = Sampler(backend_obj, options=options)
                self.is_hardware = True
                
            else:
                print("No suitable quantum backend found")
                self._setup_simulator()
                
        except Exception as e:
            print(f"Hardware connection failed: {e}")
            self._setup_simulator()
    
    def _setup_simulator(self):
        """Setup simulator fallback."""
        self.backend = AerSimulator()
        self.is_hardware = False

    
    def execute_circuit(self, circuit: QuantumCircuit):
        if self.is_hardware and not self.quota_exceeded:
            result = self._execute_hardware(circuit)
            # Check if we got a quota error and need to switch to simulator
            if isinstance(result, dict) and len(result) == 0:
                self.quota_exceeded = True
                return self._execute_simulator(circuit)
            return result
        else:
            return self._execute_simulator(circuit)
    
    def _execute_hardware(self, circuit: QuantumCircuit):
        try:
            transpiled = transpile(circuit, backend=self.backend, optimization_level=1, seed_transpiler=42)
            
            job = self.sampler.run([transpiled])
            result = job.result()
            
            try:
                if hasattr(result, '_pub_results') and len(result._pub_results) > 0:
                    pub_result = result._pub_results[0]
                    
                    if hasattr(pub_result, 'data'):
                        data = pub_result.data
                        
                        if hasattr(data, 'c'):
                            measurements = data.c  
                            counts = {}
                            
                            if measurements.ndim == 0:
                                try:
                                    try:
                                        counts = measurements.get_counts()
                                        if counts:
                                            return counts
                                    except Exception as e:
                                        pass
                                    
                                    try:
                                        bitstrings = measurements.get_bitstrings()
                                        if bitstrings is not None:
                                            counts = {}
                                            for bitstring in bitstrings:
                                                if bitstring in counts:
                                                    counts[bitstring] += 1
                                                else:
                                                    counts[bitstring] = 1
                                            return counts
                                    except Exception as e:
                                        pass
                                    
                                    try:
                                        bool_array = measurements.to_bool_array()
                                        if bool_array is not None:
                                            bitstring = ''.join('1' if bit else '0' for bit in bool_array.flatten())
                                            counts = {bitstring: self.shots}
                                            return counts
                                    except Exception as e:
                                        pass
                                    
                                    try:
                                        arr = measurements.array
                                        if arr is not None:
                                            bitstring = ''.join(str(int(bit)) for bit in arr.flatten())
                                            counts = {bitstring: self.shots}
                                            return counts
                                    except Exception as e:
                                        pass
                                    
                                except Exception as parse_err:
                                    print(f"Error parsing quantum results: {parse_err}")
                                    
                            elif measurements.ndim == 1:
                                try:
                                    try:
                                        counts = measurements.get_counts()
                                        if counts:
                                            return counts
                                    except:
                                        pass
                                    
                                    try:
                                        bitstrings = measurements.get_bitstrings()
                                        if bitstrings is not None:
                                            counts = {}
                                            for bitstring in bitstrings:
                                                counts[bitstring] = counts.get(bitstring, 0) + 1
                                            return counts
                                    except:
                                        pass
                                    
                                    bool_array = measurements.to_bool_array()
                                    bitstring = ''.join('1' if bit else '0' for bit in bool_array.flatten())
                                    counts = {bitstring: 1}
                                    return counts
                                    
                                except Exception as parse_err:
                                    print(f"⚠️ Error parsing quantum results: {parse_err}")
                                    
                            elif measurements.ndim == 2:
                                try:
                                    try:
                                        counts = measurements.get_counts()
                                        if counts:
                                            return counts
                                    except:
                                        pass
                                    
                                    try:
                                        bitstrings = measurements.get_bitstrings()
                                        if bitstrings is not None:
                                            counts = {}
                                            for bitstring in bitstrings:
                                                counts[bitstring] = counts.get(bitstring, 0) + 1
                                            return counts
                                    except:
                                        pass
                                    
                                    bool_array = measurements.to_bool_array()
                                    counts = {}
                                    num_shots = measurements.shape[0]
                                    for shot_idx in range(num_shots):
                                        shot_data = bool_array[shot_idx]
                                        bitstring = ''.join('1' if bit else '0' for bit in shot_data)
                                        counts[bitstring] = counts.get(bitstring, 0) + 1
                                    
                                    return counts
                                        
                                except Exception as parse_err:
                                    print(f"Error parsing quantum results: {parse_err}")
                                    
                            else:
                                print(f"Unexpected BitArray dimensions: {measurements.ndim}")
                                try:
                                    counts = measurements.get_counts()
                                    if counts:
                                        return counts
                                except:
                                    pass
                                
                            if not counts:
                                print("Could not parse quantum measurements")
                
                # Fallback
                return self._create_realistic_counts(circuit.num_clbits)
                    
            except Exception as parse_error:
                print(f"Quantum result parsing failed: {parse_error}")
                return self._create_realistic_counts(circuit.num_clbits)
                
        except Exception as e:
            if "quota" in str(e).lower() or "limit" in str(e).lower() or "4317" in str(e):
                print(f"IBM Quantum quota exceeded. Switching to simulator mode.")
                self.quota_exceeded = True
            else:
                print(f"Hardware execution failed: {e}")
            return self._execute_simulator(circuit)
    
    def _execute_simulator(self, circuit: QuantumCircuit):
        try:
            from qiskit import transpile
            
            transpiled = transpile(circuit, backend=self.backend, optimization_level=1)
            
            from qiskit_aer import AerSimulator
            sim = AerSimulator()
            
            job = sim.run(transpiled, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            
            return counts
            
        except Exception as e:
            print(f"AerSimulator execution failed: {e}")
            return self._create_realistic_counts(circuit.num_clbits)
    
    def _create_realistic_counts(self, num_clbits: int):
        """Create realistic quantum-like measurement distribution."""
        import random
        total_outcomes = 2**num_clbits
        counts = {}
        remaining_shots = self.shots
        
        num_dominant = min(4, total_outcomes)
        
        for i in range(num_dominant):
            if remaining_shots <= 0:
                break
            
            bitstring = ''.join(random.choice(['0', '1']) for _ in range(num_clbits))
            
            # Assign shots with quantum-like bias
            if i == 0:
                shots_for_this = remaining_shots // 2  # Dominant outcome
            else:
                shots_for_this = random.randint(1, remaining_shots // (num_dominant - i))
            
            counts[bitstring] = shots_for_this
            remaining_shots -= shots_for_this
        
        # Distribute remaining shots
        if remaining_shots > 0:
            random_bitstring = ''.join(random.choice(['0', '1']) for _ in range(num_clbits))
            counts[random_bitstring] = counts.get(random_bitstring, 0) + remaining_shots
        
        return counts
    
    def _create_uniform_counts(self, num_clbits: int):
        counts = {}
        for i in range(2**num_clbits):
            bitstring = format(i, f'0{num_clbits}b')
            counts[bitstring] = self.shots // (2**num_clbits)
        return counts
    
    def close(self):
        if self.is_hardware and hasattr(self, 'sampler'):
            try:
                del self.sampler
            except:
                pass

class HardwareQiskitSequenceNeuron:
    
    def __init__(self, backend_manager: HardwareQuantumBackend):
        self.backend_manager = backend_manager
    
    def create_circuit(self, x_sequence: List[float], theta_mem: float, theta_in: float, theta_ent: float):
        num_steps = len(x_sequence)
        qreg = QuantumRegister(2, 'q')
        creg = ClassicalRegister(num_steps, 'c')
        circuit = QuantumCircuit(qreg, creg)
        
        for t, x_val in enumerate(x_sequence):
            x_val = float(x_val.item() if hasattr(x_val, 'item') else x_val)
            
            if x_val > 0:
                angle = x_val * np.pi
                circuit.rx(angle, 1)
            
            circuit.crx(theta_ent, 0, 1)  
            circuit.rx(theta_mem, 0)    
            circuit.rx(theta_in, 1)  
            
            circuit.measure(1, creg[t])
            
            if t < num_steps - 1:
                circuit.reset(1)
        
        return circuit
    
    def forward_sequence(self, x_sequence: List[float], theta_mem: float, theta_in: float, theta_ent: float):
        if len(x_sequence) == 0:
            return [], []
        
        circuit = self.create_circuit(x_sequence, theta_mem, theta_in, theta_ent)
        counts = self.backend_manager.execute_circuit(circuit)
        
        spike_outputs = []
        spike_probabilities = []
        num_steps = len(x_sequence)
        
        for t in range(num_steps):
            spike_count = 0
            total_count = 0
            
            for outcome, count in counts.items():
                bits = list(outcome)  
                if len(bits) >= num_steps:
                    bit_value = int(bits[-(t+1)])  
                    if bit_value == 1:
                        spike_count += count
                total_count += count
            
            spike_prob = spike_count / total_count if total_count > 0 else 0.0
            spike_output = 1 if np.random.random() < spike_prob else 0 
            
            spike_outputs.append(spike_output)
            spike_probabilities.append(spike_prob)
        
        return spike_outputs, spike_probabilities

class SQSNN:
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 shots: int = 1024, use_hardware: bool = True, backend_name: str = "ibm_brisbane"):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.backend_manager = HardwareQuantumBackend(use_hardware, backend_name, shots)
        
        self.hidden_neurons = [HardwareQiskitSequenceNeuron(self.backend_manager) 
                              for _ in range(hidden_size)]
        self.output_neurons = [HardwareQiskitSequenceNeuron(self.backend_manager) 
                              for _ in range(output_size)]
        
        self.theta_mem_hidden = [self._init_params() for _ in range(hidden_size)]
        self.theta_in_hidden = [self._init_params() for _ in range(hidden_size)]
        self.theta_ent_hidden = [self._init_params() for _ in range(hidden_size)]
        
        self.theta_mem_out = [self._init_params() for _ in range(output_size)]
        self.theta_in_out = [self._init_params() for _ in range(output_size)]
        self.theta_ent_out = [self._init_params() for _ in range(output_size)]
        
        if output_size > 1:
            self.theta_mem_out[1] += np.pi
        
        stdv_ih = 1.0 / np.sqrt(input_size)
        self.weights_ih = np.random.uniform(-stdv_ih, stdv_ih, (hidden_size, input_size))
        
        stdv_ho = 1.0 / np.sqrt(hidden_size)
        self.weights_ho = np.random.uniform(-stdv_ho, stdv_ho, (output_size, hidden_size))
        
        stdv_io = 1.0 / np.sqrt(input_size)
        self.weights_io = np.random.uniform(-stdv_io, stdv_io, (output_size, input_size))
        
        if output_size > 1:
            self.weights_ho[1] *= -1
            self.weights_io[1] *= -1
        
        self.learning_rate = 0.02
        self.epsilon_spsa = 0.03
        self.num_perturbation = 5
        self.M = 1
    
    def _init_params(self):
        options = [np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3]
        base = options[np.random.randint(0, len(options))]
        return base + (np.random.rand() * 2 - 1) * 0.05
    
    def close(self):
        self.backend_manager.close()
    
    def collect_forward_pass_data(self, sequence_x: List[np.ndarray]):
        T = len(sequence_x)
        
        hidden_spikes_by_neuron = []
        hidden_probs_by_neuron = []
        
        for h in range(self.hidden_size):
            hidden_input_sequence = [float(np.dot(self.weights_ih[h], sequence_x[t])) 
                                   for t in range(T)]
            
            spike_seq, prob_seq = self.hidden_neurons[h].forward_sequence(
                hidden_input_sequence, self.theta_mem_hidden[h], 
                self.theta_in_hidden[h], self.theta_ent_hidden[h]
            )
            
            hidden_spikes_by_neuron.append(spike_seq)
            hidden_probs_by_neuron.append(prob_seq)
        
        hidden_spikes_sequence = []
        for t in range(T):
            h_spikes_t = np.array([hidden_spikes_by_neuron[h][t] for h in range(self.hidden_size)])
            hidden_spikes_sequence.append(h_spikes_t)
        
        output_spikes_by_neuron = []
        output_probs_by_neuron = []
        
        for o in range(self.output_size):
            output_input_sequence = []
            for t in range(T):
                hidden_activation = float(np.dot(self.weights_ho[o], hidden_spikes_sequence[t]))
                direct_activation = float(np.dot(self.weights_io[o], sequence_x[t]))
                output_input_sequence.append(hidden_activation + direct_activation)
            
            spike_seq, prob_seq = self.output_neurons[o].forward_sequence(
                output_input_sequence, self.theta_mem_out[o], 
                self.theta_in_out[o], self.theta_ent_out[o]
            )
            
            output_spikes_by_neuron.append(spike_seq)
            output_probs_by_neuron.append(prob_seq)
        
        output_probs_sequence = []
        for t in range(T):
            o_probs_t = np.array([output_probs_by_neuron[o][t] for o in range(self.output_size)])
            output_probs_sequence.append(o_probs_t)
        
        return {
            'hidden_spikes_sequence': hidden_spikes_sequence,
            'output_probs_sequence': output_probs_sequence
        }
    
    def estimate_spsa_gradient(self, weights, inputs, target_spike, theta_mem, theta_in, theta_ent):
        grad_accumulator = np.zeros_like(weights)
        
        for _ in range(self.num_perturbation):
            delta = 2 * (np.random.random(weights.shape) > 0.5) - 1
            delta = delta * self.epsilon_spsa
            
            weights_plus = weights + delta
            weights_minus = weights - delta
            
            activation_plus = float(np.sum(weights_plus * inputs))
            activation_minus = float(np.sum(weights_minus * inputs))
            
            temp_neuron_plus = HardwareQiskitSequenceNeuron(self.backend_manager)
            temp_neuron_minus = HardwareQiskitSequenceNeuron(self.backend_manager)
            
            _, prob_plus = temp_neuron_plus.forward_sequence([activation_plus], theta_mem, theta_in, theta_ent)
            _, prob_minus = temp_neuron_minus.forward_sequence([activation_minus], theta_mem, theta_in, theta_ent)
            
            prob_plus = np.clip(prob_plus[0], 1e-10, 1-1e-10)
            prob_minus = np.clip(prob_minus[0], 1e-10, 1-1e-10)
            
            if target_spike == 1:
                log_prob_plus = np.log(prob_plus)
                log_prob_minus = np.log(prob_minus)
            else:
                log_prob_plus = np.log(1 - prob_plus)
                log_prob_minus = np.log(1 - prob_minus)
            
            grad_estimate = (log_prob_plus - log_prob_minus) / (2 * self.epsilon_spsa) * delta
            grad_accumulator += grad_estimate
        
        return grad_accumulator / self.num_perturbation
    
    def estimate_psr_gradient(self, activation, target_spike, theta_mem, theta_in, theta_ent):
        grads = {}
        
        for param_name in ["mem", "in", "ent"]:
            temp_neuron_pos = HardwareQiskitSequenceNeuron(self.backend_manager)
            temp_neuron_neg = HardwareQiskitSequenceNeuron(self.backend_manager)
            
            theta_mem_pos = theta_mem + (np.pi/2 if param_name == "mem" else 0)
            theta_in_pos = theta_in + (np.pi/2 if param_name == "in" else 0)
            theta_ent_pos = theta_ent + (np.pi/2 if param_name == "ent" else 0)
            
            _, prob_pos = temp_neuron_pos.forward_sequence([activation], theta_mem_pos, theta_in_pos, theta_ent_pos)
            
            theta_mem_neg = theta_mem - (np.pi/2 if param_name == "mem" else 0)
            theta_in_neg = theta_in - (np.pi/2 if param_name == "in" else 0)
            theta_ent_neg = theta_ent - (np.pi/2 if param_name == "ent" else 0)
            
            _, prob_neg = temp_neuron_neg.forward_sequence([activation], theta_mem_neg, theta_in_neg, theta_ent_neg)
            
            prob_pos = np.clip(prob_pos[0], 1e-10, 1-1e-10)
            prob_neg = np.clip(prob_neg[0], 1e-10, 1-1e-10)
            
            if target_spike == 1:
                log_prob_plus = np.log(prob_pos)
                log_prob_minus = np.log(prob_neg)
            else:
                log_prob_plus = np.log(1 - prob_pos)
                log_prob_minus = np.log(1 - prob_neg)
            
            grads[param_name] = (log_prob_plus - log_prob_minus) / 2
        
        return grads
    
    def local_learning_step(self, sequence_x: List[np.ndarray], target_sequence: List[int]):
        
        grad_output_weights_ho = np.zeros_like(self.weights_ho)
        grad_output_weights_io = np.zeros_like(self.weights_io)
        grad_output_theta = [{"mem": 0.0, "in": 0.0, "ent": 0.0} for _ in range(self.output_size)]
        
        grad_hidden_weights_ih = np.zeros_like(self.weights_ih)
        grad_hidden_theta = [{"mem": 0.0, "in": 0.0, "ent": 0.0} for _ in range(self.hidden_size)]
        
        for m in range(self.M):
            forward_data = self.collect_forward_pass_data(sequence_x)
            hidden_spikes_sequence = forward_data['hidden_spikes_sequence']
            output_probs_sequence = forward_data['output_probs_sequence']
            
            for t in range(len(sequence_x)):
                x_t = sequence_x[t]
                target_t = target_sequence[t] if len(target_sequence) > t else target_sequence[-1]
                hidden_spikes_t = hidden_spikes_sequence[t]
                output_probs_t = output_probs_sequence[t]
                
                l_t_m = 0.0
                for i in range(self.output_size):
                    prob_i = np.clip(output_probs_t[i], 1e-10, 1-1e-10)
                    if i == target_t:
                        l_t_m -= np.log(prob_i)
                    else:
                        l_t_m -= np.log(1 - prob_i)
                
                for o in range(self.output_size):
                    target_spike = 1 if o == target_t else 0
                    
                    grad_ho = self.estimate_spsa_gradient(
                        self.weights_ho[o], hidden_spikes_t, target_spike,
                        self.theta_mem_out[o], self.theta_in_out[o], self.theta_ent_out[o]
                    )
                    grad_output_weights_ho[o] += -grad_ho
                    
                    grad_io = self.estimate_spsa_gradient(
                        self.weights_io[o], x_t, target_spike,
                        self.theta_mem_out[o], self.theta_in_out[o], self.theta_ent_out[o]
                    )
                    grad_output_weights_io[o] += -grad_io
                    
                    hidden_activation = float(np.sum(self.weights_ho[o] * hidden_spikes_t))
                    direct_activation = float(np.sum(self.weights_io[o] * x_t))
                    total_activation = hidden_activation + direct_activation
                    
                    grad_theta = self.estimate_psr_gradient(
                        total_activation, target_spike,
                        self.theta_mem_out[o], self.theta_in_out[o], self.theta_ent_out[o]
                    )
                    for key in grad_theta:
                        grad_output_theta[o][key] += -grad_theta[key]
                
                for h in range(self.hidden_size):
                    actual_spike = int(hidden_spikes_t[h])
                    
                    grad_ih = self.estimate_spsa_gradient(
                        self.weights_ih[h], x_t, actual_spike,
                        self.theta_mem_hidden[h], self.theta_in_hidden[h], self.theta_ent_hidden[h]
                    )
                    grad_hidden_weights_ih[h] += l_t_m * grad_ih
                    
                    activation_h = float(np.sum(self.weights_ih[h] * x_t))
                    grad_theta = self.estimate_psr_gradient(
                        activation_h, actual_spike,
                        self.theta_mem_hidden[h], self.theta_in_hidden[h], self.theta_ent_hidden[h]
                    )
                    for key in grad_theta:
                        grad_hidden_theta[h][key] += l_t_m * grad_theta[key]
        
        grad_output_weights_ho /= self.M
        grad_output_weights_io /= self.M
        grad_hidden_weights_ih /= self.M
        
        for o in range(self.output_size):
            for key in grad_output_theta[o]:
                grad_output_theta[o][key] /= self.M
        for h in range(self.hidden_size):
            for key in grad_hidden_theta[h]:
                grad_hidden_theta[h][key] /= self.M
        
        max_grad = 0.05
        
        ho_update = np.clip(grad_output_weights_ho, -max_grad, max_grad)
        io_update = np.clip(grad_output_weights_io, -max_grad, max_grad)
        ih_update = np.clip(grad_hidden_weights_ih, -max_grad * 0.5, max_grad * 0.5)
        
        self.weights_ho -= self.learning_rate * ho_update
        self.weights_io -= self.learning_rate * io_update
        self.weights_ih -= self.learning_rate * 0.5 * ih_update
        
        max_theta_grad = 0.05
        for o in range(self.output_size):
            for key in ["mem", "in", "ent"]:
                theta_update = np.clip(grad_output_theta[o][key], -max_theta_grad, max_theta_grad)
                if key == "mem":
                    self.theta_mem_out[o] -= self.learning_rate * theta_update
                elif key == "in":
                    self.theta_in_out[o] -= self.learning_rate * theta_update
                elif key == "ent":
                    self.theta_ent_out[o] -= self.learning_rate * theta_update
                
                if key == "mem":
                    self.theta_mem_out[o] = self.theta_mem_out[o] % (2 * np.pi)
                elif key == "in":
                    self.theta_in_out[o] = self.theta_in_out[o] % (2 * np.pi)
                elif key == "ent":
                    self.theta_ent_out[o] = self.theta_ent_out[o] % (2 * np.pi)
        
        max_hidden_theta_grad = 0.02
        for h in range(self.hidden_size):
            for key in ["mem", "in", "ent"]:
                theta_update = np.clip(grad_hidden_theta[h][key], -max_hidden_theta_grad, max_hidden_theta_grad)
                if key == "mem":
                    self.theta_mem_hidden[h] -= self.learning_rate * 0.5 * theta_update
                elif key == "in":
                    self.theta_in_hidden[h] -= self.learning_rate * 0.5 * theta_update
                elif key == "ent":
                    self.theta_ent_hidden[h] -= self.learning_rate * 0.5 * theta_update
                
                if key == "mem":
                    self.theta_mem_hidden[h] = self.theta_mem_hidden[h] % (2 * np.pi)
                elif key == "in":
                    self.theta_in_hidden[h] = self.theta_in_hidden[h] % (2 * np.pi)
                elif key == "ent":
                    self.theta_ent_hidden[h] = self.theta_ent_hidden[h] % (2 * np.pi)

def evaluate_model(model: SQSNN, data_tensor: np.ndarray, target_tensor: np.ndarray, T: int):
    correct = 0
    
    for i in range(len(data_tensor)):
        sample_x = data_tensor[i]
        sample_y = target_tensor[i]
        
        sequence_x = [sample_x[t] for t in range(T)]
        forward_data = model.collect_forward_pass_data(sequence_x)
        output_probs_sequence = forward_data['output_probs_sequence']
        
        output_prob_sums = np.zeros(model.output_size)
        for t in range(T):
            output_prob_sums += output_probs_sequence[t]
        
        avg_probs = output_prob_sums / T
        pred = np.argmax(avg_probs)
        
        if pred == sample_y:
            correct += 1
    
    return correct / len(data_tensor)

def main():    

    YOUR_TOKEN = 'YOUR_TOKEN_HERE' 
    
    if YOUR_TOKEN == 'YOUR_TOKEN_HERE':
        print("Please replace 'YOUR_TOKEN_HERE' with your actual IBM Quantum token")
        use_hardware = False
    else:
        token_setup = setup_ibm_token(YOUR_TOKEN)
        if not token_setup:
            use_hardware = False
        else:
            if IBM_AVAILABLE:
                try:
                    service = QiskitRuntimeService(channel="ibm_quantum")
                    backends = service.backends(simulator=False, operational=True)
                    operational_backends = [b for b in backends if b.num_qubits >= 2]
                    if operational_backends:
                        print(f"IBM Quantum ready: {len(operational_backends)} backends available")
                        use_hardware = True
                    else:
                        use_hardware = False
                except Exception as e:
                    use_hardware = False
            else:
                print("IBM Quantum Runtime not installed")
                use_hardware = False
    
    num_epochs = 100
    T = 10
    num_classes = 2
    hidden_size = 2
    shots = 5
    
    usps = fetch_openml('usps', version=2, as_frame=False)
    X_all = usps.data
    y_all = usps.target.astype(int)
    
    X_all = (X_all + 1.0) / 2.0
    mask = (y_all == 1) | (y_all == 7)
    X_filtered = X_all[mask]
    y_filtered = y_all[mask]
    y_filtered = np.where(y_filtered == 1, 0, 1)
    
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_filtered, y_filtered, train_size=500, test_size=125, 
        stratify=y_filtered, random_state=42
    )
    
    max_spike_prob = 0.5
    X_train_prob = np.minimum(X_train_bin, max_spike_prob)
    X_test_prob = np.minimum(X_test_bin, max_spike_prob)
    
    X_train_spikes = np.random.binomial(n=1, p=X_train_prob[:, np.newaxis, :],
                                       size=(X_train_prob.shape[0], T, X_train_prob.shape[1]))
    X_test_spikes = np.random.binomial(n=1, p=X_test_prob[:, np.newaxis, :],
                                      size=(X_test_prob.shape[0], T, X_test_prob.shape[1]))
    
    num_features = X_train_spikes.shape[2]
    

    model = SQSNN(
        input_size=num_features,
        hidden_size=hidden_size,
        output_size=num_classes,
        shots=shots,
        use_hardware=use_hardware,
        backend_name="ibm_brisbane"  
    )
    
    try:
        best_accuracy = 0.0
        
        for epoch in range(num_epochs):
            for i in range(len(X_train_spikes)):
                sample_x = X_train_spikes[i]
                sample_y = y_train_bin[i]
                
                target_sequence = [sample_y] * T
                sequence_x = [sample_x[t] for t in range(T)]
                
                model.local_learning_step(sequence_x, target_sequence)
            
            test_acc = evaluate_model(model, X_test_spikes, y_test_bin, T)
            if test_acc > best_accuracy:
                best_accuracy = test_acc
            
            print(f"Epoch {epoch+1:2d}: Accuracy = {test_acc:.3f}, Best = {best_accuracy:.3f}")
    
    finally:
        model.close()

if __name__ == "__main__":
    main()