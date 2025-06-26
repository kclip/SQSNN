import torch
import numpy as np
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from SQS import SQS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_epochs = 100
T = 10 
num_classes = 2
hidden_size = 2
M = 1  
num_perturbation = 5  
num_shots = 0  
epsilon_spsa = 0.03
learning_rate = 0.02


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

X_train_tensor = torch.tensor(X_train_spikes, dtype=torch.float64).to(device)
X_test_tensor = torch.tensor(X_test_spikes, dtype=torch.float64).to(device)
y_train_tensor = torch.tensor(y_train_bin, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test_bin, dtype=torch.long).to(device)

num_features = X_train_tensor.shape[2]


def initialize_quantum_params():
    options = [np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3]
    base_value = options[torch.randint(0, len(options), (1,)).item()]
    spread = 0.05
    return torch.tensor(base_value + (torch.rand(1).item() * 2 - 1) * spread, 
                        dtype=torch.float64, device=device)


class LocalLearningSQSNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LocalLearningSQSNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.hidden_neurons = nn.ModuleList([SQS(encoding_type="custom", surrogate_type="sigmoid").to(device) 
                                            for _ in range(hidden_size)])
        self.output_neurons = nn.ModuleList([SQS(encoding_type="custom", surrogate_type="sigmoid").to(device) 
                                            for _ in range(output_size)])
        
        self.theta_mem_hidden = [initialize_quantum_params() for _ in range(hidden_size)]
        self.theta_in_hidden = [initialize_quantum_params() for _ in range(hidden_size)]
        self.theta_ent_hidden = [initialize_quantum_params() for _ in range(hidden_size)]
        
        self.theta_mem_out = [initialize_quantum_params() for _ in range(output_size)]
        self.theta_in_out = [initialize_quantum_params() for _ in range(output_size)]
        self.theta_ent_out = [initialize_quantum_params() for _ in range(output_size)]
        
        if output_size > 1:
            self.theta_mem_out[1] = self.theta_mem_out[1] + np.pi
        
        stdv_ih = 1.0 / np.sqrt(input_size)
        self.weights_ih = torch.Tensor(hidden_size, input_size).uniform_(-stdv_ih, stdv_ih).to(device).double()
        
        stdv_ho = 1.0 / np.sqrt(hidden_size)
        self.weights_ho = torch.Tensor(output_size, hidden_size).uniform_(-stdv_ho, stdv_ho).to(device).double()
        
        stdv_io = 1.0 / np.sqrt(input_size)
        self.weights_io = torch.Tensor(output_size, input_size).uniform_(-stdv_io, stdv_io).to(device).double()
        
        if output_size > 1:
            self.weights_ho[1] *= -1
            self.weights_io[1] *= -1
        
    def reset_neurons(self):
        for neuron in self.hidden_neurons:
            neuron.reset_memory()
        for neuron in self.output_neurons:
            neuron.reset_memory()
    
    def forward(self, x_t):
        batch_size = x_t.shape[0]
        x_t = x_t.double()
        
        hidden_activations = torch.matmul(x_t, self.weights_ih.T)
        hidden_spikes = torch.zeros((batch_size, self.hidden_size), dtype=torch.float64, device=device)
        hidden_probs = torch.zeros((batch_size, self.hidden_size), dtype=torch.float64, device=device)
        
        for h in range(self.hidden_size):
            spike, prob = self.hidden_neurons[h](hidden_activations[:, h].unsqueeze(-1), 
                                               self.theta_mem_hidden[h], 
                                               self.theta_in_hidden[h], 
                                               self.theta_ent_hidden[h])
            hidden_spikes[:, h] = spike.squeeze().double()
            hidden_probs[:, h] = prob.squeeze().double()
        
        output_from_hidden = torch.matmul(hidden_spikes, self.weights_ho.T)
        output_direct = torch.matmul(x_t, self.weights_io.T)
        output_activations = output_from_hidden + output_direct
        output_spikes = torch.zeros((batch_size, self.output_size), dtype=torch.float64, device=device)
        output_probs = torch.zeros((batch_size, self.output_size), dtype=torch.float64, device=device)
        
        for o in range(self.output_size):
            spike, prob = self.output_neurons[o](output_activations[:, o].unsqueeze(-1), 
                                               self.theta_mem_out[o], 
                                               self.theta_in_out[o], 
                                               self.theta_ent_out[o])
            output_spikes[:, o] = spike.squeeze().double()
            output_probs[:, o] = prob.squeeze().double()
        
        return hidden_spikes, hidden_probs, output_spikes, output_probs
    
    def estimate_log_prob_gradient_spsa_weights(self, neuron_weights, neuron_inputs, target_spike, theta_mem, theta_in, theta_ent, num_perturbation=num_perturbation, num_shots=num_shots):
        grad_accumulator = torch.zeros_like(neuron_weights)
        for _ in range(num_perturbation):
            delta = 2 * torch.bernoulli(torch.ones_like(neuron_weights) * 0.5) - 1
            delta = delta * epsilon_spsa
            
            temp_neuron_plus = SQS(encoding_type="custom", surrogate_type="sigmoid").to(device)
            temp_neuron_minus = SQS(encoding_type="custom", surrogate_type="sigmoid").to(device)
            
            weights_plus = neuron_weights + delta
            weights_minus = neuron_weights - delta
            
            activation_plus = torch.sum(weights_plus * neuron_inputs)
            activation_minus = torch.sum(weights_minus * neuron_inputs)
            
            _, prob_plus = temp_neuron_plus(activation_plus.unsqueeze(0).unsqueeze(-1), theta_mem, theta_in, theta_ent)
            _, prob_minus = temp_neuron_minus(activation_minus.unsqueeze(0).unsqueeze(-1), theta_mem, theta_in, theta_ent)
            
            prob_plus = torch.clamp(prob_plus.squeeze(), 1e-10, 1-1e-10)
            prob_minus = torch.clamp(prob_minus.squeeze(), 1e-10, 1-1e-10)

            epsilon = 1e-10
            if num_shots > 0:
                p_positive = min(max(prob_plus.item(), epsilon), 1.0 - epsilon)
                p_negative = min(max(prob_minus.item(), epsilon), 1.0 - epsilon)
                
                samples = torch.bernoulli(torch.full((num_shots,), p_positive))
                prob_plus = samples.mean()
                samples = torch.bernoulli(torch.full((num_shots,), p_negative))
                prob_minus = samples.mean()
                
                prob_plus = torch.tensor(min(max(prob_plus.item(), epsilon), 1.0 - epsilon), device=device, dtype=torch.float64)
                prob_minus = torch.tensor(min(max(prob_minus.item(), epsilon), 1.0 - epsilon), device=device, dtype=torch.float64)
            
            if target_spike == 1:
                log_prob_plus = torch.log(prob_plus)
                log_prob_minus = torch.log(prob_minus)
            else:
                log_prob_plus = torch.log(1 - prob_plus)
                log_prob_minus = torch.log(1 - prob_minus)
            
            grad_estimate = (log_prob_plus - log_prob_minus) / (2 * epsilon_spsa) * delta
            grad_accumulator += grad_estimate
        grad_avg = grad_accumulator / num_perturbation
        return grad_avg
    
    def estimate_log_prob_gradient_psr_theta(self, activation, target_spike, theta_mem, theta_in, theta_ent, num_shots=num_shots):
        activation_input = activation.unsqueeze(0).unsqueeze(-1)
        grads = {}
        
        for param_name in ["mem", "in", "ent"]:
            temp_neuron_pos = SQS(encoding_type="custom", surrogate_type="sigmoid").to(device)
            temp_neuron_neg = SQS(encoding_type="custom", surrogate_type="sigmoid").to(device)
            
            theta_mem_pos = theta_mem + (np.pi/2 if param_name == "mem" else 0)
            theta_in_pos = theta_in + (np.pi/2 if param_name == "in" else 0)
            theta_ent_pos = theta_ent + (np.pi/2 if param_name == "ent" else 0)
            
            _, prob_pos = temp_neuron_pos(activation_input, theta_mem_pos, theta_in_pos, theta_ent_pos)
            prob_pos = torch.clamp(prob_pos.squeeze(), 1e-10, 1-1e-10)
            
            theta_mem_neg = theta_mem - (np.pi/2 if param_name == "mem" else 0)
            theta_in_neg = theta_in - (np.pi/2 if param_name == "in" else 0)
            theta_ent_neg = theta_ent - (np.pi/2 if param_name == "ent" else 0)
            
            _, prob_neg = temp_neuron_neg(activation_input, theta_mem_neg, theta_in_neg, theta_ent_neg)
            prob_neg = torch.clamp(prob_neg.squeeze(), 1e-10, 1-1e-10)

            epsilon = 1e-10
            if num_shots > 0:
                p_positive = min(max(prob_pos.item(), epsilon), 1.0 - epsilon)
                p_negative = min(max(prob_neg.item(), epsilon), 1.0 - epsilon)
                
                samples = torch.bernoulli(torch.full((num_shots,), p_positive))
                prob_pos = samples.mean()
                samples = torch.bernoulli(torch.full((num_shots,), p_negative))
                prob_neg = samples.mean()
                
                prob_pos = torch.tensor(min(max(prob_pos.item(), epsilon), 1.0 - epsilon), device=device, dtype=torch.float64)
                prob_neg = torch.tensor(min(max(prob_neg.item(), epsilon), 1.0 - epsilon), device=device, dtype=torch.float64)
            
            if target_spike == 1:
                log_prob_plus = torch.log(prob_pos)
                log_prob_minus = torch.log(prob_neg)
            else:
                log_prob_plus = torch.log(1 - prob_pos)
                log_prob_minus = torch.log(1 - prob_neg)
            
            grad_estimate = (log_prob_plus - log_prob_minus) / 2
            grads[param_name] = grad_estimate
        
        return grads
    
    def local_learning_step(self, sequence_x, target_sequence):
        grad_output_weights_ho = torch.zeros_like(self.weights_ho)
        grad_output_weights_io = torch.zeros_like(self.weights_io)
        grad_output_theta = [{"mem": torch.tensor(0.0, dtype=torch.float64, device=device),
                             "in": torch.tensor(0.0, dtype=torch.float64, device=device),
                             "ent": torch.tensor(0.0, dtype=torch.float64, device=device)} 
                            for _ in range(self.output_size)]
        
        grad_hidden_weights_ih = torch.zeros_like(self.weights_ih)
        grad_hidden_theta = [{"mem": torch.tensor(0.0, dtype=torch.float64, device=device),
                             "in": torch.tensor(0.0, dtype=torch.float64, device=device),
                             "ent": torch.tensor(0.0, dtype=torch.float64, device=device)} 
                            for _ in range(self.hidden_size)]
        
        for m in range(M):
            self.reset_neurons()
            hidden_spikes_sequence = []
            output_probs_sequence = []
            
            for t in range(len(sequence_x)):
                x_t = sequence_x[t].unsqueeze(0).double()
                h_spikes, h_probs, o_spikes, o_probs = self.forward(x_t)
                hidden_spikes_sequence.append(h_spikes.squeeze().clone())
                output_probs_sequence.append(o_probs.squeeze().clone())
            
            for t in range(len(sequence_x)):
                x_t = sequence_x[t].double()
                target_t = target_sequence[t] if len(target_sequence) > t else target_sequence[-1]
                hidden_spikes_t = hidden_spikes_sequence[t]
                output_probs_t = output_probs_sequence[t]
                
                l_t_m = 0.0
                for i in range(self.output_size):
                    prob_i = torch.clamp(output_probs_t[i], 1e-10, 1-1e-10)
                    if i == target_t:
                        l_t_m -= torch.log(prob_i).item()  
                    else:
                        l_t_m -= torch.log(1 - prob_i).item()  
                
                for o in range(self.output_size):
                    target_spike = 1 if o == target_t else 0

                    grad_log_prob_ho = self.estimate_log_prob_gradient_spsa_weights(
                        self.weights_ho[o], hidden_spikes_t, target_spike,
                        self.theta_mem_out[o], self.theta_in_out[o], self.theta_ent_out[o]
                    )

                    grad_output_weights_ho[o] += -grad_log_prob_ho
                    

                    grad_log_prob_io = self.estimate_log_prob_gradient_spsa_weights(
                        self.weights_io[o], x_t, target_spike,
                        self.theta_mem_out[o], self.theta_in_out[o], self.theta_ent_out[o]
                    )

                    grad_output_weights_io[o] += -grad_log_prob_io
                    

                    hidden_activation = torch.sum(self.weights_ho[o] * hidden_spikes_t)
                    direct_activation = torch.sum(self.weights_io[o] * x_t)
                    total_activation = hidden_activation + direct_activation
                    
                    grad_log_prob_theta = self.estimate_log_prob_gradient_psr_theta(
                        total_activation, target_spike,
                        self.theta_mem_out[o], self.theta_in_out[o], self.theta_ent_out[o]
                    )

                    for key in grad_log_prob_theta:
                        grad_output_theta[o][key] += -grad_log_prob_theta[key]
                
                for h in range(self.hidden_size):
                    actual_spike = hidden_spikes_t[h].item()
                    
                    grad_log_prob_ih = self.estimate_log_prob_gradient_spsa_weights(
                        self.weights_ih[h], x_t, actual_spike,
                        self.theta_mem_hidden[h], self.theta_in_hidden[h], self.theta_ent_hidden[h]
                    )

                    grad_hidden_weights_ih[h] += l_t_m * grad_log_prob_ih
                    

                    activation_h = torch.sum(self.weights_ih[h] * x_t)
                    grad_log_prob_theta = self.estimate_log_prob_gradient_psr_theta(
                        activation_h, actual_spike,
                        self.theta_mem_hidden[h], self.theta_in_hidden[h], self.theta_ent_hidden[h]
                    )

                    for key in grad_log_prob_theta:
                        grad_hidden_theta[h][key] += l_t_m * grad_log_prob_theta[key]
        
        grad_output_weights_ho /= M
        grad_output_weights_io /= M
        for o in range(self.output_size):
            for key in grad_output_theta[o]:
                grad_output_theta[o][key] /= M
        
        grad_hidden_weights_ih /= M
        for h in range(self.hidden_size):
            for key in grad_hidden_theta[h]:
                grad_hidden_theta[h][key] /= M
        
        max_grad = 0.05
        
        ho_update = torch.clamp(grad_output_weights_ho, -max_grad, max_grad)
        io_update = torch.clamp(grad_output_weights_io, -max_grad, max_grad)
        ih_update = torch.clamp(grad_hidden_weights_ih, -max_grad * 0.5, max_grad * 0.5)
        
        self.weights_ho.data -= learning_rate * ho_update
        self.weights_io.data -= learning_rate * io_update  
        self.weights_ih.data -= learning_rate * 0.5 * ih_update
        
        max_theta_grad = 0.05
        for o in range(self.output_size):
            theta_mem_update = torch.clamp(grad_output_theta[o]["mem"], -max_theta_grad, max_theta_grad)
            theta_in_update = torch.clamp(grad_output_theta[o]["in"], -max_theta_grad, max_theta_grad)
            theta_ent_update = torch.clamp(grad_output_theta[o]["ent"], -max_theta_grad, max_theta_grad)
            
            self.theta_mem_out[o].data -= learning_rate * theta_mem_update
            self.theta_in_out[o].data -= learning_rate * theta_in_update
            self.theta_ent_out[o].data -= learning_rate * theta_ent_update
            
            self.theta_mem_out[o].data = torch.remainder(self.theta_mem_out[o].data, 2 * np.pi)
            self.theta_in_out[o].data = torch.remainder(self.theta_in_out[o].data, 2 * np.pi)
            self.theta_ent_out[o].data = torch.remainder(self.theta_ent_out[o].data, 2 * np.pi)
        
        max_hidden_theta_grad = 0.02
        for h in range(self.hidden_size):
            theta_mem_update = torch.clamp(grad_hidden_theta[h]["mem"], -max_hidden_theta_grad, max_hidden_theta_grad)
            theta_in_update = torch.clamp(grad_hidden_theta[h]["in"], -max_hidden_theta_grad, max_hidden_theta_grad)
            theta_ent_update = torch.clamp(grad_hidden_theta[h]["ent"], -max_hidden_theta_grad, max_hidden_theta_grad)
            
            self.theta_mem_hidden[h].data -= learning_rate * 0.5 * theta_mem_update
            self.theta_in_hidden[h].data -= learning_rate * 0.5 * theta_in_update
            self.theta_ent_hidden[h].data -= learning_rate * 0.5 * theta_ent_update
            
            self.theta_mem_hidden[h].data = torch.remainder(self.theta_mem_hidden[h].data, 2 * np.pi)
            self.theta_in_hidden[h].data = torch.remainder(self.theta_in_hidden[h].data, 2 * np.pi)
            self.theta_ent_hidden[h].data = torch.remainder(self.theta_ent_hidden[h].data, 2 * np.pi)


def evaluate_model(model, data_tensor, target_tensor):
    correct = 0
    
    for i in range(len(data_tensor)):
        sample_x = data_tensor[i]
        sample_y = target_tensor[i].item()
        
        model.reset_neurons()
        
        output_spike_counts = torch.zeros(model.output_size, dtype=torch.float64, device=device)
        output_prob_sums = torch.zeros(model.output_size, dtype=torch.float64, device=device)
        
        for t in range(T):
            x_t = sample_x[t].unsqueeze(0).double()
            _, _, output_spikes, output_probs = model.forward(x_t)
            output_spike_counts += output_spikes.squeeze()
            output_prob_sums += output_probs.squeeze()
        
        avg_probs = output_prob_sums / T
        
        if output_spike_counts.sum() == 0:
            pred = avg_probs.argmax().item()
        else:
            pred = output_spike_counts.argmax().item()
        
        if pred == sample_y:
            correct += 1
    
    return correct / len(data_tensor)


model = LocalLearningSQSNN(input_size=num_features, hidden_size=hidden_size, output_size=num_classes).to(device)

print("Starting training...")
best_accuracy = 0.0

for epoch in range(num_epochs):
    for i in range(len(X_train_tensor)):
        sample_x = X_train_tensor[i]
        sample_y = y_train_tensor[i].item()
        
        target_sequence = [sample_y] * T
        
        sequence_x = [sample_x[t] for t in range(T)]
        
        model.local_learning_step(sequence_x, target_sequence)
    
    test_acc = evaluate_model(model, X_test_tensor, y_test_tensor)

    
    print(f"Epoch {epoch+1}/{num_epochs}: Test Acc: {test_acc:.3f}, Best: {best_accuracy:.3f}")
        