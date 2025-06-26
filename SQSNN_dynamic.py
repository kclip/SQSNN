from neurodata.load_data import create_dataloader
import tables
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from SQS import QuantumSpikingLayer

dataset_path = './mnist_dvs/mnist_dvs_events.hdf5'
dataset = tables.open_file(dataset_path)


dtype = torch.float32
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_dataset, test_dataset = create_dataloader(dataset_path, batch_size=1, size=[np.prod([1, 26, 26])], 
                                              classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                              sample_length_train=2000 * 1000, sample_length_test=2000 * 1000, 
                                              dt=25000, polarity=0, ds=1, shuffle_test=True, num_workers=0)


batch_size = 64 
surrogate_type = "sigmoid"
num_steps = 20
num_inputs = 676
num_hidden = 256
num_outputs = 10
num_epochs = 20
num_runs = 10

# Lambda values to test
spike_reg_lambda = 5e-10
spike_reg_type = "membrane_sparsity"

def custom_collate(batch):
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    stacked_data = torch.stack(data, dim=1).to(dtype)
    stacked_targets = torch.stack(targets, dim=0).to(dtype)
    
    return stacked_data, stacked_targets

def calculate_membrane_sparsity_regularization(mem1_rec, mem2_rec):
    reg_loss = torch.tensor(0.0, dtype=dtype, device=mem1_rec.device)
    
    for t in range(mem1_rec.size(0)):
        mem1_pos = torch.clamp(mem1_rec[t], min=0.0)
        if mem1_pos.sum() > 0:
            l1_norm = mem1_pos.abs().sum()
            l2_norm_sq = (mem1_pos ** 2).sum()
            if l2_norm_sq > 0:
                reg_loss += (l1_norm ** 2) / l2_norm_sq
        
        mem2_pos = torch.clamp(mem2_rec[t], min=0.0)
        if mem2_pos.sum() > 0:
            l1_norm = mem2_pos.abs().sum()
            l2_norm_sq = (mem2_pos ** 2).sum()
            if l2_norm_sq > 0:
                reg_loss += (l1_norm ** 2) / l2_norm_sq
    
    return reg_loss

class Net(nn.Module):
    def __init__(self, num_steps):
        super().__init__()
        self.num_steps = num_steps
        # Use float32 to match QuantumSpikingLayer
        self.linear1 = nn.Linear(num_inputs, num_hidden).float()
        self.q_hidden = QuantumSpikingLayer(num_hidden, surrogate_type=surrogate_type)
        self.linear2 = nn.Linear(num_hidden, num_outputs).float()
        self.q_output = QuantumSpikingLayer(num_outputs, surrogate_type=surrogate_type)

    def forward(self, x):
        self.q_hidden.reset_memory()
        self.q_output.reset_memory()

        spk1_rec = []
        spk2_rec = []
        mem1_rec = []
        mem2_rec = []

        for t in range(min(self.num_steps, x.size(0))):
            x_t = x[t]
            h1 = self.linear1(x_t)
            h1_spk, mem1 = self.q_hidden(h1)
            h2 = self.linear2(h1_spk)
            out_spk, mem2 = self.q_output(h2)
            spk2_rec.append(out_spk)
            spk1_rec.append(h1_spk)
            mem1_rec.append(mem1)
            mem2_rec.append(mem2)
            
        return torch.stack(spk2_rec, dim=0), torch.stack(spk1_rec, dim=0), torch.stack(mem2_rec, dim=0), torch.stack(mem1_rec, dim=0)

    
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

net = Net(num_steps).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=5e-4)


# Training loop
for epoch in range(num_epochs):
    net.train()
    
    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)
        
        true_labels = torch.argmax(targets[:, :, 0], dim=1)
        
        optimizer.zero_grad()

        spk2_rec, spk1_rec, mem2_rec, mem1_rec = net(data)
        
        classification_loss = criterion(spk2_rec.sum(dim=0)/num_steps, true_labels)
        
        spike_reg_loss = calculate_membrane_sparsity_regularization(mem1_rec, mem2_rec)
        total_loss = classification_loss + spike_reg_lambda * spike_reg_loss
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        
        optimizer.step()
            

    net.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for test_data, test_targets in test_loader:
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)
            
            test_labels = torch.argmax(test_targets[:, :, 0], dim=1)
            
            test_spk, hidden_spk, _, _ = net(test_data)
            
            _, test_pred = torch.max(test_spk.sum(dim=0), dim=1)
            test_correct += (test_pred == test_labels).sum().item()
            test_total += test_data.size(1)
            
    print(acc = test_correct / test_total)
        


dataset.close()