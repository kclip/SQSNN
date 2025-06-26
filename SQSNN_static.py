import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from SQS import QuantumSpikingLayer

surrogate_type = "sigmoid"
data_type = 'mnist'

class MNISTTemporalDataset(Dataset):
    def __init__(self, root, train=True, T=10, use_rate_encoding=True, max_spike_prob=0.5, data_type = 'mnist'):
        self.T = T
        self.use_rate_encoding = use_rate_encoding
        self.max_spike_prob = max_spike_prob

        if use_rate_encoding:
            self.transform = transforms.Compose([transforms.Resize((28, 28)), transforms.Grayscale(), transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.Resize((28, 28)), transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0,), (1,))])

        if data_type == 'mnist':
            self.data = datasets.MNIST(root=root, train=train, transform=self.transform, download=True)
        elif data_type == 'fmnist':
            self.data = datasets.FashionMNIST(root=root, train=train, transform=self.transform, download=True)
        elif data_type == 'kmnist':
            self.data = datasets.KMNIST(root=root, train=train, transform=self.transform, download=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_tensor, label = self.data[idx]
        img_tensor = img_tensor.view(-1).to(torch.float64)

        if self.use_rate_encoding:
            img_prob = torch.clamp(img_tensor, 0.0, 1.0)
            img_prob = torch.minimum(img_prob, torch.tensor(self.max_spike_prob, dtype=torch.float64))
            spikes = torch.bernoulli(img_prob.unsqueeze(0).expand(self.T, -1))
            return spikes, label
        else:
            repeated = img_tensor.unsqueeze(0).expand(self.T, -1)
            return repeated, label

class SQSNN(nn.Module):
    def __init__(self, input_size, hidden_size=10, output_size=10, T=10):
        super().__init__()
        self.T = T
        self.linear1 = nn.Linear(input_size, hidden_size, bias=False).double()
        self.q_hidden = QuantumSpikingLayer(hidden_size, surrogate_type=surrogate_type)
        self.linear2 = nn.Linear(hidden_size, output_size, bias=False).double()
        self.q_output = QuantumSpikingLayer(output_size, surrogate_type=surrogate_type)

    def forward(self, x):
        self.q_hidden.reset_memory()
        self.q_output.reset_memory()

        spk1_rec = []
        spk2_rec = []

        for t in range(self.T):
            x_t = x[:, t, :]
            h1 = self.linear1(x_t)
            h1_spk, _ = self.q_hidden(h1)
            h2 = self.linear2(h1_spk)
            out_spk, _ = self.q_output(h2)
            spk2_rec.append(out_spk)
            spk1_rec.append(h1_spk)
        return torch.stack(spk2_rec, dim=0), torch.stack(spk1_rec, dim=0)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

use_rate_encoding = True
T = 5
batch_size = 256
num_epochs = 10
num_runs = 10

train_dataset = MNISTTemporalDataset(root="./data_mnist", train=True, T=T, use_rate_encoding=use_rate_encoding, max_spike_prob=0.5, data_type=data_type)
test_dataset = MNISTTemporalDataset(root="./data_mnist", train=False, T=T, use_rate_encoding=use_rate_encoding, max_spike_prob=0.5, data_type=data_type)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


input_size = 784
hidden_size = 1000
output_size = 10

model = SQSNN(input_size, hidden_size, output_size, T=T).to(device).double()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# Store results for this run
run_results = []

for epoch in range(num_epochs):
    model.train()

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device).to(torch.float64)
        y_batch = y_batch.to(device).to(torch.long)

        optimizer.zero_grad()
        spk_rec, _ = model(X_batch) 
        loss = criterion(spk_rec.sum(dim=0)/T, y_batch)
        loss.backward()
        optimizer.step()

    # Evaluate on test set
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for X_batch_test, y_batch_test in test_loader:
            X_batch_test = X_batch_test.to(device).to(torch.float64)
            y_batch_test = y_batch_test.to(device).to(torch.long)

            test_spk, hidden_spk = model(X_batch_test)
            _, test_pred = torch.max(test_spk.sum(dim=0), dim=1)
            test_correct += (test_pred == y_batch_test).sum().item()
            test_total += X_batch_test.size(0)

    print(accuracy = 100 * test_correct / test_total)

    model.train()


