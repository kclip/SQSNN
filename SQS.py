import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


H = (1.0/np.sqrt(2)) * torch.tensor([[1, 1],
                                     [1, -1]], dtype=torch.complex128)

class StraightThroughBernoulli(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p):
        out = torch.bernoulli(p)  # sample 0/1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class SigmoidSurrogateBernoulli(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p):
        out = torch.bernoulli(p)
        ctx.save_for_backward(p)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        p, = ctx.saved_tensors
        return grad_output * p * (1 - p)


def straight_through_sample(p):
    return StraightThroughBernoulli.apply(p)

def initialize_density_matrix():
    rho = torch.zeros((4, 4), dtype=torch.complex128)
    rho[0, 0] = 1.0  # |00⟩⟨00|
    return rho

def rotation_y(theta):
    if theta.dim() == 0:
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        ry = torch.zeros((2, 2), dtype=torch.complex128, device=theta.device)
        ry[0, 0] = c
        ry[0, 1] = -s
        ry[1, 0] = s
        ry[1, 1] = c
        return ry
    else:
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        ry = torch.zeros((theta.size(0), 2, 2), dtype=torch.complex128, device=theta.device)
        ry[:, 0, 0] = c
        ry[:, 0, 1] = -s
        ry[:, 1, 0] = s
        ry[:, 1, 1] = c
        return ry

def rotation_x(theta):
    if theta.dim() == 0:
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        rx = torch.zeros((2, 2), dtype=torch.complex128, device=theta.device)
        rx[0, 0] = c
        rx[0, 1] = -1j * s
        rx[1, 0] = -1j * s
        rx[1, 1] = c
        return rx
    else:
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        rx = torch.zeros((theta.size(0), 2, 2), dtype=torch.complex128, device=theta.device)
        rx[:, 0, 0] = c
        rx[:, 0, 1] = -1j * s
        rx[:, 1, 0] = -1j * s
        rx[:, 1, 1] = c
        return rx

def crx_gate(theta):
    if theta.dim() == 0:
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        rx = torch.zeros((2, 2), dtype=torch.complex128, device=theta.device)
        rx[0, 0] = c
        rx[0, 1] = -1j * s
        rx[1, 0] = -1j * s
        rx[1, 1] = c
        crx_ = torch.eye(4, dtype=torch.complex128, device=theta.device)
        crx_[2:4, 2:4] = rx
        return crx_
    else:
        bsz = theta.size(0)
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        rx = torch.zeros((bsz, 2, 2), dtype=torch.complex128, device=theta.device)
        rx[:, 0, 0] = c
        rx[:, 0, 1] = -1j * s
        rx[:, 1, 0] = -1j * s
        rx[:, 1, 1] = c
        crx_ = torch.eye(4, dtype=torch.complex128, device=theta.device).unsqueeze(0).expand(bsz,4,4).clone()
        crx_[:, 2:4, 2:4] = rx
        return crx_

def apply_gate(rho, gate, wires):
    if rho.dim() == 2:
        device = rho.device
        gate = gate.to(device)
        size = rho.shape[0]
        identity = torch.eye(size // 2, dtype=rho.dtype, device=device)
        if wires == [1]:
            full_gate = torch.kron(identity, gate)
        elif wires == [0]:
            full_gate = torch.kron(gate, identity)
        else:
            full_gate = gate
        return full_gate @ rho @ full_gate.conj().T
    else:
        B = rho.shape[0]
        device = rho.device
        if gate.dim() == 2:
            gate = gate.unsqueeze(0).expand(B, -1, -1)
        if wires == [0]:
            full_gate = torch.zeros((B,4,4), dtype=rho.dtype, device=device)
            full_gate[:, 0:2, 0:2] = gate
            full_gate[:, 2:4, 2:4] = gate
        elif wires == [1]:
            full_gate = torch.zeros((B,4,4), dtype=rho.dtype, device=device)
            full_gate[:, 0, 0] = gate[:, 0, 0]
            full_gate[:, 0, 1] = gate[:, 0, 1]
            full_gate[:, 1, 0] = gate[:, 1, 0]
            full_gate[:, 1, 1] = gate[:, 1, 1]
            full_gate[:, 2, 2] = gate[:, 0, 0]
            full_gate[:, 2, 3] = gate[:, 0, 1]
            full_gate[:, 3, 2] = gate[:, 1, 0]
            full_gate[:, 3, 3] = gate[:, 1, 1]
        else:
            full_gate = gate
        full_gate_dag = full_gate.conj().transpose(-2, -1)
        tmp = torch.bmm(full_gate, rho)
        return torch.bmm(tmp, full_gate_dag)

def partial_trace(rho):
    if rho.dim() == 2:
        device = rho.device
        out = torch.zeros((2,2), dtype=rho.dtype, device=device)
        out[0,0] = rho[0,0] + rho[2,2]
        out[0,1] = rho[0,1] + rho[2,3]
        out[1,0] = rho[1,0] + rho[3,2]
        out[1,1] = rho[1,1] + rho[3,3]
        return out
    else:
        B = rho.shape[0]
        device = rho.device
        out = torch.zeros((B,2,2), dtype=rho.dtype, device=device)
        out[:,0,0] = rho[:,0,0].real + rho[:,2,2].real
        out[:,0,1] = rho[:,0,1] + rho[:,2,3]
        out[:,1,0] = rho[:,1,0] + rho[:,3,2]
        out[:,1,1] = rho[:,1,1].real + rho[:,3,3].real
        return out

class SQS(nn.Module):
    def __init__(self, encoding_type="custom", surrogate_type="straight_through"):
        super().__init__()
        self.full_state = None
        self.encoding_type = encoding_type
        self.surrogate_type = surrogate_type

        if surrogate_type == "straight_through":
            self.surrogate_fn = StraightThroughBernoulli
        elif surrogate_type == "sigmoid":
            self.surrogate_fn = SigmoidSurrogateBernoulli
        else:
            raise ValueError(f"Unsupported surrogate_type: {surrogate_type}")

    def reset_memory(self):
        self.full_state = None

    def forward(self, x_t, theta_mem, theta_in, theta_ent):
        device = x_t.device
        B = x_t.shape[0]

        if self.full_state is None:
            init_dm = initialize_density_matrix().to(device)
            self.full_state = init_dm.unsqueeze(0).expand(B, -1, -1).clone()

        rho = self.full_state

        if self.encoding_type == "RY":
            angles = x_t[:,0]
            RY_in = rotation_y(angles)
            rho = apply_gate(rho, H, [1])
            rho = apply_gate(rho, RY_in, [1])
        elif self.encoding_type == "custom":
            pi_const = torch.tensor(np.pi, dtype=torch.float64, device=device)
            angles = torch.where(x_t[:, 0] > 0, x_t[:, 0] * pi_const,
                                 torch.tensor(0.0, device=device))
            RX_en = rotation_x(angles)
            # rho = apply_gate(rho, RX_en, [0])
            rho = apply_gate(rho, RX_en, [1])
        else:
            raise ValueError("Invalid encoding type. Choose 'RY' or 'custom'.")

        crx_ = crx_gate(theta_ent).to(device)
        rho = apply_gate(rho, crx_, [0,1])

        RX_mem_ = rotation_x(theta_mem).to(device)
        RX_in_  = rotation_x(theta_in).to(device)
        rho = apply_gate(rho, RX_mem_, [0])
        rho = apply_gate(rho, RX_in_,  [1])

        rho_in = partial_trace(rho)
        prob_0 = rho_in[:,0,0].real
        p_spike = 1.0 - prob_0
        p_spike = torch.clamp(p_spike, 0.0, 1.0)
        meas = self.surrogate_fn.apply(p_spike)

        P0 = torch.diag(torch.tensor([1,0,1,0], dtype=torch.complex128, device=device))
        P1 = torch.diag(torch.tensor([0,1,0,1], dtype=torch.complex128, device=device))
        two_proj = torch.stack([P0, P1], dim=0)
        proj = two_proj[meas.long()]

        actual_prob = torch.where(meas == 0, prob_0, 1.0 - prob_0).unsqueeze(-1).unsqueeze(-1)
        tmp = torch.bmm(proj, rho)
        rho_post = torch.bmm(tmp, proj) / actual_prob

        rho_mem = partial_trace(rho_post)

        ket0 = torch.tensor([[1,0],[0,0]], dtype=torch.complex128, device=device).unsqueeze(0).expand(B,2,2)

        a = rho_mem[:, 0, 0].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        b = rho_mem[:, 0, 1].unsqueeze(-1).unsqueeze(-1)
        c = rho_mem[:, 1, 0].unsqueeze(-1).unsqueeze(-1)
        d = rho_mem[:, 1, 1].unsqueeze(-1).unsqueeze(-1)

        block00 = a * ket0
        block01 = b * ket0
        block10 = c * ket0
        block11 = d * ket0

        top = torch.cat([block00, block01], dim=2)   # (B,2,4)
        bot = torch.cat([block10, block11], dim=2)   # (B,2,4)
        new_rho = torch.cat([top, bot], dim=1)       # (B,4,4)

        self.full_state = new_rho

        return meas.unsqueeze(-1).to(device), p_spike.unsqueeze(-1).to(device)



class QuantumSpikingLayer(nn.Module):
    def __init__(self, num_neurons, encoding_type="custom", surrogate_type="sigmoid"):
        super().__init__()
        self.num_neurons = num_neurons
        self.qneuron = SQS(encoding_type=encoding_type, surrogate_type=surrogate_type)

        self.theta_mem = nn.Parameter(torch.randn(num_neurons, dtype=torch.float64) * 0.1)
        self.theta_in  = nn.Parameter(torch.randn(num_neurons, dtype=torch.float64) * 0.1)
        self.theta_ent = nn.Parameter(torch.randn(num_neurons, dtype=torch.float64) * 0.1)

    def reset_memory(self):
        self.qneuron.reset_memory()

    def forward(self, x):
        theta_mem = self.theta_mem.unsqueeze(0).expand(x.size(0), -1).reshape(-1)
        theta_in  = self.theta_in.unsqueeze(0).expand(x.size(0), -1).reshape(-1)
        theta_ent = self.theta_ent.unsqueeze(0).expand(x.size(0), -1).reshape(-1)

        x = x.reshape(x.shape[0], self.num_neurons)
        x_flat = x.reshape(-1, 1)
        spike, p_spike = self.qneuron(x_flat, theta_mem, theta_in, theta_ent)
        return spike.reshape(x.shape), p_spike.reshape(x.shape)
