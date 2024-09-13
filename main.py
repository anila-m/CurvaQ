import torch
import numpy as np
from qnns.cuda_qnn import CudaPennylane
from victor_thesis_landscapes import generate_random_datapoints
from victor_thesis_landscapes import generate_2d_loss_landscape
from data import random_unitary_matrix

num_qubits = 2
qnn = CudaPennylane(num_wires=num_qubits, num_layers=3, device="cpu")
print(qnn.params)

U = torch.tensor(np.array([[1, 1], [1, -1]]) / np.sqrt(2), dtype=torch.complex128, device="cpu")
print(U)

random_unitary = torch.tensor(np.array(random_unitary_matrix(num_qubits)), dtype=torch.complex128, device="cpu")
print(random_unitary)

inputs = generate_random_datapoints(3, num_qubits, random_unitary)
print(inputs)

loss_landscape = generate_2d_loss_landscape(50, inputs, U, qnn)
print(loss_landscape)
