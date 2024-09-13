import torch
import numpy as np
from qnns.cuda_qnn import CudaPennylane
from qnns.qnn import get_qnn
from victor_thesis_landscapes import generate_random_datapoints
from victor_thesis_landscapes import generate_2d_loss_landscape
from data import random_unitary_matrix
from victor_thesis_metrics import *

"""
num_qubits = 2
qnn = CudaPennylane(num_wires=num_qubits, num_layers=3, device="cpu")
print(qnn.params)

#U = torch.tensor(np.array([[1, 1], [1, -1]]) / np.sqrt(2), dtype=torch.complex128, device="cpu")
U = torch.tensor(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), dtype=torch.complex128, device="cpu")
print(U)

random_unitary = torch.tensor(np.array(random_unitary_matrix(num_qubits)), dtype=torch.complex128, device="cpu")
print(random_unitary)

inputs = generate_random_datapoints(3, num_qubits, random_unitary)
print(inputs)

loss_landscape = generate_2d_loss_landscape(4, inputs, U, qnn)
print(loss_landscape)

"""

qnn = get_qnn("CudaU2", [0], 1, device="cpu")
#qnn = CudaPennylane(num_wires=0, num_layers=1, device="cpu")

unitary = torch.tensor(np.array([[1, 1], [1, -1]]) / np.sqrt(2), dtype=torch.complex128, device="cpu")
inputs = generate_random_datapoints(3, 1, unitary)
landscape = generate_2d_loss_landscape(4, inputs, unitary, qnn)

scalar_curvature = calc_scalar_curvature(landscape)
print(scalar_curvature)

total_variation = calc_total_variation(landscape)
print(total_variation)

fourier_density = calc_fourier_density(landscape)
print(fourier_density)

IGSD = calc_IGSD(landscape)
print(IGSD)