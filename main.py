from victor_thesis_landscapes import *
from victor_thesis_metrics import *


def calculate_metrics(landscape):
    scalar_curvature = calc_scalar_curvature(landscape)
    print("Scalar Curvature: " + str(scalar_curvature))

    total_variation = calc_total_variation(landscape)
    print("Total Variation: " + str(total_variation))

    fourier_density = calc_fourier_density(landscape)
    print("Fourier Density: " + str(fourier_density))

    IGSD = calc_IGSD(landscape)
    print("Inverse Standard Gradient Deviation: " + str(IGSD))


# test landscape generation and metric calculation for QNNs with one qubit
def test_one_qubit():
    qnn = get_qnn("CudaU2", [0], 1, device="cpu")
    unitary = torch.tensor(np.array([[1, 1], [1, -1]]) / np.sqrt(2), dtype=torch.complex128, device="cpu")
    inputs = generate_random_datapoints(3, 1, unitary)
    landscape = generate_2d_loss_landscape(10, inputs, unitary, qnn)
    calculate_metrics(landscape)


# test landscape generation and metric calculation for QNNs with two qubits
# NOT YET WORKING
def test_two_qubits():
    num_qubits = 2
    qnn = get_qnn("CudaU2", list(range(num_qubits)), 3, device="cpu")
    print(qnn.params)

    matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    unitary = torch.tensor(matrix, dtype=torch.complex128, device="cpu")
    print(unitary)

    random_unitary = torch.tensor(np.array(random_unitary_matrix(num_qubits)), dtype=torch.complex128, device="cpu")
    print(random_unitary)

    inputs = generate_random_datapoints(3, num_qubits, random_unitary)
    print(inputs)

    landscape = generate_2d_loss_landscape(10, inputs, unitary, qnn)
    print(landscape)

    calculate_metrics(landscape)


test_one_qubit()
test_two_qubits()
