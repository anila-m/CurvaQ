from victor_thesis_landscapes import *
from victor_thesis_metrics import *


def calculate_metrics(landscape):
    scalar_curvature = calc_scalar_curvature(landscape)
    print("Scalar Curvature: " + str(scalar_curvature))

    total_variation = calc_total_variation(landscape)
    print("Total Variation: " + str(total_variation))

    fourier_density = calc_fourier_density(landscape)
    print("Fourier Density: " + str(fourier_density))

    inverse_standard_gradient_deviation = calc_IGSD(landscape)
    print("Inverse Standard Gradient Deviation: " + str(inverse_standard_gradient_deviation))


# test landscape generation and metric calculation for QNNs with one qubit
def test_one_qubit():
    qnn = get_qnn("CudaU2", list(range(1)), 1, device="cpu")
    print("Number of Parameter: " + str(len(qnn.params)))

    unitary = torch.tensor(np.array([[1, 1], [1, -1]]) / np.sqrt(2), dtype=torch.complex128, device="cpu")
    inputs = generate_random_datapoints(3, 1, unitary)
    landscape = generate_2d_loss_landscape(10, inputs, unitary, qnn)
    calculate_metrics(landscape)


# test landscape generation and metric calculation for QNNs with two qubits
# NOT YET WORKING
def test_two_qubits():
    qnn = get_qnn("CudaU2", list(range(2)), 3, device="cpu")
    print("Number of Parameter: " + str(len(qnn.params)))

    matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    unitary = torch.tensor(matrix, dtype=torch.complex128, device="cpu")
    print(unitary)

    random_unitary = torch.tensor(np.array(random_unitary_matrix(2)), dtype=torch.complex128, device="cpu")
    print(random_unitary)

    inputs = generate_random_datapoints(3, 2, random_unitary)
    print(inputs)

    landscape = generate_2d_loss_landscape(10, inputs, unitary, qnn)
    print(landscape)

    calculate_metrics(landscape)


def test(qnn_type, num_qubits, num_layers, unitary):
    qnn = get_qnn("CudaU2", list(range(num_qubits)), num_layers, device="cpu")
    random_unitary = torch.tensor(np.array(random_unitary_matrix(2)), dtype=torch.complex128, device="cpu")
    inputs = generate_random_datapoints(3, num_qubits, random_unitary)

    dimensions = len(qnn.params)
    landscape = generate_loss_landscape(10, dimensions, inputs, unitary, qnn)
    calculate_metrics(landscape)




matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
unitary = torch.tensor(matrix, dtype=torch.complex128, device="cpu")
# test("Pennylane", 2, 3, unitary)


if __name__ == "__main__":
    test_one_qubit()
    # test_two_qubits()