from landscapes import *
from metrics import *
from qnns.cuda_qnn import *
from qnns.qnn import *
from expressibility import *
from entanglement import *

from circuit import CircuitDescriptor
import cirq
import sympy
from qiskit import QuantumCircuit

from fastapi import *

app = FastAPI()


@app.get("/metrics")
def calculate_metrics(num_qubits: int, num_layers: int):
    unitary = torch.tensor(data=np.array([[1, 1], [1, -1]]) / np.sqrt(2), dtype=torch.complex128, device="cpu")
    landscape = get_loss_landscape(1, 1, unitary)
    return {"total_variation": calc_total_variation(landscape),
            "fourier_density": calc_fourier_density(landscape),
            "inverse_standard_gradient_deviation": calc_IGSD(landscape),
            "scalar_curvature": calc_scalar_curvature(landscape)}


@app.get("/metrics/total_variation", response_model=dict[str, float])
def calculate_total_variation(num_qubits: int, num_layers: int) -> dict[str, float]:
    if num_qubits < 1 or num_layers < 1:
        raise HTTPException(status_code=404, detail="invalid numer of qubits or layers")
    unitary = torch.tensor(data=np.array([[1, 1], [1, -1]]) / np.sqrt(2), dtype=torch.complex128, device="cpu")
    landscape = get_loss_landscape(1, 1, unitary)
    return {"total_variation": calc_total_variation(landscape)}


@app.get("/metrics/fourier_density", response_model=dict[str, float])
def calculate_fourier_density(num_qubits: int, num_layers: int) -> dict[str, float]:
    unitary = torch.tensor(data=np.array([[1, 1], [1, -1]]) / np.sqrt(2), dtype=torch.complex128, device="cpu")
    landscape = get_loss_landscape(1, 1, unitary)
    return {"fourier_density": calc_fourier_density(landscape)}


@app.get("/metrics/inverse_standard_gradient_deviation", response_model=dict[str, list])
def calculate_inverse_standard_gradient_deviation(num_qubits: int, num_layers: int) -> dict[str, list]:
    unitary = torch.tensor(data=np.array([[1, 1], [1, -1]]) / np.sqrt(2), dtype=torch.complex128, device="cpu")
    landscape = get_loss_landscape(1, 1, unitary)
    return {"inverse_standard_gradient_deviation": calc_IGSD(landscape).tolist()}


@app.get("/metrics/scalar_curvature", response_model=dict[str, list])
def calculate_scalar_curvature(num_qubits: int, num_layers: int) -> dict[str, list]:
    unitary = torch.tensor(data=np.array([[1, 1], [1, -1]]) / np.sqrt(2), dtype=torch.complex128, device="cpu")
    landscape = get_loss_landscape(1, 1, unitary)
    return {"scalar_curvature": calc_scalar_curvature(landscape).tolist()}


@app.get("/ansatz_characteristics")
def calculate_ansatz_characteristics(num_qubits: int, num_layers: int):
    return {"ansatz_characteristics": 3}


@app.get("/ansatz_characteristics/entanglement_capability")
def calculate_entanglement_capability(qasm:str, measure: str, shots: int):

    cricuit = CircuitDescriptor.from_qasm(qasm,[],None,"qiskit")
    

    entagle_calc = EntanglementCapability(cricuit)
    return {"entanglement_capability": entagle_calc.entanglement_capability(measure, shots)}


@app.get("/ansatz_characteristics/expressibility")
def calculate_expressibility(num_tries: int, num_bins: int, num_qubits: int):
    return {"expressibility": expressibility(num_tries, num_bins, num_qubits)}


@app.get("/ZX-calculus")
def calculate_zx_calculus(num_qubits: int, num_layers: int):
    return {"ZX-calculus": 4}


def get_loss_landscape(num_qubits, num_layers, unitary):
    qnn = get_qnn(qnn_name="CudaU2", x_wires=list(range(num_qubits)), num_layers=num_layers, device="cpu")
    inputs = generate_random_datapoints(numb_points=2, s_rank=1, U=unitary)
    loss_landscape = generate_2d_loss_landscape(grid_size=50, inputs=inputs, U=unitary, qnn=qnn)
    return loss_landscape


def calculate_metrics(landscape):
    scalar_curvature = calc_scalar_curvature(landscape)
    print("Scalar Curvature: " + str(scalar_curvature))

    total_variation = calc_total_variation(landscape)
    print("Total Variation: " + str(total_variation))

    fourier_density = calc_fourier_density(landscape)
    print("Fourier Density: " + str(fourier_density))

    inverse_standard_gradient_deviation = calc_IGSD(landscape)
    print("Inverse Standard Gradient Deviation: " + str(inverse_standard_gradient_deviation))


def test_one_qubit():
    '''
    Test landscape generation and metric calculation for QNNs with one qubit
    '''
    qnn = get_qnn(qnn_name="CudaU2", x_wires=[0], num_layers=1, device="cpu")
    print(f"Number of parameters: {str(len(qnn.params))}")

    U = torch.tensor(data=np.array([[1, 1], [1, -1]]) / np.sqrt(2), dtype=torch.complex128, device="cpu")

    inputs = generate_random_datapoints(numb_points=2, s_rank=1, U=U)

    loss_landscape = generate_2d_loss_landscape(grid_size=50, inputs=inputs, U=U, qnn=qnn)

    calculate_metrics(loss_landscape)


def test_two_qubits():
    '''
    Test landscape generation and metric calculation for QNNs with two qubits
    '''
    num_qubits = 2
    qnn = CudaPennylane(num_wires=num_qubits, num_layers=1, device='cpu')
    print(f"Number of parameters: {str(len(qnn.params))}")

    U = torch.tensor(data=np.array(random_unitary_matrix(num_qubits)), dtype=torch.complex128, device="cpu")

    inputs = generate_random_datapoints(numb_points=3, s_rank=num_qubits, U=U)

    loss_landscape = generate_loss_landscape(grid_size=3, dimensions=6, inputs=inputs, U=U, qnn=qnn)

    calculate_metrics(loss_landscape)


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
    print("\nOne qubit:")
    test_one_qubit()

    print("\nTwo qubits:")
    test_two_qubits()
