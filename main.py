from landscapes import *
from metrics import *
from qnns.qnn import *
from expressibility import *
# from entanglement import *

# from circuit import CircuitDescriptor

from fastapi import *

from subprocess import Popen, PIPE


app = FastAPI()


@app.get("/metrics")
def calculate_metrics(num_qubits: int, num_layers: int):
    check_inputs(num_qubits, num_layers)
    landscape = get_loss_landscape(num_qubits, num_layers)
    return {"total_variation": calc_total_variation(landscape),
            "fourier_density": calc_fourier_density(landscape),
            "inverse_standard_gradient_deviation": calc_IGSD(landscape),
            "scalar_curvature": calc_scalar_curvature(landscape)}


@app.get("/metrics/total_variation", response_model=dict[str, float])
def calculate_total_variation(num_qubits: int, num_layers: int) -> dict[str, float]:
    check_inputs(num_qubits, num_layers)
    landscape = get_loss_landscape(num_qubits, num_layers)
    return {"total_variation": calc_total_variation(landscape)}


@app.get("/metrics/fourier_density", response_model=dict[str, float])
def calculate_fourier_density(num_qubits: int, num_layers: int) -> dict[str, float]:
    check_inputs(num_qubits, num_layers)
    landscape = get_loss_landscape(num_qubits, num_layers)
    return {"fourier_density": calc_fourier_density(landscape)}


@app.get("/metrics/inverse_standard_gradient_deviation", response_model=dict[str, list])
def calculate_inverse_standard_gradient_deviation(num_qubits: int, num_layers: int) -> dict[str, list]:
    check_inputs(num_qubits, num_layers)
    landscape = get_loss_landscape(num_qubits, num_layers)
    return {"inverse_standard_gradient_deviation": calc_IGSD(landscape).tolist()}


@app.get("/metrics/scalar_curvature", response_model=dict[str, list])
def calculate_scalar_curvature(num_qubits: int, num_layers: int) -> dict[str, list]:
    check_inputs(num_qubits, num_layers)
    landscape = get_loss_landscape(num_qubits, num_layers)
    return {"scalar_curvature": calc_scalar_curvature(landscape).tolist()}


@app.get("/ansatz_characteristics")
def calculate_ansatz_characteristics(num_qubits: int, num_layers: int):
    return {"ansatz_characteristics": 3}


@app.get("/ansatz_characteristics/entanglement_capability")
def calculate_entanglement_capability(qasm: str, measure: str, shots: int):
    cricuit = CircuitDescriptor.from_qasm(qasm, [], None, "qiskit")

    entagle_calc = EntanglementCapability(cricuit)
    return {"entanglement_capability": entagle_calc.entanglement_capability(measure, shots)}


@app.get("/ansatz_characteristics/expressibility")
def calculate_expressibility(num_tries: int, num_bins: int, num_qubits: int):
    return {"expressibility": expressibility(num_tries, num_bins, num_qubits)}


@app.get("/ZX-calculus")
def calculate_zx_calculus(num_qubits: int, num_layers: int):
    return {"ZX-calculus": 4}


binary_path = 'zx-calculus/target/release/bpdetect'

def zx_calculus(ansatz: str, qubits: int, layers: int, hamiltonian: str, parameter: int):
    p = Popen([binary_path, ansatz, str(qubits), str(layers), hamiltonian, str(parameter)], stdout=PIPE, stderr=PIPE)

    variance, _ = p.communicate()

    if p.returncode != 0:
        print(_.decode('ASCII').strip())
    else:
        variance = variance.decode('ASCII').rstrip()
        s = f"{ansatz}-{qubits}-{layers}-{hamiltonian}-{parameter}: {variance}"
        print(s)


def check_inputs(num_qubits, num_layers):
    if num_qubits < 1:
        raise HTTPException(status_code=404, detail="invalid numer of qubits")
    if num_layers < 1:
        raise HTTPException(status_code=404, detail="invalid numer of layers")


def get_loss_landscape(num_qubits, num_layers, schmidt_rank=2, num_data_points=3, grid_size=3):
    qnn = get_qnn("CudaU2", list(range(num_qubits)), num_layers, device="cpu")
    unitary = torch.tensor(data=np.array(random_unitary_matrix(num_qubits)), dtype=torch.complex128, device="cpu")
    inputs = generate_data_points(type_of_data=1, schmidt_rank=schmidt_rank, num_data_points=num_data_points, U=unitary, num_qubits=num_qubits)
    dimensions = num_qubits * num_layers * 3
    loss_landscape = generate_loss_landscape(grid_size=grid_size, dimensions=dimensions, inputs=inputs, U=unitary, qnn=qnn)
    return loss_landscape


def test(qnn_type, num_qubits, num_layers, unitary):
    qnn = get_qnn("CudaU2", list(range(num_qubits)), num_layers, device="cpu")
    random_unitary = torch.tensor(np.array(random_unitary_matrix(2)), dtype=torch.complex128, device="cpu")
    inputs = generate_random_datapoints(3, num_qubits, random_unitary)

    dimensions = len(qnn.params)
    landscape = generate_loss_landscape(10, dimensions, inputs, unitary, qnn)
    calculate_metrics(landscape)


def test_qnn_generation():
    for num_qubits in range(1, 10):
        for num_layers in range(1,10):
            qnn = get_qnn("CudaU2", list(range(num_qubits)), num_layers, device="cpu")
            print(qnn.params)


def test_input_generation():
    # schmidt_rank <= 2^(num_qubits)
    for num_qubits in range(1, 6):
        for s_rank in range(1, 2**num_qubits+1):
            unitary = torch.tensor(data=np.array(random_unitary_matrix(num_qubits)), dtype=torch.complex128, device="cpu")
            inputs = generate_data_points(type_of_data=1, schmidt_rank=4, num_data_points=100, U = unitary, num_qubits=6)
            print(inputs.shape)


def test_loss_landscape_calculation():
    for num_qubits in range (1, 4):
        for num_layers in range(1, 3):
            print(num_qubits, num_layers)
            qnn = get_qnn("CudaU2", list(range(num_qubits)), num_layers, device="cpu")
            unitary = torch.tensor(data=np.array(random_unitary_matrix(num_qubits)), dtype=torch.complex128, device="cpu")
            inputs = generate_data_points(type_of_data=1, schmidt_rank=2, num_data_points=3, U=unitary, num_qubits=num_qubits)
            dimensions = num_qubits * num_layers * 3
            loss_landscape = generate_loss_landscape(grid_size=3, dimensions=dimensions, inputs=inputs, U=unitary, qnn=qnn)


def test_metrics():
    num_qubits = 3
    num_layers = 1
    print(calculate_total_variation(num_qubits, num_layers))
    print(calculate_fourier_density(num_qubits, num_layers))
    print(calculate_inverse_standard_gradient_deviation(num_qubits, num_layers))
    print(calculate_scalar_curvature(num_qubits, num_layers))
    print(calculate_metrics(num_qubits, num_layers))


if __name__ == "__main__":
    # test_qnn_generation()
    # test_input_generation()
    # test_loss_landscape_calculation()
    # test_metrics()
    zx_calculus(ansatz='sim1', qubits=2, layers=1, hamiltonian='ZZ', parameter=0)
