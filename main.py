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

from flask import Flask
from flask import jsonify, abort, request
import logging
from flask_smorest import Api
from flask_smorest import Blueprint
import marshmallow as ma
from marshmallow import fields, ValidationError
import threading

app = Flask(__name__)

app.config.update(
    API_TITLE = "QML Toolbox",
    API_VERSION = "0.1",
    OPENAPI_VERSION = "3.0.2",
    OPENAPI_URL_PREFIX = "/api",
    OPENAPI_SWAGGER_UI_PATH = "/swagger-ui",
    OPENAPI_SWAGGER_UI_VERSION = "3.24.2",
    OPENAPI_SWAGGER_UI_URL = "https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/3.24.2/",

    API_SPEC_OPTIONS = {
        "info": {
            "description": "Toolbox for QML developement",
        },
        #"license": {"name": "Apache v2 License"},
    }
)

api = Api(app)


@app.route("/")
def heartbeat():
    return '<h1>QML-Toolbox</h1> <h3>View the API Docs <a href="/api/swagger-ui">here</a></h3>'


class MetricsRequestSchema(ma.Schema):
    num_qubits = ma.fields.Int()
    num_layers = ma.fields.Int()

class MetricsResponseSchema(ma.Schema):
    total_variation = ma.fields.Float()
    fourier_density = ma.fields.Float()
    inverse_standard_gradient_deviation = ma.fields.List(ma.fields.Float())
    scalar_curvature = ma.fields.String() #Datentyp anpassen




blp_metrics = Blueprint(
    "metrics",
    __name__,
    "calculates the metrics"
)
@blp_metrics.route("/metrics", methods=["POST"])
@blp_metrics.arguments(
    MetricsRequestSchema,
    example=dict(
        num_qubits=1,
        num_layers=1
    ),
)
@blp_metrics.response(200, MetricsResponseSchema)
def calculate_metrics(inputs: dict):
    print(inputs)
    unitary = torch.tensor(data=np.array(random_unitary_matrix(inputs["num_qubits"])), dtype=torch.complex128, device="cpu")
    landscape = get_loss_landscape(inputs["num_qubits"], inputs["num_layers"], unitary)
    outputs ={"total_variation": calc_total_variation(landscape),
            "fourier_density": calc_fourier_density(landscape),
            "inverse_standard_gradient_deviation": calc_IGSD(landscape),
            "scalar_curvature": str(calc_scalar_curvature(landscape))
    }  

    return outputs


@blp_metrics.route("/calculate_total_variation", methods=["POST"])
@blp_metrics.arguments(
    MetricsRequestSchema,
    example=dict(
        num_qubits=1,
        num_layers=1
    ),
)
@blp_metrics.response(200, MetricsResponseSchema)
def calculate_total_variation(inputs: dict):
    unitary = torch.tensor(data=np.array(random_unitary_matrix(inputs["num_qubits"])), dtype=torch.complex128, device="cpu")
    landscape = get_loss_landscape(inputs["num_qubits"], inputs["num_layers"], unitary)
    return {"total_variation": calc_total_variation(landscape)}


@blp_metrics.route("/fourier_density", methods=["POST"])
@blp_metrics.arguments(
    MetricsRequestSchema,
    example=dict(
        num_qubits=1,
        num_layers=1
    ),
)
@blp_metrics.response(200, MetricsResponseSchema)
def calculate_fourier_density(inputs: dict):
    unitary = torch.tensor(data=np.array(random_unitary_matrix(inputs["num_qubits"])), dtype=torch.complex128, device="cpu")
    landscape = get_loss_landscape(inputs["num_qubits"], inputs["num_layers"], unitary)
    return {"fourier_density": calc_fourier_density(landscape)}


@blp_metrics.route("/inverse_standard_gradient_deviation", methods=["POST"])
@blp_metrics.arguments(
    MetricsRequestSchema,
    example=dict(
        num_qubits=1,
        num_layers=1
    ),
)
@blp_metrics.response(200, MetricsResponseSchema)
def calculate_inverse_standard_gradient_deviation(inputs: dict):
    unitary = torch.tensor(data=np.array(random_unitary_matrix(inputs["num_qubits"])), dtype=torch.complex128, device="cpu")
    landscape = get_loss_landscape(inputs["num_qubits"], inputs["num_layers"], unitary)
    return {"inverse_standard_gradient_deviation": calc_IGSD(landscape).tolist()}



@blp_metrics.route("/scalar_curvature", methods=["POST"])
@blp_metrics.arguments(
    MetricsRequestSchema,
    example=dict(
        num_qubits=1,
        num_layers=1
    ),
)
@blp_metrics.response(200, MetricsResponseSchema)
def calculate_scalar_curvature(inputs: dict):
    unitary = torch.tensor(data=np.array(random_unitary_matrix(inputs["num_qubits"])), dtype=torch.complex128, device="cpu")
    landscape = get_loss_landscape(inputs["num_qubits"], inputs["num_layers"], unitary)
    return {"scalar_curvature": str(calc_scalar_curvature(landscape).tolist())}

'''
@app.get("/ansatz_characteristics")
def calculate_ansatz_characteristics(num_qubits: int, num_layers: int):
    return {"ansatz_characteristics": 3}
'''

blp_characteristics = Blueprint(
    "ansatz_characteristics",
    __name__,
    "calculates the ansatz_characteristics"
)

class EntanglementCapabilityRequestSchema(ma.Schema):
    qasm = ma.fields.String()
    measure = ma.fields.String()
    shots = ma.fields.Int()

class EntanglementCapabilityResponseSchema(ma.Schema):
    entanglement_capability = ma.fields.Float()

@blp_characteristics.route("/entanglement_capability", methods=["POST"])
@blp_characteristics.arguments(
    EntanglementCapabilityRequestSchema,
    example=dict(
        qasm='''OPENQASM 3;include "stdgates.inc";input float[64] a;input float[64] a1;input float[64] b;input float[64] b1;input float[64] c;qubit[4] _all_qubits;let q = _all_qubits[0:3];u3(a, b, c) q[0];cx q[0], q[1];cx q[1], q[2];cx q[2], q[3];u3(c, a1, b1) q[3];cx q[3], q[0];cx q[1], q[3];u3(b1, a1, c) q[2];''',
        measure='''meyer-wallach''',
        shots = 1024


    ),
)
@blp_characteristics.response(200, EntanglementCapabilityResponseSchema)
def calculate_entanglement_capability(inputs: dict):

    cricuit = CircuitDescriptor.from_qasm(inputs["qasm"],[],None,"qiskit")
    

    entagle_calc = EntanglementCapability(cricuit)
    return {"entanglement_capability": entagle_calc.entanglement_capability(inputs["measure"], inputs["shots"])}



class ExpressibilityRequestSchema(ma.Schema):
    num_tries = ma.fields.Int()
    num_bins = ma.fields.Int()
    num_qubits = ma.fields.Int()



class ExpressibilityResponseSchema(ma.Schema):
    expressibility = ma.fields.Float()

@blp_characteristics.route("/expressibility", methods=["POST"])
@blp_characteristics.arguments(
    ExpressibilityRequestSchema,
    example=dict(
        num_tries = 1000,
        num_bins = 50,
        num_qubits = 2


    ),
)
@blp_characteristics.response(200, ExpressibilityResponseSchema)
def calculate_expressibility(inputs:dict):
    return {"expressibility": expressibility(inputs["num_tries"] , inputs["num_bins"], inputs["num_qubits"])}

'''
@app.get("/ZX-calculus")
def calculate_zx_calculus(num_qubits: int, num_layers: int):
    return {"ZX-calculus": 4}
'''

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


api.register_blueprint(blp_metrics)
api.register_blueprint(blp_characteristics)

matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
unitary = torch.tensor(matrix, dtype=torch.complex128, device="cpu")
# test("Pennylane", 2, 3, unitary)


if __name__ == "__main__":
    #print("\nOne qubit:")
    #test_one_qubit()

    #print("\nTwo qubits:")
    #test_two_qubits()

    print("Starting Server")
    app.run(host="0.0.0.0", port=8000)
