"""Module to draw samples from the circuit.
Used for computing properties of the circuit like Entanglability and Expressibility.
"""

import typing

import cirq
import numpy as np
import qiskit

from qiskit import transpile

import qiskit.pulse.instructions

from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

from qiskit_aer.library import *

from qiskit_aer.noise import NoiseModel as qiskitNoiseModel
from cirq.devices.noise_model import NoiseModel as cirqNoiseModel
from pyquil.noise import NoiseModel as pyquilNoiseModel

from circuit import CircuitDescriptor


class CircuitSimulator:
    """The interface for users to execute their CircuitDescriptor objects"""

    def __init__(
        self,
        circuit: CircuitDescriptor,
        noise_model: typing.Union[
            cirqNoiseModel, qiskitNoiseModel, pyquilNoiseModel, None
        ] = None,
    ) -> None:
        """Initialize the state simulator
        :type circuit: CircuitDescriptor
        :param circuit: the target circuit to simulate
        :type noise_model: Noise model as a dict or in the library format
        :param noise_model: the noise model as dict or empty dict for density matrix simulations,
            None if performing state vector simulations
        """
        self.circuit = circuit
        self.noise_model = noise_model
        self._result = None

    @property
    def result(
        self,
    ) -> typing.Optional[np.ndarray]:
        """Get the results stored from the circuit simulator
        :return: stored result of the circuit simulation if it has been performed, else None.
        :rtype: np.array or None
        """
        return self._result

    def simulate(
        self,
        param_resolver: typing.Dict[qiskit.circuit.Parameter, float],
        shots: int = 1024,
    ) -> np.ndarray:
        """Simulate to get the state vector or the density matrix
        :type param_resolver: Dict to resolve all parameters to a static float value
        :param param_resolver: a dictionary of all the symbols/parameters mapping to their values
        :type shots: int
        :param shots: number of times to run the qiskit density matrix simulator
        :returns: state vector or density matrix resulting from the simulation
        :rtype: np.array
        :raises NotImplementedError: if circuit simulation is not supported for a backend
        """
        if self.circuit.default_backend == "qiskit":
            circuit = self.circuit.qiskit_circuit.assign_parameters(param_resolver)
            if self.noise_model is not None:
                circuit.save_density_matrix(label="final")

                #backend = AerSimulator(method="density_matrix", noise_model = self.noise_model)
                backend = AerSimulator(method="density_matrix", noise_model = None)
                circuit = transpile(circuit, backend)

                result = backend.run(circuit, parameter_binds=[param_resolver]).result()

                result_data = result.data(0)["final"]

            else:
                circuit.save_statevector("final")

                backend = AerSimulator(method="statevector")
                circuit = transpile(circuit, backend)

                result = backend.run(circuit, parameter_binds=[param_resolver]).result()
                
                result_data = result.data(0)["final"]
                

        elif self.circuit.default_backend == "cirq":

            circuit = self.circuit.cirq_circuit
            non_unitary_flag = False
            for op in circuit.all_operations():
                op_name = str(op).split("(")[0]
                if op_name in [
                    "phase_flip",
                    "phase_damp",
                    "amplitude_damp",
                    "depolarize",
                    "asymmetric_depolarize",
                ]:
                    non_unitary_flag = True
                    break

            if self.noise_model is None and not non_unitary_flag:
                simulator = cirq.Simulator()  # type: ignore
                result = simulator.simulate(self.circuit.cirq_circuit, param_resolver)
                result_data = result.final_state_vector
            else:
                simulator = cirq.DensityMatrixSimulator(noise=self.noise_model)  # type: ignore
                result = simulator.simulate(self.circuit.cirq_circuit, param_resolver)
                result_data = result.final_density_matrix

        else:
            raise NotImplementedError(
                "Parametrized circuit simulation is not implemented for this backend."
            )

        self._result = result_data
        return result_data
