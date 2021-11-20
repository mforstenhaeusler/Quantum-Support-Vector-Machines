from numpy import ndarray
from qiskit import QuantumCircuit, execute, Aer
from qiskit.pulse.configuration import Kernel
from qiskit.utils import QuantumInstance
from qiskit.circuit import instruction
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.quantumregister import QuantumRegister
import numpy as np

class QuantumKernel:
    def __init__(self, feature_map, quantum_backend, sim_params):
        self._feature_map = feature_map
        self.n_qubits = feature_map.n_qubits
        self._quantum_backend = quantum_backend
        self.sim_params = sim_params

        if str(self._quantum_backend)  == 'statevector_simulator':
            self._statevector_sim = True
        else:
            self._statevector_sim = False

    def construct_circuit(self, X1, X2):
        circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        if self._statevector_sim: # statevector simulator
            raise BackendError
        else:
            instruction= self._feature_map.map(X1, reverse=False)
            instruction_re = self._feature_map.map(X2, reverse=True)
            circuit.append(instruction, [0,1])
            circuit.append(instruction_re, [0,1])
        circuit.barrier()
        circuit.measure([i for i in range(self.n_qubits)], [i for i in range(self.n_qubits)])

        return circuit

    def evaluate(self, X):
        measurement_basis = "0" * self._feature_map.n_qubits

        # calculate kernel
        if self._statevector_sim: # statevector simulator
            raise BackendError
        else:
            N, D = X.shape
            circuits = []
            for i in range(N):
                for j in range(N):
                    circuits.append(self.construct_circuit(X[i], X[j]))
            
            k_values = []
            # calculate the inner products via the unintary operator
            job = execute(circuits, self._quantum_backend, shots=self.sim_params['shots'], seed_simulator=self.sim_params['seed'], see_transpiler=self.sim_params['seed'])
            # get the results
            for j in range(len(circuits)):
                # calculate the kernel values
                k_values.append(self._compute_kernel_val(j, job, measurement_basis))
            
            kernel = np.array(k_values).reshape(X.shape[0], X.shape[0])
            return kernel
            

    def _compute_kernel_val(self, idx, job, measurement_basis):
        """
        Computes the kernel values form the results of the inner products.
        """
        if self._statevector_sim:
           raise BackendError
        else:
            result = job.result().get_counts(idx)

            kernel_value = result.get(measurement_basis, 0) / sum(result.values())
        return kernel_value

class BackendError(Exception):
    """ Choose a qasm simulator """
