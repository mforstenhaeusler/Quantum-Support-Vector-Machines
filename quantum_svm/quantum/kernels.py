from qiskit import QuantumCircuit, execute
import numpy as np

class QuantumKernel:
    def __init__(self, feature_map, quantum_backend, sim_params):
        """ Qunatum Kernel 
        Implements Equation ... from slides.
        
        Params:
        -------
        feature_map : qiskit instruction
                      instruction for parameterized quantum circuit
        quantum_backend : qiskit backend simulator 
                          qiskit backend simulator that allows to simulate a quantum circuit
        sim_param : dict
                    simulation parameters required by qiskit backend
        """
        self._feature_map = feature_map
        self.n_qubits = feature_map.n_qubits
        self._quantum_backend = quantum_backend
        self.sim_params = sim_params

        if str(self._quantum_backend)  == 'statevector_simulator':
            self._statevector_sim = True
        else:
            self._statevector_sim = False

    def construct_circuit(self, X1, X2):
        """ Constructs the parameterzied circuits for the quantum kernel calculation
        
        Params:
        -------
        X1 : np.ndarray
             classical data point to be transfered to quantum state 
        X1 : np.ndarray
             classical data point to be transfered to quantum state 

        Return:
        -------
        Parameterized quantum circuits for each entry of the qunatum kernel matrix.
        """
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

    def evaluate(self, x_vec, y_vec=None) -> np.ndarray:
        """ Computes the quantum kernel
        
        Params:
        -------
        x_vec : 1D or 2D np.ndarray 
                classical data to be transfered to quantum state 
        y_vec : 1D or 2D np.ndarray
                classical data to be transfered to quantum state 

        Return:
        -------
        Quantum Kernel Matrix
        """
        # - - - adopted form qiskit source code - - -
        if not isinstance(x_vec, np.ndarray):
            x_vec = np.asarray(x_vec)
        if y_vec is not None and not isinstance(y_vec, np.ndarray):
            y_vec = np.asarray(y_vec)

        if x_vec.ndim > 2:
            raise ValueError("x_vec must be a 1D or 2D array")

        if x_vec.ndim == 1:
            x_vec = np.reshape(x_vec, (-1, 2))

        if y_vec is not None and y_vec.ndim > 2:
            raise ValueError("y_vec must be a 1D or 2D array")

        if y_vec is not None and y_vec.ndim == 1:
            y_vec = np.reshape(y_vec, (-1, 2))

        if y_vec is not None and y_vec.shape[1] != x_vec.shape[1]:
            raise ValueError(
                "x_vec and y_vec have incompatible dimensions.\n"
                f"x_vec has {x_vec.shape[1]} dimensions, but y_vec has {y_vec.shape[1]}."
            )

        if x_vec.shape[1] != self._feature_map.n_qubits:
            try:
                self._feature_map.num_qubits = x_vec.shape[1]
            except AttributeError:
                raise ValueError(
                    "x_vec and class feature map have incompatible dimensions.\n"
                    f"x_vec has {x_vec.shape[1]} dimensions, "
                    f"but feature map has {self._feature_map.n_qubits}."
                ) 

        if y_vec is not None and y_vec.shape[1] != self._feature_map.n_qubits:
            raise ValueError(
                "y_vec and class feature map have incompatible dimensions.\n"
                f"y_vec has {y_vec.shape[1]} dimensions, but feature map "
                f"has {self._feature_map.n_qubits}."
            )
        # - - - adopted form qiskit source code - - -
        measurement_basis = "0" * self._feature_map.n_qubits

        # calculate kernel
        if self._statevector_sim: # statevector simulator
            raise BackendError
        else:
            if y_vec is None:
                N, D = x_vec.shape
                circuits = []
                for i in range(N):
                    for j in range(N):
                        circuits.append(self.construct_circuit(x_vec[i], x_vec[j]))
                
                k_values = []
                # calculate the inner products via the unintary operator
                job = execute(circuits, self._quantum_backend, shots=self.sim_params['shots'], 
                                seed_simulator=self.sim_params['seed'], see_transpiler=self.sim_params['seed'])
                # get the results
                for j in range(len(circuits)):
                    # calculate the kernel values
                    k_values.append(self.__compute_kernel_val(j, job, measurement_basis))
                
                kernel = np.array(k_values).reshape(x_vec.shape[0], x_vec.shape[0])
                return kernel
            else:
                N, M = x_vec.shape[0], y_vec.shape[0]
                circuits = []
                for i in range(N):
                    for j in range(N):
                        circuits.append(self.construct_circuit(x_vec[i], y_vec[j]))
                
                k_values = []
                # calculate the inner products via the unintary operator
                job = execute(circuits, self._quantum_backend, shots=self.sim_params['shots'], 
                                seed_simulator=self.sim_params['seed'], see_transpiler=self.sim_params['seed'])
                # get the results
                for j in range(len(circuits)):
                    # calculate the kernel values
                    k_values.append(self.__compute_kernel_val(j, job, measurement_basis))
                kernel = np.array(k_values).reshape(N, M)
                return kernel

    def __compute_kernel_val(self, idx, job, measurement_basis):
        """
        Computes the kernel values form the results of the inner products.

        Params:
        -------

        Return:
        -------
        Kernel value of entry [i, j] in Quantum Kernel Matrix K_{ij}
        """
        if self._statevector_sim:
           raise BackendError
        else:
            result = job.result().get_counts(idx)

            kernel_value = result.get(measurement_basis, 0) / sum(result.values())
        return kernel_value

class BackendError(Exception):
    """ Choose a qasm simulator, statevector simulation has not been implemented yet"""
