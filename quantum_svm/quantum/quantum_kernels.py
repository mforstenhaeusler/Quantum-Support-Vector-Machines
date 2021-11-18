import qiskit
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit import BasicAer, Aer
from qiskit.utils import QuantumInstance, algorithm_globals

def quantum_kernel(params):
    """ Quantum Kernel implementation using Qiskit """
    if params is not None:
        feature_dimension= params['feature_dimension']
        reps=params['reps']
        seed=params['seed']
        shots=params['shots']
        provider_backend=params['provider_backend']

        algorithm_globals.random_seed = seed

        feature_map = ZZFeatureMap(feature_dimension=feature_dimension, reps=reps, entanglement='linear', insert_barriers=True)
        
        if provider_backend is None:
            backend = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=shots,
                                    seed_simulator=seed, seed_transpiler=seed)
            kernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)
            return kernel.evaluate
        else:
            kernel = QuantumKernel(feature_map=feature_map, quantum_instance=provider_backend)
            return kernel.evaluate

    else:
        return NotInitializedError

class NotInitializedError(Exception): 
    """ quantum parameters have not been initialized """ 