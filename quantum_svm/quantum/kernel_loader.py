from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals

from quantum_svm.quantum.kernels import QuantumKernel as QuantumKernel_custom
from quantum_svm.quantum.feature_maps import ZZFeatureMap as ZZFeatureMap_custom
from quantum_svm.quantum.feature_maps import ZFeatureMap as ZFeatureMap_custom 

def quantum_kernel_loader(params, feature_map_in=None, data_map=None, qiskit_indicator=True):
    """ Quantum Kernel implementation using Qiskit 
    
    Params:
    -------
    params : dict
             quantum parameters required for init of feature_map and kernel  
    
    feature_map_in : parameterized qiskit circuit
                     if None, default Feature Map == ZZFeatureMap, else use initialized Feature Map

    data_map : float
               Data map function, f: R^n -> R

    qiskit_indicator : bool
                       determines if qiskit's QuantumKernel or custom implementation is used 

    Return:
    -------
    Quantum kernel prepared for classification 
    """
    if params is not None:
        feature_dimension= params['feature_dimension']
        reps=params['reps']
        seed=params['seed']
        shots=params['shots']
        provider_backend=params['provider_backend']

        
        if qiskit_indicator:
            if feature_map_in is None:
                algorithm_globals.random_seed = seed

                feature_map = ZZFeatureMap(
                    feature_dimension=feature_dimension, 
                    reps=reps, 
                    entanglement='linear',
                    data_map_func=data_map.map, 
                    insert_barriers=True
                )
            else:
                feature_map = feature_map_in(
                    feature_dimension=feature_dimension, 
                    reps=reps, 
                    entanglement='linear',
                    data_map_func=data_map.map, 
                    insert_barriers=True
                )
            
                if provider_backend is None:
                    backend = QuantumInstance(
                            Aer.get_backend('qasm_simulator'), 
                            shots=shots, seed_simulator=seed, seed_transpiler=seed
                        )
                else:
                    backend = provider_backend

                    kernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)
                    return kernel.evaluate
        else: 
            if feature_map_in is None:
                feature_map_custom = ZZFeatureMap_custom(
                    feature_dimension, 
                    reps, 
                    data_map, 
                    insert_barriers=True
                )
            else:
                feature_map_custom = feature_map_in(
                    feature_dimension, 
                    reps, 
                    data_map, 
                    insert_barriers=True
                )

                if provider_backend is None:
                    backend = Aer.get_backend('qasm_simulator')
                else: 
                    backend = provider_backend

                kernel_custom = QuantumKernel_custom(
                    feature_map=feature_map_custom, 
                    quantum_backend=backend, 
                    sim_params=params
                )

                return kernel_custom.evaluate
    else:
        return NotInitializedError

class NotInitializedError(Exception): 
    """ Quantum parameters have not been initialized """  