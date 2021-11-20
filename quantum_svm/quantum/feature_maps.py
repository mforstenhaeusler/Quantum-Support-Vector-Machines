from numpy import ndarray
from qiskit import QuantumCircuit
from .utils import data_map_func


class ZZFeatureMap:
    def __init__(self, n_qubits, reps, data_map=data_map_func, insert_barriers=False):
        self.n_qubits = n_qubits
        self.data_map = data_map
        self.reps = reps
        self.insert_barriers = insert_barriers
    
    def map(self, data: ndarray, reverse=False):
        circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.reps):
            if i > 0:
                if self.insert_barriers: circuit.barrier()
            circuit.h(0)
            circuit.h(1)
            if self.insert_barriers: circuit.barrier()
            if not reverse:  
                circuit.u(0,0,2*self.data_map(data[:1]),0)
                circuit.u(0,0,2*self.data_map(data[1:]),1)
            else:
                circuit.u(0,0,-2*self.data_map(data[:1]),0)
                circuit.u(0,0,-2*self.data_map(data[1:]),1)
            circuit.cx(0,1)
            if not reverse: 
                circuit.u(0,0,2*self.data_map(data),1)
            else:
                circuit.u(0,0,-2*self.data_map(data),1)
            circuit.cx(0,1)      
        
        if not reverse: 
            return circuit.to_instruction()
        else:
            return circuit.to_instruction().reverse_ops()


class ZZFeatureMap_2:
    def __init__(self, n_qubits, reps, data_map=data_map_func, insert_barriers=False):
        self.n_qubits = n_qubits
        self.data_map = data_map
        self.reps = reps
        self.insert_barriers = insert_barriers
    
    def map_statevector(self, X1, X2):
        circuit = QuantumCircuit(self.n_qubits)
        # U
        for i in range(self.reps):
            if i > 0:
                if self.insert_barriers: circuit.barrier()
            circuit.h(0)
            circuit.h(1)
            if self.insert_barriers: circuit.barrier()
            circuit.u(0,0,2*self.data_map(X1[:1]),0)
            circuit.u(0,0,2*self.data_map(X1[1:]),1)
            circuit.cx(0,1)
            circuit.u(0,0,2*self.data_map(X1),1)
            circuit.cx(0,1) 
        
        
        # U^{dagger}
        for i in range(self.reps):
            if i > 0:
                if self.insert_barriers: circuit.barrier()
            circuit.cx(0,1)        
            circuit.u(0,0,-2*self.data_map(X2),1)
            circuit.cx(0,1)
            circuit.u(0,0,-2*self.data_map(X2[:1]),0)
            circuit.u(0,0,-2*self.data_map(X2[1:]),1)
            if self.insert_barriers: circuit.barrier()
            circuit.h(0)
            circuit.h(1)
            if self.insert_barriers: circuit.barrier()
        
        return circuit.to_instruction()