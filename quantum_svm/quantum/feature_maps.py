from qiskit import QuantumCircuit
from .data_maps import DataMap


class ZZFeatureMap:
    def __init__(self, n_qubits, reps, data_map=None, insert_barriers=False) -> None:
        """ ZZFeature Map
        Special Pauli-Feature Map. Implements Equation 41 from the slides.

        Params:
        -------
        n_qubits : int 
                   number of qubits

        reps : int
               number of repetitions of unitary operator 
        
        data_map : float
                   Data map function, f: R^n -> R

        insert_barriers : bool
                          if true, inserts barriers into the quantum circuit in qiskit
        """
        self.n_qubits = n_qubits
        if data_map is None:
            self.data_map = DataMap()
        else:
            self.data_map = data_map
        self.reps = reps
        self.insert_barriers = insert_barriers
        self._circuit = None
    
    def map(self, data, reverse=False):
        """ Builds the parameterized quantum circuit of the ZZ-Feature Map"""
        circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.reps):
            if i > 0:
                if self.insert_barriers: circuit.barrier()
            circuit.h(0)
            circuit.h(1)
            if self.insert_barriers: circuit.barrier()
            if not reverse:  
                circuit.u(0,0,2*self.data_map.map(data[:1]),0)
                circuit.u(0,0,2*self.data_map.map(data[1:]),1)
            else:
                circuit.u(0,0,-2*self.data_map.map(data[:1]),0)
                circuit.u(0,0,-2*self.data_map.map(data[1:]),1)
            circuit.cx(0,1)
            if not reverse: 
                circuit.u(0,0,2*self.data_map.map(data),1)
            else:
                circuit.u(0,0,-2*self.data_map.map(data),1)
            circuit.cx(0,1)      
        
        if not reverse: 
            return circuit.to_instruction()
        else:
            return circuit.to_instruction().reverse_ops()
    
    def __repr__(self) -> str:
        return f"ZZFeatureMap(feature_dimensions={self.n_qubits}, reps={self.reps})"

    def __str__(self) -> str:
        return f"ZZFeatureMap(feature_dimensions={self.n_qubits}, reps={self.reps})"

class ZFeatureMap:
    def __init__(self, n_qubits, reps, data_map, insert_barriers=False) -> None:
        """ ZFeature Map
        Special Pauli-Feature Map.

        Params:
        -------
        n_qubits : int 
                   number of qubits
                   
        reps : int
               number of repetitions of unitary operator 
        
        data_map : float
                   Data map function, f: R^n -> R

        insert_barriers : bool
                          if true, inserts barriers into the quantum circuit in qiskit
        """
        self.n_qubits = n_qubits
        self.data_map = data_map
        self.reps = reps
        self.insert_barriers = insert_barriers
        self._circuit = None
    
    def map(self, data, reverse=False):
        """ Builds the parameterized quantum circuit of the Z-Feature Map"""
        circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.reps):
            if i > 0:
                if self.insert_barriers: circuit.barrier()
            circuit.h(0)
            circuit.h(1)
            if self.insert_barriers: circuit.barrier()
            if not reverse:  
                circuit.u(0,0,2*self.data_map.map(data[:1]),0)
                circuit.u(0,0,2*self.data_map.map(data[1:]),1)
            else:
                circuit.u(0,0,-2*self.data_map.map(data[:1]),0)
                circuit.u(0,0,-2*self.data_map.map(data[1:]),1)     
        
        if not reverse: 
            return circuit.to_instruction()
        else:
            return circuit.to_instruction().reverse_ops()
    
    def __repr__(self) -> str:
        return f"ZFeatureMap(feature_dimensions={self.n_qubits}, reps={self.reps})"
    
    def __str__(self) -> str:
        return f"ZFeatureMap(feature_dimensions={self.n_qubits}, reps={self.reps})"