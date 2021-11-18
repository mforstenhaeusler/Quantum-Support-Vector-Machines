import numpy as np
from .gates import hadamard_gate, X_gate, Y_gate, Z_gate, CNOT

class Circuit:
    def __init__(self, qubits):
        self.qubits = qubits
        self.h = hadamard_gate()
        self.X_gate = X_gate()
        self.Y_gate = Y_gate()
        self.Z_gate = Z_gate()
