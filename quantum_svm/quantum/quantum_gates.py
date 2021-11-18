import numpy as np
from .gates import SingleQubitQuantumGate, QuantumGate
from .utils import X_gate, Y_gate, Z_gate, hadamard_gate

class Hadamard(SingleQubitQuantumGate):
    def __init__(self, wires, n):
        super().__init__(wires, n, hadamard_gate())

    def apply(self, st_vec: np.ndarray):
        wire = self.wires[0]
        n = self.n

        # reshape qubit statevector to [2,2,1]
        st_vec = st_vec.reshape((2 ** (n - wire - 1), 2, 2 ** wire)).astype(np.complex_)
        
        # copy the vector 
        new_st_vec = st_vec.copy()

        #apply the Hadamard Gate
        new_st_vec[:, 0, :] = (st_vec[:, 0, :] + st_vec[:, 1, :]) * (1/np.sqrt(2))
        new_st_vec[:, 1, :] = (st_vec[:, 0, :] - st_vec[:, 1, :]) * (1/np.sqrt(2))
        new_st_vec = new_st_vec.reshape((2 ** n))

        return new_st_vec

class PauliX(SingleQubitQuantumGate):
    def __init__(self, wire, n):
        super().__init__(wire, n, X_gate())

    def apply(self, st_vec: np.ndarray):
        wire = self.wires[0]
        n = self.n

        st_vec = st_vec.reshape((2 ** (n - wire - 1), 2, 2 ** wire)).astype(np.complex_)
        st_vec = st_vec[:, [1, 0], :]
        st_vec = st_vec.reshape((2 ** n))

        return st_vec


class PauliY(SingleQubitQuantumGate):
    def __init__(self, wires, n):
        super().__init__(wires, n, Y_gate())
        self.multiplier = np.array([-1j, 1j], dtype=np.complex_).reshape((1, 2, 1))

    def apply(self, st_vec: np.ndarray):
        wire = self.wires[0]
        n = self.n

        st_vec = st_vec.reshape((2 ** (n - wire - 1), 2, 2 ** wire)).astype(np.complex_)
        st_vec = st_vec[:, [1, 0], :] * self.multiplier
        st_vec = st_vec.reshape((2 ** n))

        return st_vec


class PauliZ(SingleQubitQuantumGate):
    def __init__(self, wires, n):
        super().__init__(wires, n, Z_gate())

    def apply(self, st_vec: np.ndarray):
        wire = self.wires[0]
        n = self.n

        st_vec = st_vec.copy().reshape((2 ** (n - wire - 1), 2, 2 ** wire)).astype(np.complex_)
        st_vec[:, 1, :] = st_vec[:, 1, :] * (-1)
        st_vec = st_vec.reshape((2 ** n))

        return st_vec

class CNOT(QuantumGate):
    def __init__(self, wires, target, n):
        super().__init__(n)
        self.wires = list(sorted(wires))
        self.target = target
        assert len(self.wires) == 2
        assert self.target in self.wires

    def _build_naive_op_mat(self):
        zero_mat = np.eye(1, dtype=np.complex_)
        one_mat = np.eye(1, dtype=np.complex_)
        for i in range(self.n):
            if i not in self.wires:
                zero_mat = np.kron(np.eye(2, dtype=np.complex_), zero_mat)
                one_mat = np.kron(np.eye(2, dtype=np.complex_), one_mat)
            else:
                if i != self.target:
                    zero_mat = np.kron(np.array([[1, 0], [0, 0]], dtype=np.complex_), zero_mat)
                    one_mat = np.kron(np.array([[0, 0], [0, 1]], dtype=np.complex_), one_mat)
                else:
                    zero_mat = np.kron(np.eye(2, dtype=np.complex_), zero_mat)
                    one_mat = np.kron(np.array([[0, 1], [1, 0]], dtype=np.complex_), one_mat)
        self.op_mat = zero_mat + one_mat

    def apply(self, st_vec: np.ndarray):
        n = self.n

        slices = []
        pref_wire = n
        for wire in reversed(self.wires):
            slices += [2 ** (pref_wire - wire - 1), 2]
            pref_wire = wire
        slices += [2 ** pref_wire]

        st_vec = st_vec.copy().reshape(tuple(slices)).astype(np.complex_)

        if self.target == self.wires[0]:
            st_vec[:, 1, :, :, :] = np.take(st_vec[:, 1, :, :, :], [1, 0], axis=2)
        else:
            st_vec[:, :, :, 1, :] = np.take(st_vec[:, :, :, 1, :], [1, 0], axis=1)

        st_vec = st_vec.reshape((2 ** n))
        return st_vec
    