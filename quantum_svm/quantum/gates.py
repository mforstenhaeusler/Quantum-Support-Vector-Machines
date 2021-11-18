import numpy as np

class QuantumGate:
    def __init__(self, n):
        self.wires = []
        self.n = n
        self.op_mat = None

    def _build_naive_op_mat(self):
        raise NotImplementedError

    def apply(self, st_vec: np.ndarray):
        raise NotImplementedError

    def naive_apply(self, st_vec: np.ndarray):
        if self.op_mat is None: self._build_naive_op_mat()
        return self.op_mat @ st_vec

    def __len__(self):
        return len(self.wires)

class SingleQubitQuantumGate(QuantumGate):
    def __init__(self, wire, n, single_qubit_matrix):
        super().__init__(n)
        self.wires = [int(wire)]
        assert len(self.wires) == 1
        self.single_qubit_matrix = single_qubit_matrix
        assert self.single_qubit_matrix.shape == (2, 2)

    def _build_naive_op_mat(self):
        self.op_mat = np.eye(1, dtype=np.complex_)
        wire = self.wires[0]
        for i in range(self.n):
            if i != wire:
                self.op_mat = np.kron(np.eye(2, dtype=np.complex_), self.op_mat)
            else:
                self.op_mat = np.kron(self.single_qubit_matrix, self.op_mat)

    def apply(self, st_vec: np.ndarray):
        wire = self.wires[0]
        n = self.n

        st_vec = st_vec.reshape((2 ** (n - wire - 1), 2, 2 ** wire)).astype(np.complex_)
        new_st_vec = st_vec.copy()
        new_st_vec[:, 0, :] = st_vec[:, 0, :] * self.single_qubit_matrix[0, 0] + st_vec[:, 1, :] * self.single_qubit_matrix[0, 1]
        new_st_vec[:, 1, :] = st_vec[:, 0, :] * self.single_qubit_matrix[1, 0] + st_vec[:, 1, :] * self.single_qubit_matrix[1, 1]
        new_st_vec = new_st_vec.reshape((2 ** n))

        return new_st_vec