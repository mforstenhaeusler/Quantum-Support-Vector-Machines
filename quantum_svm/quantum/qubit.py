import numpy as np
from .quantum_gates import Hadamard

class Qubit:
    def __init__(self, bit='0'):
        """ 
        implements a quantum qubit
            bit = 0 - |0> = [1, 0], 
            bit = 1 - |1> = [0, 1], 
            
            following:
            |00> = [1,0,0,0]
            |01> = [0,1,0,0]
        
        Params:
        -------
        bit : str
              str of bits, i.e '0' == |0>, '1' == |1>, '01' == [0,1,0,0]
        """

        self.dim = 2**len(bit)
        self.isqrt2 = 1/np.sqrt(2) 
        self.bit = bit
        self.H = Hadamard(0,len(self.bit))
        
        self.state = np.zeros((1, self.dim), dtype=np.complex)

        if self.dim == 2:
            if bit == '0':
                self.state[0,0] = 1
            elif bit == '1':
                self.state[0,1] = 1
        if self.dim == 4:
            if bit == '00':
                self.state[0, 0] = 1
            elif bit == '01':
                self.state[0, 1] = 1
            elif bit == '10':
                self.state[0, 2] = 1
            else:
                self.state[0, 3] = 1

    def get_statevector(self):
        return self.state
    
    def get_statevector_superposition(self):
        return self.H.apply(self.state)
    
    def __repr__(self):
        return f"|{self.bit}> = \n\n{self.state.T}"

    def __str__(self):
        return f"|{self.bit}> = \n\n{self.state.T}"
    