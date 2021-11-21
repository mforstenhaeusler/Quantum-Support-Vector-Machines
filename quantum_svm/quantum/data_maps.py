import numpy as np
import functools

class DataMap:
    def __init__(self, default=True) -> None:
        self.default = default

    def __repr__(self) -> str:
        if self.default:
            return "phi_S : x --> {x if S={i}, (pi - x[0])(pi - x[0] if S={i,j})"
        else:
            raise NotImplementedError

    def _custom(self, x) -> float:
        raise NotImplementedError
    
    def _default(self, x) -> float:
        """ Data map function, f: R^n -> R
        
        Params:
        -------
        x : np.ndarray
            data

        Returns:
        --------
        coef : float
            the mapped value
        """
        coeff = x[0] if len(x) == 1 else functools.reduce(lambda m, n: m * n, np.pi - x)
        return coeff
    
    def map(self, x):
        if self.default:
            return self._default(x)
        else:
            return self._custom(x) 


class DataMap_Sin(DataMap):
    def __init__(self, default=False) -> None:
        super().__init__(default)
        self.default
    
    def __repr__(self) -> str:
        if self.default:
            return "phi_S : x --> {x if S={i}, (pi - x[i])(pi - x[j]) if S={i,j})"
        else:
            return "phi_S : x --> {x if S={i}, sin((pi - x[i]))sin((pi - x[j])) if S={i,j})"

    def _custom(self, x) -> float:
        """ Data map function, f: R^n -> R
    
        Params:
        -------
        x : np.ndarray
            data

        Returns:
        --------
        coef : float
            the mapped value
        """
        coeff = x[0] if len(x) == 1 else functools.reduce(lambda m, n: m * n, np.sin(np.pi - x))
        return coeff
        

class DataMap_Exp(DataMap):
    def __init__(self, default=False) -> None:
        super().__init__(default)
        self.default
    
    def __repr__(self) -> str:
        if self.default:
            return "phi_S : x --> {x if S={i}, (pi - x[i])(pi - x[j]) if S={i,j})"
        else:
            return "phi_S : x --> {x if S={i}, sin((pi - x[i]))sin((pi - x[j])) if S={i,j})"

    def _custom(self, x) -> float:
        """ Data map function, f: R^n -> R
        implements Eq (10) from Suzuki et. al arXiv:1906.10467v3 [quant-ph]

        Params:
        -------
        x : np.ndarray
            data

        Returns:
        --------
        coef : float
            the mapped value
        """
        coeff = x[0] if len(x) == 1 else functools.reduce(lambda m, n: np.pi*np.exp(((m - n)*(m - n))/8), x)
        return coeff

class DataMap_Exp(DataMap):
    def __init__(self, default=False) -> None:
        super().__init__(default)
        self.default
    
    def __repr__(self) -> str:
        if self.default:
            return "phi_S : x --> {x if S={i}, (pi - x[i])(pi - x[j]) if S={i,j})"
        else:
            return "phi_S : x --> {x if S={i}, sin((pi - x[i]))sin((pi - x[j])) if S={i,j})"

    def _custom(self, x) -> float:
        """ Data map function, f: R^n -> R
        implements Eq (11) from Suzuki et. al arXiv:1906.10467v3 [quant-ph]
    
        Params:
        -------
        x : np.ndarray
            data

        Returns:
        --------
        coef : float
            the mapped value
        """
        coeff = x[0] if len(x) == 1 else functools.reduce(lambda m, n: (np.pi/3)*(m * n), 1/(np.cos(x)))
        return coeff
    