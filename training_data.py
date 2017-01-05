import numpy as np
from typing import Sized, Iterable

class TrainingData(Sized, Iterable):
    def __init__(self,
                 input_dt: np.ndarray,
                 action_dt: np.ndarray,
                 Qvalue_dt: np.ndarray,
                 batch_size: int) -> None:
        assert input_dt.shape[0] == action_dt.shape[0] == Qvalue_dt.shape[0]
        self.n = input_dt.shape[0]
        self.input_dt = input_dt
        self.action_dt = action_dt
        self.Qvalue_dt = Qvalue_dt
        self.batch_size = batch_size
        self.batch_n = self.n // batch_size
        self.real_n = self.batch_n * batch_size

    def __iter__(self):
        '''
        Iter through batches
        yield (input, action, Qvalue) each time
        '''
        perm = np.random.permutation(self.n)
        s, e = 0, self.batch_size
        while e <= self.n:
            yield (self.input_dt[perm[s:e]],
                   self.action_dt[perm[s:e]],
                   self.Qvalue_dt[perm[s:e]])

            (s, e) = (e, e+self.batch_size)

    def __len__(self) -> int:
        '''Return the number of batches'''
        return self.batch_n
