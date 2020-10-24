from abc import ABCMeta, abstractmethod
import math
import numpy as np
from baredl.core import cupy

class Dataset(metaclass=ABCMeta):

    def __init__(self, train=True, transform=None, target_trainform=None):
        self.train = train
        self.transform = lambda x:x if transform is None else transform
        self.target_transform = lambda x:x if target_trainform is None else target_trainform
        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    @abstractmethod
    def prepare(self):
        """ implement data creation for self.data and self.label """
        pass


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.gpu = gpu
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)

        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i*batch_size : (i+1)*batch_size]
        batch = [self.dataset[i] for i in batch_index]

        if self.gpu and baredl.core.cupy is None:
            raise Exception('CuPy not loaded.')
        xp = baredl.core.cupy if self.gpu else np
        x = xp.array([e[0] for e in batch])
        t = xp.array([e[1] for e in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True