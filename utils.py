import os
import shutil
import time
import pprint

import torch
import torch.nn.functional as F


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def delete_path(path):
    shutil.rmtree(path)
    os.makedirs(path)



class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

class Averager_vector():

    def __init__(self,num):
        self.n = 0
        self.v = torch.zeros(num)

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

class Averager_matrix():

    def __init__(self,M, N):
        self.n = 0
        self.v = torch.zeros(M,N)

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v



class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


if __name__=='__main__':
    output = torch.randn(10,6)
    pred = F.softmax(output,dim=1)
    print(pred)
    index_common, index_private = find_know_and_unknow(pred, 3)
    print(index_common)
    print(index_private)