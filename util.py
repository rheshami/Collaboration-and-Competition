import torch
import numpy as np
class Utils(object):
    @staticmethod
    def getDevice():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



