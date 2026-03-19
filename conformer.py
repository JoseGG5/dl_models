import torch
import torch.nn as nn
import numpy as np

if __name__ == "__main__":
    signal = torch.rand(size=48_000*5)
    print(signal.shape)