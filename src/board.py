from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

writer = SummaryWriter()

for n_iter in range(10):
    tensors = torch.rand(100, 100)
    writer.add_histogram("rrr", tensors, n_iter + 1)
    # writer.add_scalar('Loss/train', np.random.random(), n_iter)
    # writer.add_scalar('Loss/test', np.random.random(), n_iter)
    # writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    # writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    


writer.add_histogram("origin_q", fcs[0].weight, 0)
writer.add_histogram("origin", tensors, 0)
writer.add_histogram("smoothed", tensors, 0)

writer.add