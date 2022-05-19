# Dung de debug khi gap loi giup cho bien luon khong doi tuy nhien dan den viec lam tang thoi gian train


import torch
import numpy as np
import random

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# If using cuda
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

x = torch.rand((5,5))
print(x)