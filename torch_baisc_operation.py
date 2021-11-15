# -*- encoding: utf-8 -*-
from pathlib import Path
from typing import List
import pickle
import json
from pprint import pprint
from utils.logger import get_logger
from icecream import ic
import os, sys, shutil
import torch
from torch import nn, einsum
import torch.nn.functional as F
sys.path.append(os.path.abspath('.'))
from utils.logger import get_logger

logger = get_logger(name=__name__)

a = torch.Tensor([[1,2,4.]])
b = torch.Tensor([[4,5,7], [3,9,8], [9,6,7]])
c = torch.cat((a,b), dim=0)
d = torch.stack((b, b), dim=1)
ic(d.shape)
ic(c)
ic(-torch.finfo(a.dtype).max)
ic(torch.finfo(a.dtype).min)
ic(d.flatten(1))
id = nn.Identity()
ic(id(c))
b = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 0, 1]], dtype=torch.bool)
b = F.pad(b, (1, 0), value=True)
print(b.shape)
# add dimension
ic(b[:,None, :].shape)
ic(b[:, :, None].shape)
e = b[:,None, :] * b[:, :, None]
f = b[:,None, :].mul(b[:, :, None])
ic(e)
ic(~e)
# ic(f)