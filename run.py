# -*- encoding: utf-8 -*-
import re, random
from pathlib import Path
from typing import List
import pickle
import json
from pprint import pprint
from icecream import ic
import os, sys, shutil
from vit_pytorch import ViT
import torch
from utils.logger import get_logger

logger = get_logger(name=__name__)

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
logger.info(preds)

if __name__ == '__main__':
    
    print()
