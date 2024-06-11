import logging
import argparse
import sys
import asyncio
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os

import syft as sy
from syft.workers import websocket_client
from syft.workers.websocket_client import WebsocketClientWorker
from syft.frameworks.torch.fl import utils

import my_utils
from my_utils import MyWebsocketClientWorker, model_to_device, ConvNet1D

hook = sy.TorchHook(torch)

# 连接代码
worker_a = MyWebsocketClientWorker(hook=hook, host="192.168.3.15", port=9292, id="DD")
worker_b = MyWebsocketClientWorker(hook=hook, host="192.168.3.16", port=9292, id="EE")

tensor = torch.tensor([1, 2])
# tensor.send(worker_a)

# 关闭连接
# worker_a.close()
# worker_b.close()
