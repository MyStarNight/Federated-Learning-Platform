import logging
import argparse
import sys
import asyncio
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
from my_utils import MyWebsocketClientWorker
from datetime import datetime

import run_websocket_client


def model_to_device(model: run_websocket_client.ConvNet1D, device=None):
    for param in model.parameters():
        # param.data = param.data.to(device)
        print(param.data.shape)
        print(param.data.type())
        print(param.data)
        print('')
    for buf in model.buffers():
        buf.data = buf.data.to(device)

async def main():
    hook = sy.TorchHook(torch)

    raspi = {"host": "192.168.3.4", "hook": hook}
    jetson_nano = {"host": "192.168.3.5", "hook": hook}

    worker_a = MyWebsocketClientWorker(id='A', port=9292, **jetson_nano)
    worker_b = MyWebsocketClientWorker(id='B', port=9292, **raspi)

    worker_instances = [worker_a, worker_b]
    client_devices = ['cuda', 'cpu']

    for worker in worker_instances:
        worker.clear_objects_remote()

    model = run_websocket_client.ConvNet1D(input_size=400, num_classes=7).to('cuda')
    traced_model = torch.jit.trace(model, torch.zeros([1, 400, 3], dtype=torch.float).to('cuda'))

    test_num = 5
    for curr_round in range(1, 1+1):
        print(curr_round)
        results = await asyncio.gather(
            *[
                run_websocket_client.fit_model_on_worker(
                    worker=worker,
                    traced_model=traced_model,
                    batch_size=16,
                    curr_round=curr_round,
                    max_nr_batches=10,
                    lr=0.001,
                    device=client_device
                )
                for worker, client_device in zip(worker_instances, client_devices)
            ]
        )

        test_models = curr_round % test_num == 0 or curr_round == 1
        if test_models:
            print(results)

        model_list = []
        for worker_id, model, _, _1 in results:
            model_list.append(model)
            # model_to_device(model, 'cuda')
            print(next(model.parameters()).is_cuda)

        start_time = datetime.now()
        new_model = run_websocket_client.ConvNet1D(input_size=400, num_classes=7).to('cuda')
        new_model.load_state_dict(model_list[0].state_dict())
        print(next(new_model.parameters()).is_cuda)
        print(type(new_model))
        traced_model = torch.jit.trace(new_model, torch.zeros([1, 400, 3], dtype=torch.float).to('cuda'))
        end_time = datetime.now()
        print((end_time-start_time).total_seconds())


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())