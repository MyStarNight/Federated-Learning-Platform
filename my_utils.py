import binascii
from typing import Union
from typing import List

import torch
import torch.nn as nn

import syft as sy
from syft.workers.base import BaseWorker
import weakref
import torch
from syft.workers.websocket_client import WebsocketClientWorker
from syft.workers.websocket_server import WebsocketServerWorker
import websockets
from syft.messaging.message import ObjectRequestMessage
from datetime import datetime

TIMEOUT_INTERVAL = 30


def model_to_device(model, device='cpu'):
    """由于在此环境下，模型进行计算以后无法在cpu和cuda之间切换
    因此使用这个函数，进行模型的运算设备切换

    Args:
        model: 模型
        device: cpu or cuda

    Returns:
        traced model to device you set
    """
    new_model = ConvNet1D(input_size=400, num_classes=7).to(device)
    new_model.load_state_dict(model.state_dict())
    traced_model = torch.jit.trace(new_model, torch.zeros([1, 400, 3], dtype=torch.float).to(device))
    return traced_model

class ConvNet1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ConvNet1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * ((input_size-3+1)//2), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class MyWebsocketClientWorker(WebsocketClientWorker):
    async def async_fit_on_device(self, dataset_key: str, return_ids: List[int] = None):
        if return_ids is None:
            return_ids = [sy.ID_PROVIDER.pop()]

        # Close the existing websocket connection in order to open a asynchronous connection
        # This code is not tested with secure connections (wss protocol).
        self.close()
        async with websockets.connect(
            self.url, timeout=TIMEOUT_INTERVAL, max_size=None, ping_timeout=TIMEOUT_INTERVAL
        ) as websocket:
            message = self.create_worker_command_message(
                command_name="fit_on_device", return_ids=return_ids, dataset_key=dataset_key
            )

            # Send the message and return the deserialized response.
            serialized_message = sy.serde.serialize(message)
            await websocket.send(str(binascii.hexlify(serialized_message)))
            await websocket.recv()  # returned value will be None, so don't care

        # Reopen the standard connection
        self.connect()

        # Send an object request message to retrieve the result tensor of the fit() method
        msg = ObjectRequestMessage(return_ids[0], None, "")
        serialized_message = sy.serde.serialize(msg)
        response = self._send_msg(serialized_message)

        # Return the deserialized response.
        return sy.serde.deserialize(response)


class MyWebsocketServerWorker(WebsocketServerWorker):
    def fit_on_device(self, dataset_key: str, **kwargs):
        """
        此函数可以让设备在接受到模型以后，若有cuda，可以自动调整到cuda进行运算
        Args:
            dataset_key: 训练数据集的名称
            **kwargs:

        Returns:
            训练之后的loss
        """
        self._check_train_config()

        # 自动选择训练设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'{device} is available.')

        if dataset_key not in self.datasets:
            raise ValueError(f"Dataset {dataset_key} unknown.")

        # 将模型转移到所需的训练设备上
        traced_model = self.get_obj(self.train_config._model_id).obj
        model = model_to_device(traced_model, device)
        loss_fn = self.get_obj(self.train_config._loss_fn_id).obj

        self._build_optimizer(
            self.train_config.optimizer, model, optimizer_args=self.train_config.optimizer_args
        )

        return self._fit_on_device(model=model, traced_model=traced_model, dataset_key=dataset_key, loss_fn=loss_fn, device =device)

    def _fit_on_device(self, model, traced_model, dataset_key, loss_fn, device):
        # 训练开始时间
        start_time = datetime.now()
        print(f"Training start time: {start_time}")

        # 训练过程
        model.train()
        data_loader = self._create_data_loader(
            dataset_key=dataset_key, shuffle=self.train_config.shuffle
        )

        loss = None
        iteration_count = 0

        for _ in range(self.train_config.epochs):
            for (data, target) in data_loader:
                # Set gradients to zero
                self.optimizer.zero_grad()

                # Update model
                output = model(data.to(device))
                loss = loss_fn(target=target.to(device), pred=output)
                loss.backward()
                self.optimizer.step()

                # Update and check interation count
                iteration_count += 1
                if iteration_count >= self.train_config.max_nr_batches >= 0:
                    break

        # 训练结束时间和消耗的时间
        end_time = datetime.now()
        print(f"Training end time: {end_time}")
        print(f"Time Consuming: {(end_time-start_time).total_seconds()}\n")

        # 将训练好的参数加载到traced_model上
        new_model = model_to_device(model, 'cpu')
        traced_model.load_state_dict(new_model.state_dict())

        return loss.to('cpu')