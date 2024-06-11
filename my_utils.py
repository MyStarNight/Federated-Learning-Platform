import binascii
import os.path
from typing import Union
from typing import List
import torch.nn.functional as F
import os

import numpy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import syft as sy
from syft.generic.pointers.object_wrapper import ObjectWrapper
from syft.workers.abstract import AbstractWorker
from syft.frameworks.torch.fl import utils
from syft.workers.base import BaseWorker
import weakref
import torch
from syft.workers.websocket_client import WebsocketClientWorker
from syft.workers.websocket_server import WebsocketServerWorker

import websockets
from syft.messaging.message import ObjectRequestMessage
from datetime import datetime
import logging
import time

TIMEOUT_INTERVAL = 30

client_device_mapping_id = {
    "192.168.3.5": "AA",
    "192.168.3.6": "BB",
    "192.168.3.9": "CC",
    # "192.168.3.15": "DD",
    # "192.168.3.16": "EE",
    # "192.168.3.2": "A",
    # "192.168.3.3": "B",
    # "192.168.3.4": "C",
    # "192.168.3.7": "D",
    # "192.168.3.8": "E",
    # "192.168.3.10": "F",
    # "192.168.3.11": "G",
    # "192.168.3.12": "H",
    # "192.168.3.13": "I",
    # "192.168.3.20": "J",
    # "192.168.3.17": "testing"
}


def set_logger(save_path, action):
    path = os.path.join(save_path, f'{action}_{time.strftime("%Y-%m-%d_%H-%M-%S")}')
    if not os.path.exists(path):
        os.mkdir(path)
    file_path = os.path.join(path, "logging.txt")
    logging.basicConfig(
        filename=file_path,
        level=logging.DEBUG,
        format="%(asctime)s | %(message)s"
    )

    logger = logging.getLogger(action)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    return logger, path


def read_raspi_cpu_temperature():
    # Execute the command to get CPU temperature
    temp_output = os.popen('vcgencmd measure_temp').readline()
    # Extract temperature from the output
    temp = float(temp_output.replace("temp=", "").replace("'C\n", ""))
    return temp


def read_nano_gpu_temperature():
    # Path to the thermal zone corresponding to the GPU
    thermal_zone_path = "/sys/class/thermal/thermal_zone2/temp"

    try:
        # Read the temperature from the system file
        with open(thermal_zone_path, 'r') as file:
            temp_str = file.read().strip()
            # Convert the temperature from millidegrees to degrees Celsius
            temp = float(temp_str) / 1000.0
            return temp
    except FileNotFoundError:
        print("Thermal zone file not found. Ensure the path is correct for the GPU thermal zone.")
        return None


def plot_probability_histogram(data, save_path, bins='auto', title='Probability Histogram', xlabel='Data Points'):
    # 计算直方图数据和bin边界
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    # 计算bin的宽度
    bin_widths = np.diff(bin_edges)
    # 计算概率
    probabilities = counts * bin_widths

    # 绘制概率直方图
    plt.bar(bin_edges[:-1], probabilities, width=bin_widths, align='edge', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Probability')
    plt.savefig(save_path)
    # plt.show()


def plot_line_chart(data, save_path, title='Temperature Line Chart', xlabel='Epoch', ylabel='Temperature(°C)'):
    # 生成数据的索引作为X轴数据
    x_values = list(range(len(data)))

    # 创建折线图
    plt.figure(figsize=(10, 5))  # 可以调整图形大小
    plt.plot(x_values, data, marker='o', linestyle='-', color='b')  # 折线图，带圆形标记
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 添加网格线
    plt.grid(True)
    # 显示图表
    plt.savefig(save_path)
    # plt.show()



@torch.jit.script
def loss_fn(pred, target):
    return F.cross_entropy(pred, target.argmax(dim=1))


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


class AggregationPolicies:
    def __init__(self, aggregate_list: list):
        self.aggregate_worker_index = None
        self.aggregate_worker_num_list = aggregate_list
        self.length = len(aggregate_list)

    def reset(self):
        self.aggregate_worker_index = 0
        return self.aggregate_worker_index

    def delete_inaccessible_worker(self, worker_num):
        self.aggregate_worker_num_list[worker_num] = None

    def aggregate_in_order(self):
        while True:
            self.aggregate_worker_index += 1
            worker_num = self.aggregate_worker_num_list[self.aggregate_worker_index % self.length]
            if worker_num is not None:
                break
        return worker_num


class ConvNet1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ConvNet1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * ((input_size - 3 + 1) // 2), 128)
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
    async def async_fit_on_device(self, dataset_key: str, train_time_consuming_id, return_ids: List[int] = None):
        if return_ids is None:
            return_ids = [sy.ID_PROVIDER.pop()]

        # Close the existing websocket connection in order to open a asynchronous connection
        # This code is not tested with secure connections (wss protocol).
        self.close()
        async with websockets.connect(
                self.url, timeout=TIMEOUT_INTERVAL, max_size=None, ping_timeout=TIMEOUT_INTERVAL
        ) as websocket:
            message = self.create_worker_command_message(
                command_name="fit_on_device", return_ids=return_ids, train_time_consuming_id=train_time_consuming_id, dataset_key=dataset_key
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

    def aggregate(self, return_ids: List[int] = None):
        return self._send_msg_and_deserialize('model_aggregation')


class MyWebsocketServerWorker(WebsocketServerWorker):
    def __init__(self, hook, host: str, port: int, id, verbose):
        super().__init__(hook=hook, host=host, port=port, id=id, verbose=verbose)
        self.aggregate_config = None
        self.train_time_consuming = None

    def set_aggregate_config(self, ID: int):
        self.aggregate_config = self.get_obj(ID).obj
        # print(self.aggregate_config)

    def _check_aggregate_config(self):
        if self.aggregate_config is None:
            raise ValueError("Operation needs Aggregate object to be set.")
        print("Aggregate object is Okay.")

    def fit_on_device(self, dataset_key: str, train_time_consuming_id, **kwargs):
        """
        此函数可以让设备在接受到模型以后，若有cuda，可以自动调整到cuda进行运算
        Args:
            dataset_key: 训练数据集的名称
            **kwargs:
            train_time_consuming_id:

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

        return self._fit_on_device(model=model, traced_model=traced_model, dataset_key=dataset_key, loss_fn=loss_fn,
                                   device=device, train_time_consuming_id=train_time_consuming_id)

    def _fit_on_device(self, model, traced_model, dataset_key, loss_fn, device, train_time_consuming_id):
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
        train_time_consuming = (end_time - start_time).total_seconds()
        print(f"Time Consuming: {train_time_consuming}\n")

        # 将训练好的参数加载到traced_model上
        new_model = model_to_device(model, 'cpu')
        traced_model.load_state_dict(new_model.state_dict())

        self.train_time_consuming = torch.tensor([train_time_consuming])
        self.register_obj(self.train_time_consuming, train_time_consuming_id)

        return loss.to('cpu')

    def model_aggregation(self, **kwargs):
        # 聚合开始时间
        start_time = datetime.now()
        print(f"Aggregating start time: {start_time}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'{device} is available.')

        model_dict = {}
        for i, _model_id in enumerate(self.aggregate_config['model_id_list'][1]):
            worker_model = self.get_obj(_model_id).obj
            worker_model_to_local = model_to_device(worker_model, device)
            model_dict[i] = worker_model_to_local

        federated_model = utils.federated_avg(model_dict)
        model = model_to_device(federated_model, 'cpu')

        traced_model = self.get_obj(self.aggregate_config['federated_model_id']).obj
        traced_model.load_state_dict(model.state_dict())

        # 聚合结束时间
        end_time = datetime.now()
        print(f"Aggregating end time: {end_time}")
        print(f"Time Consuming: {(end_time - start_time).total_seconds()}\n")

        return


class AggregatedConfig():
    def __init__(self,
                 model_dict,
                 federated_model,
                 owner: AbstractWorker = None,
                 ID=None,
                 model_id_list=[],
                 federated_model_id=None
                 ):
        self.models = model_dict
        self.model_ptr_list = []
        self._model_id_list = model_id_list

        self.federated_model = federated_model
        self.federated_model_ptr = None
        self._federated_model_id = federated_model_id

        self.owner = owner if owner else sy.hook.local_worker
        self.id = ID if ID is not None else sy.ID_PROVIDER.pop()

    def _wrap_and_send_obj(self, obj, location):
        """Wrappers object and send it to location."""
        obj_with_id = ObjectWrapper(id=sy.ID_PROVIDER.pop(), obj=obj)
        obj_ptr = self.owner.send(obj_with_id, location)
        obj_id = obj_ptr.id_at_location
        return obj_ptr, obj_id

    def send_model(self, location: BaseWorker) -> weakref:
        self.model_ptr_list = []
        self._model_id_list = []
        # 发送需要聚合的模型
        for model in self.models.values():
            model_ptr, _model_id = self._wrap_and_send_obj(model, location)
            self.model_ptr_list.append(model_ptr)
            self._model_id_list.append(_model_id)

        # 发送一个聚合模型
        self.federated_model_ptr, self._federated_model_id = self._wrap_and_send_obj(self.federated_model, location)

    def get(self, ID, location):
        for model_id in self._model_id_list:
            self.owner.request_obj(model_id, location)
        return self.owner.request_obj(ID, location)

    @staticmethod
    def simplify(worker: BaseWorker, aggregate_config: "AggregatedConfig") -> dict:
        # 示例：仅简化ID和模型ID列表，实际应用中可能需要更复杂的逻辑
        return {
            'ID': sy.serde.msgpack.serde._simplify(worker, aggregate_config.id),
            'model_id_list': sy.serde.msgpack.serde._simplify(worker, aggregate_config._model_id_list),
            'federated_model_id': sy.serde.msgpack.serde._simplify(worker, aggregate_config._federated_model_id),
        }

    @staticmethod
    def detail(worker: BaseWorker, aggregate_config_tuple: tuple) -> "AggregatedConfig":
        # 从元组中恢复ID和模型ID列表
        id, model_id_list, federated_model_id = aggregate_config_tuple

        id = sy.serde.msgpack.serde._detail(worker, id)
        model_id_list = sy.serde.msgpack.serde._detail(worker, model_id_list)
        federated_model_id = sy.serde.msgpack.serde._detail(worker, federated_model_id)

        # 创建AggregatedConfig实例，可能需要额外的逻辑来处理model_dict和federated_model
        aggregate_config = AggregatedConfig(
            model_dict=None,  # 需要特定逻辑来处理
            federated_model=None,  # 需要特定逻辑来处理
            owner=worker,
            ID=id,
            model_id_list=model_id_list,
            federated_model_id=federated_model_id
        )

        return aggregate_config
