import logging
import argparse
import numpy as np
import torch
# from torchvision import datasets
# from torchvision import transforms

from torch.utils.data import ConcatDataset, DataLoader
import pickle
import syft as sy
from syft.workers import websocket_server
from collections import Counter

import my_utils
from my_utils import MyWebsocketServerWorker
import socket

client_device_mapping_id = my_utils.client_device_mapping_id

KEEP_LABELS_DICT = {
    "A": [1],
    "B": [2],
    "C": [3],
    "D": [4],
    "E": [5],
    "F": [6],
    "G": [7],
    "H": [8],
    "I": [9],
    "J": [10],
    "AA": [11, 12],
    "BB": [13, 14],
    "CC": [15, 16],
    "DD": [17, 18],
    "EE": [19, 20],
    "testing": [21, 22, 23, 24],
    None: [21, 22, 23, 24],
}


def get_host_ip():
    """
    查询本机ip地址
    :return: ip
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def start_websocket_server_worker(id, host, port, hook, verbose, n_samples, keep_users=None, training=True,):
    """Helper function for spinning up a websocket server and setting up the local datasets."""

    server = MyWebsocketServerWorker(
        id=id, host=host, port=port, hook=hook, verbose=verbose
    )

    logger.info(f"Federated Worker ID: {id}, IP: {ip}, port: {port}", )
    logger.info(f"selected user: {keep_users}")

    # 加载数据集
    data_path = '../Dataset/HAR/shuffled_HAR_datasets.pkl'
    with open(data_path, 'rb') as f:
        HAR_datasets = pickle.load(f)

    if training:
        selected_data = []
        selected_target = []
        for user in keep_users:
            selected_data.append(HAR_datasets[user].tensors[0])
            selected_target.append(HAR_datasets[user].tensors[-1])

        # 将选择的数据集dict形式转换为tensor形式储存
        selected_data = torch.cat(selected_data, dim=0)
        selected_target = torch.cat(selected_target, dim=0)

        # 计算stage的次数
        length = len(selected_data)
        n_train_stages = length//n_samples + 1

        # 建立不同stage的数据集
        for stage in range(1, n_train_stages+1):
            start_index = n_samples * (stage - 1)
            end_index = n_samples * stage
            selected_data_tensor = selected_data[start_index: end_index]
            selected_target_tensor = selected_target[start_index: end_index]

            dataset = sy.BaseDataset(
                data=selected_data_tensor,
                targets=selected_target_tensor
            )

            print(f"stage{stage}: {Counter(selected_target_tensor.argmax(dim=1).numpy())}, num of samples: {len(selected_target_tensor)}")
            key = "HAR-" + str(stage)
            server.add_dataset(dataset, key)

    else:
        selected_data = []
        selected_target = []
        for user in keep_users:
            selected_data.append(HAR_datasets[user].tensors[0])
            selected_target.append(HAR_datasets[user].tensors[-1])

        selected_data_tensor = torch.cat(selected_data, dim=0)
        selected_target_tensor = torch.cat(selected_target, dim=0)

        dataset = sy.BaseDataset(
            data=selected_data_tensor,
            targets=selected_target_tensor.argmax(dim=1)
        )

        print(Counter(selected_target_tensor.argmax(dim=1).numpy()))
        key = "HAR-testing"

        server.add_dataset(dataset, key)
        logger.info(f"selected datasets shape:{selected_data_tensor.shape} ")

    logger.info(f"datasets: f{server.datasets.keys()}")

    server.start()
    return server


if __name__ == '__main__':
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("run_websocket_server")
    logger.setLevel(level=logging.DEBUG)

    ip = get_host_ip()
    worker_id = client_device_mapping_id[ip]

    # Parse args
    # port, host ,testing, verbose
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=9292,
        help="port number of the websocket server worker, e.g. --port 9292",
    )
    parser.add_argument("--host", type=str, default=ip, help="host for the connection")
    parser.add_argument(
        "--testing",
        action="store_true",
        help="if set, websocket server worker will load the test dataset instead of the training dataset",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket server worker will be started in verbose mode",
    )
    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default=80,
        help="num of samples for one stage training"
    )

    args = parser.parse_args()
    hook = sy.TorchHook(torch)

    server = start_websocket_server_worker(
        id=worker_id,
        host=args.host,
        port=args.port,
        hook=hook,
        verbose=args.verbose,
        n_samples=args.num,
        keep_users=KEEP_LABELS_DICT[worker_id],
        training=not args.testing,
    )