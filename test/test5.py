import asyncio
import websockets
import websocket
import threading
import syft as sy
import torch
import my_utils
from my_utils import MyWebsocketClientWorker
import run_websocket_client


hook = sy.TorchHook(torch)
client_device_mapping_id = my_utils.client_device_mapping_id

all_nodes = []
for ip, ID in client_device_mapping_id.items():
    kwargs_websocket = {"hook": hook, "host": ip, "port": 9292, "id": ID}
    all_nodes.append(MyWebsocketClientWorker(**kwargs_websocket))

model = my_utils.ConvNet1D(input_size=400, num_classes=7)
traced_model = torch.jit.trace(model, torch.zeros([1, 400, 3], dtype=torch.float))

train_config = sy.TrainConfig(
    model=traced_model,
    loss_fn=run_websocket_client.loss_fn,
    batch_size=32,
    shuffle=True,
    epochs=5,
    optimizer='SGD'
)

threads = []
for worker in all_nodes:
    # train_config._wrap_and_send_obj(traced_model, worker)
    thread = threading.Thread(target=train_config._wrap_and_send_obj)
