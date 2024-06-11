import test2
import my_utils
import torch
import run_websocket_client
import asyncio
import syft as sy
from syft.workers.virtual import VirtualWorker
from datetime import datetime
from syft.workers.websocket_client import WebsocketClientWorker
from syft.generic.pointers.object_wrapper import ObjectWrapper
from syft.messaging.message import ObjectMessage

hook = sy.TorchHook(torch)

model = my_utils.ConvNet1D(input_size=400, num_classes=7)
traced_model = torch.jit.trace(model, torch.zeros([1, 400, 3], dtype=torch.float))
train_config = test2.MyTrainConfig(
    model=traced_model,
    loss_fn=run_websocket_client.loss_fn,
    batch_size=32,
    shuffle=True,
    epochs=5,
    optimizer='SGD'
)


worker_a = my_utils.MyWebsocketClientWorker(hook=hook, host='192.168.3.5', port=9292, id="AA")
worker_b = my_utils.MyWebsocketClientWorker(hook=hook, host='192.168.3.6', port=9292, id="BB")
worker_c = my_utils.MyWebsocketClientWorker(hook=hook, host='192.168.3.9', port=9292, id="CC")
worker_d = my_utils.MyWebsocketClientWorker(hook=hook, host='192.168.3.15', port=9292, id="DD")
worker_e = my_utils.MyWebsocketClientWorker(hook=hook, host='192.168.3.16', port=9292, id="EE")

worker_list = [worker_a, worker_b, worker_c, worker_d, worker_e]

for worker in worker_list:
    worker.clear_objects_remote()


async def main():
    start = datetime.now()
    await asyncio.gather(
        *[
            train_config.async_wrap_and_send(traced_model, worker)
            for worker in worker_list
        ]
    )
    print(f'{(datetime.now()-start).total_seconds()}')

asyncio.get_event_loop().run_until_complete(main())