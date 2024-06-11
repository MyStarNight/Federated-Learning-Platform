import asyncio
import binascii
import time
from typing import Union

import syft as sy
import websockets
from syft import TrainConfig
from syft.workers.websocket_client import WebsocketClientWorker
from syft.generic.pointers.object_wrapper import ObjectWrapper
from syft.messaging.message import ObjectMessage
from syft.generic.tensor import AbstractTensor
from syft.generic.frameworks.types import FrameworkTensorType
import websockets
from datetime import datetime


class MyTrainConfig(TrainConfig):
    async def async_wrap_and_send(
            self,
            obj: Union[FrameworkTensorType, AbstractTensor],
            location: WebsocketClientWorker
    ):
        try:
            location.close()

            async with websockets.connect(
                    location.url, timeout=60, max_size=None, ping_timeout=60
            ) as websocket:
                obj_id = sy.ID_PROVIDER.pop()
                print(location.id, obj_id)
                obj_with_id = ObjectWrapper(id=obj_id, obj=obj)
                obj_message = ObjectMessage(obj_with_id)
                bin_message = sy.serde.serialize(obj_message, worker=self.owner)
                print(f"User-{location.id} sending start:{datetime.now()}")
                await websocket.send(str(binascii.hexlify(bin_message)))
                print(f"User-{location.id} sending end:{datetime.now()}")
                await websocket.recv()
                print(f"User-{location.id} receive:{datetime.now()}")

            location.connect()

            return

        except Exception as e:
            print(f"An error occurred during async_wrap_and_send: {str(e)}")
            return None


