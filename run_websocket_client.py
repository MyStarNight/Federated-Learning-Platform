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

LOG_INTERVAL = 25
logger = logging.getLogger("run_websocket_client")
# loss = nn.CrossEntropyLoss()


@torch.jit.script
def loss_fn(pred, target):
    return F.cross_entropy(pred, target.argmax(dim=1))


@torch.jit.script
def loss_fn_test(pred, target):
    return F.cross_entropy(pred, target)


def define_and_get_arguments(args=sys.argv[1:]):
    # 选定参数
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of the training")
    parser.add_argument(
        "--test_batch_size", type=int, default=128, help="batch size used for the test data"
    )
    parser.add_argument(
        "--training_rounds", type=int, default=5, help="number of federated learning rounds"
    )
    parser.add_argument(
        "--federate_after_n_batches",
        type=int,
        default=10,
        help="number of training steps performed on each remote worker before averaging",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--seed", type=int, default=12345, help="seed used for randomization")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket client workers will be started in verbose mode",
    )
    parser.add_argument("--stage", type=int, default=1, help="continual learning stage")

    args = parser.parse_args(args=args)
    return args


async def fit_model_on_worker(
    worker: MyWebsocketClientWorker,
    traced_model: torch.jit.ScriptModule,
    batch_size: int,
    curr_round: int,
    max_nr_batches: int,
    lr: float,
    dataset_key: str,
    state: bool
    # device: str
):
    if not state:
        return worker.id, None, None, None, None

    try:
        start_time = datetime.now()
        print(f"User-{worker.id} Federated Learning start time: {start_time}")

        train_config = sy.TrainConfig(
            model=traced_model,
            loss_fn=loss_fn,
            batch_size=batch_size,
            shuffle=True,
            # max_nr_batches=max_nr_batches,
            epochs=1,
            optimizer="SGD",
            optimizer_args={"lr": lr},
        )

        print(f'User-{worker.id} model send start {datetime.now()}')
        train_config.send(worker)
        print(f'User-{worker.id} model send end {datetime.now()}')

        train_time_consuming_id = sy.ID_PROVIDER.pop()
        # 远程训练模型；等待远程训练模型的完成

        print(f'User-{worker.id} model start training {datetime.now()}')
        loss = await worker.async_fit_on_device(dataset_key=dataset_key, return_ids=[0], train_time_consuming_id=train_time_consuming_id)
        print(f'User-{worker.id} model end training {datetime.now()}')

        print(f'User-{worker.id}: model get back start {datetime.now()}')
        model = train_config.model_ptr.get().obj
        print(f'User-{worker.id}: model get back end {datetime.now()}')

        train_time_consuming = train_config.owner.request_obj(train_time_consuming_id, worker)

        end_time = datetime.now()
        print(f"User-{worker.id} Federated Learning end time: {end_time}")
        consuming_time = end_time - start_time

        return worker.id, model, loss, consuming_time.total_seconds(), float(train_time_consuming)

    except Exception as e:
        print(f"User-{worker.id} {datetime.now()} Inaccessible: {e}")
        # traceback.print_exc()
        return worker.id, None, None, None, None


def evaluate_model_on_worker(
    model_identifier,
    worker,
    dataset_key,
    model,
    nr_bins,
    batch_size,
    device,
    print_target_hist=False,
):
    model.eval()

    # Create and send train config
    train_config = sy.TrainConfig(
        batch_size=batch_size, model=model, loss_fn=loss_fn_test, optimizer_args=None, epochs=1
    )

    train_config.send(worker)

    result = worker.evaluate(
        dataset_key=dataset_key,
        return_histograms=True,
        nr_bins=nr_bins,
        return_loss=True,
        return_raw_accuracy=True,
        device=device,
    )
    test_loss = result["loss"]
    correct = result["nr_correct_predictions"]
    len_dataset = result["nr_predictions"]
    hist_pred = result["histogram_predictions"]
    hist_target = result["histogram_target"]

    if print_target_hist:
        logger.info("Target histogram: %s", hist_target)


    logger.info(
        "%s: Average loss: %s, Accuracy: %s/%s (%s%%)",
        model_identifier,
        f"{test_loss:.4f}",
        correct,
        len_dataset,
        f"{100.0 * correct / len_dataset:.2f}",
    )

    accuracy = correct / len_dataset

    return test_loss, accuracy


def aggregate(
        model_dict: dict,
        worker: MyWebsocketClientWorker
):
    try:
        federated_model = ConvNet1D(input_size=400, num_classes=7)
        traced_model = torch.jit.trace(federated_model, torch.zeros([1, 400, 3], dtype=torch.float))

        aggregate_config = my_utils.AggregatedConfig(
            model_dict=model_dict,
            federated_model=traced_model
        )

        # 发送模型到聚合点
        start = datetime.now()
        aggregate_config.send_model(worker)
        print(f'model send time: {(datetime.now() - start).total_seconds()}')

        # 发送训练模型的id给client
        aggregate_config_dict = aggregate_config.simplify(aggregate_config.owner, aggregate_config)
        # print(aggregate_config_dict)
        start = datetime.now()
        ptr, ID = aggregate_config._wrap_and_send_obj(aggregate_config_dict, worker)
        print(f'model information send time: {(datetime.now() - start).total_seconds()}')

        # 在远程对变量进行赋值
        worker._send_msg_and_deserialize("set_aggregate_config", ID=ID)

        # 检查是否发送到设备端
        worker._send_msg_and_deserialize("_check_aggregate_config")

        # 发送命令让模型开始聚合
        worker._send_msg_and_deserialize("model_aggregation")

        # 收回模型
        start = datetime.now()
        model = aggregate_config.get(aggregate_config_dict['federated_model_id'], worker).obj
        print(f'model get time: {(datetime.now() - start).total_seconds()}')

        new_model = model_to_device(model, 'cpu')

        return new_model

    except Exception as e:
        print(f"User-{worker.id} Aggregation Failed. Reason: {e}")
        return None


def visualization(input_df:pd.DataFrame, title, y_label, log_path):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    for column in input_df.columns:
        plt.figure(figsize=(5, 5))
        plt.plot(input_df.index, input_df[column], label=column, color='red', linewidth=2, alpha=0.7, marker='o')

        plt.xlabel('Training Round')
        plt.ylabel(y_label)
        plt.title(f'{column}_{title}')
        plt.savefig(f'{log_path}/{column}_accuracy.png')
        # plt.show()


async def main():
    args = define_and_get_arguments()
    stage = 'stage' + str(args.stage)
    dataset_key = 'HAR-' + str(args.stage)

    hook = sy.TorchHook(torch)

    # 连接所有节点
    client_device_mapping_id = my_utils.client_device_mapping_id
    all_nodes = []
    for ip, ID in client_device_mapping_id.items():
        kwargs_websocket = {"hook": hook, "host": ip, "port": 9292, "id": ID}
        all_nodes.append(MyWebsocketClientWorker(**kwargs_websocket))
    worker_instances = all_nodes[:-1]
    worker_states = [True for i in range(len(worker_instances))]
    testing = all_nodes[-1]

    for wcw in all_nodes:
        wcw.clear_objects_remote()

    loss_list = []
    accuracy_list = []
    client_loss_list = []
    client_accuracy_list = []

    # Ubuntu Laptop: use cpu
    device = torch.device('cpu')

    if args.stage == 1:
        model = ConvNet1D(input_size=400, num_classes=7).to(device)
        traced_model = torch.jit.trace(model, torch.zeros([1, 400, 3], dtype=torch.float).to(device))
    else:
        model = ConvNet1D(input_size=400, num_classes=7).to(device)
        model_path = f'model/HAR_stage{str(args.stage-1)}.pt'
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        model.train()
        traced_model = torch.jit.trace(model, torch.zeros([1, 400, 3], dtype=torch.float).to(device))

    learning_rate = args.lr

    global_federated_time_list = []
    federated_time_list = []
    model_aggregation_time_list = []
    train_time_list = []
    test_num = 5

    aggregate_policy = my_utils.AggregationPolicies([i for i in range(len(worker_instances))])
    aggregate_worker_num = aggregate_policy.reset()

    # 开始训练
    for curr_round in range(1, args.training_rounds + 1):
        logger.info("Training round %s/%s", curr_round, args.training_rounds)

        global_federated_start_time = datetime.now()

        results = await asyncio.gather(
            *[
                fit_model_on_worker(
                    worker=worker,
                    traced_model=traced_model,
                    batch_size=args.batch_size,
                    curr_round=curr_round,
                    max_nr_batches=args.federate_after_n_batches,
                    lr=learning_rate,
                    dataset_key=dataset_key,
                    state=state
                )
                for worker, state in zip(worker_instances, worker_states)
            ]
        )

        global_federated_end_time = datetime.now()
        global_federated_time_consuming = (global_federated_end_time - global_federated_start_time).total_seconds()
        global_federated_time_list.append(global_federated_time_consuming)
        print(global_federated_time_consuming)

        models = {}
        loss_values = {}
        federated_time_consuming = {}
        train_time_consuming = {}
        accuracy_dict = {}
        loss_test_values = {}

        test_models = curr_round % test_num == 0 or curr_round == args.training_rounds or curr_round == 1
        # test_models = True
        if test_models:
            logger.info("Evaluating models")
            np.set_printoptions(formatter={"float": "{: .0f}".format})
            for worker_id, worker_model, _, _1, _2 in results:
                if worker_model is not None:
                    loss, accuracy = evaluate_model_on_worker(
                        model_identifier="Model update " + worker_id,
                        worker=testing,
                        dataset_key="HAR-testing",
                        model=model_to_device(worker_model, 'cpu'),
                        nr_bins=7,
                        batch_size=32,
                        device=torch.device('cpu'),
                        print_target_hist=False,
                    )
                    accuracy_dict[worker_id] = accuracy
                    loss_test_values[worker_id] = loss
                else:
                    accuracy_dict[worker_id] = None
                    loss_test_values[worker_id] = None

            client_loss_list.append(loss_test_values)
            client_accuracy_list.append(accuracy_dict)

        # Federate models (note that this will also change the model in models[0]
        for worker_id, worker_model, worker_loss, worker_federated_time, worker_train_time in results:
            if worker_model is not None:
                models[worker_id] = model_to_device(worker_model, 'cpu')
                loss_values[worker_id] = worker_loss
                federated_time_consuming[worker_id] = worker_federated_time
                train_time_consuming[worker_id] = worker_train_time
            else:
                # 删除无效节点
                index = list(my_utils.client_device_mapping_id.values()).index(worker_id)
                worker_states[index] = False

        federated_time_list.append(federated_time_consuming)
        train_time_list.append(train_time_consuming)

        # 模型聚合
        while True:
            # print(aggregate_worker_num)
            aggregate_worker = worker_instances[aggregate_worker_num]
            print(f"Model Aggregation on device: {aggregate_worker.id}")
            model_aggregation_start_time = datetime.now()
            traced_model = aggregate(models, aggregate_worker)
            model_aggregation_end_time = datetime.now()
            if traced_model is not None:
                aggregate_worker_num = aggregate_policy.aggregate_in_order()
                model_aggregation_time_consuming = (model_aggregation_end_time - model_aggregation_start_time).total_seconds()
                print(model_aggregation_time_consuming)
                model_aggregation_time_list.append(model_aggregation_time_consuming)
                break
            else:
                aggregate_policy.delete_inaccessible_worker(aggregate_worker_num)
                aggregate_worker_num = aggregate_policy.aggregate_in_order()

        # traced_model = utils.federated_avg(models)

        if curr_round == args.training_rounds:
            torch.save(traced_model.state_dict(), f"model/HAR_{stage}.pt")

        if test_models:
            loss, accuracy = evaluate_model_on_worker(
                model_identifier="Federated model",
                worker=testing,
                dataset_key="HAR-testing",
                model=model_to_device(traced_model, 'cpu'),
                nr_bins=7,
                batch_size=128,
                device=torch.device('cpu'),
                print_target_hist=True,
            )
            loss_list.append(loss)
            accuracy_list.append(accuracy)

        # decay learning rate
        # learning_rate = max(0.98 * learning_rate, args.lr * 0.01)
        learning_rate = args.lr

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f'log/{stage}_{formatted_time}'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # Visualization
    # loss
    plt.figure(figsize=(5, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot([i * test_num for i in range(len(loss_list))], loss_list, label='Loss', color='red', linewidth=2, alpha=0.7, marker='o')
    # plt.title('Loss Function Curve')
    # plt.xlabel('Training Round')
    # plt.ylabel('Loss')

    # accuracy
    # plt.subplot(1, 2, 2)
    plt.plot([i * test_num for i in range(len(accuracy_list))], accuracy_list, label='Accuracy', color='blue',
             linewidth=2, alpha=0.7, marker='o')
    plt.title('Accuracy Curve')
    plt.xlabel('Training Round')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(f'{log_path}/accuracy.png')
    # plt.show()

    # 保存federated learning时间消耗
    federated_time_df = pd.DataFrame(federated_time_list)
    federated_time_df.to_csv(f'{log_path}/federated_time_consuming.csv')
    # 保存整个的federated learning时间消耗
    global_federated_time_df = pd.DataFrame(global_federated_time_list)
    global_federated_time_df.to_csv(f'{log_path}/global_federated_time_consuming.csv')
    # 保存模型转发的时间
    model_aggregation_time_df = pd.DataFrame(model_aggregation_time_list)
    model_aggregation_time_df.to_csv(f'{log_path}/model_aggregation_time_consuming.csv')
    # 保存单个设备的训练时间
    train_time_df = pd.DataFrame(train_time_list)
    train_time_df.to_csv(f'{log_path}/train_time_consuming.csv')
    # 保存整体准确率
    accuracy_df = pd.DataFrame(accuracy_list, index=[i * test_num for i in range(len(accuracy_list))])
    accuracy_df.to_csv(f'{log_path}/global_accuracy.csv')
    # 保存所有Client的准确率
    raspi_accuracy = pd.DataFrame(client_accuracy_list, index=[i * test_num for i in range(len(accuracy_list))])
    raspi_accuracy.to_csv(f'{log_path}/raspi_accuracy.csv')

    visualization(raspi_accuracy, title='Accuracy Curve', y_label='Accuracy', log_path=log_path + '/raspi_pic')

if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    # Websockets setup
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())

    # Run main
    asyncio.get_event_loop().run_until_complete(main())