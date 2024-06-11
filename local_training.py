import torch
import my_utils
import os
import pickle
from torch.utils.data import TensorDataset, ConcatDataset
from torch.utils.data import DataLoader
import argparse
import torch.optim as optim

import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
import pandas as pd

# 定义argparse
parser = argparse.ArgumentParser(description="Local Training")
parser.add_argument("-ne","--num_epochs", type=int, default=5, help="the number of training epochs")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("-tn", "--training_num", type=int, default=2, help="the number of training users")
parser.add_argument("-d", "--device", type=str, default="nano", help="raspi or nano is Okay")
args = parser.parse_args()

# 定义logger，记录实验过程
logger_path = os.path.join('log')
logger, path = my_utils.set_logger(
    save_path=logger_path,
    action="local_training"
)

logger.info(f"The number of epochs: {args.num_epochs}\nThe learning rate: {args.learning_rate}\nThe number of training users: {args.training_num}")
logger.info("Local Training Preparing")

# 读取数据集
data_path = '../Dataset/HAR/shuffled_HAR_datasets.pkl'
with open(data_path, 'rb') as f:
    har_datasets = pickle.load(f)

train_dataset = ConcatDataset([har_datasets[i+1] for i in range(args.training_num)])
test_dataset = ConcatDataset([har_datasets[24-i] for i in range(4)])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
total_batches = len(train_loader)

# 测试是否可以使用cuda
device = "cuda" if torch.cuda.is_available() else "cpu"

# 模型的建立
model = my_utils.ConvNet1D(input_size=400, num_classes=7).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# 保存所需要的数据；训练的时间、温度
consuming_time_list = []
temperature_list = []

# 模型训练
logger.info("Training Start")
for epoch in range(1, args.num_epochs+1):
    # 温度测试
    if args.device == "raspi":
        temperature = my_utils.read_raspi_cpu_temperature()
        logger.info(f"CPU temperature: {temperature}°C")
    elif args.device == "nano":
        temperature = my_utils.read_nano_gpu_temperature()
        logger.info(f"GPU temperature: {temperature}°C")
    else:
        temperature = None
    temperature_list.append(temperature)

    model.train()
    batch_index = 0
    for data, targets in train_loader:
        batch_index += 1
        # 进行时间测试
        start = time.perf_counter()

        optimizer.zero_grad()
        output = model(data.to(device))
        loss = my_utils.loss_fn(target=targets.to(device), pred=output)
        loss.backward()
        optimizer.step()

        end = time.perf_counter()
        consuming_time = end-start
        if batch_index == total_batches:
            continue
        consuming_time_list.append(consuming_time)

    logger.info(f"Training Epoch {epoch}/{args.num_epochs}: {loss.item()} ")

    # 测试模型
    if epoch % 5 == 0 or epoch == args.num_epochs:
        model.eval()

        test_correct = 0
        test_total = 0
        for inputs, targets in test_loader:
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs, 1)
            _, true_class = torch.max(targets, 1)
            true_class = true_class.to(device)

            test_total += targets.size(0)
            test_correct += (predicted == true_class).sum().item()

        logger.info(f"Accuracy {100*test_correct/test_total:.2f}%")

# 将时间保存
consuming_time_array = np.array(consuming_time_list).reshape([args.num_epochs, total_batches - 1])
logger.info(f"The number of epochs: {args.num_epochs}\nThe learning rate: {args.learning_rate}\nThe number of training users:{args.training_num}")
logger.info(f"Average Consuming Time for 1 Batch: {np.mean(consuming_time_array)}s")
consuming_time_df = pd.DataFrame(consuming_time_array)
consuming_time_df.to_csv(os.path.join(path, 'consuming_time.csv'), index=False, header=False)

# 将温度保存
temperature_df = pd.DataFrame(temperature_list, columns=['temperature'])
temperature_df.to_csv(os.path.join(path, 'temperature.csv'), index=False)

# 绘制直方图
my_utils.plot_probability_histogram(data=consuming_time_list[10:],
                                    save_path=os.path.join(path, 'consuming_time.png'),
                                    title="Consuming Time Probability Histogram")
my_utils.plot_line_chart(data=temperature_list,
                         save_path=os.path.join(path, 'temperature.png'))

# 保存训练的模型
save_path = os.path.join("model", "local_training.pt")
torch.save(model.state_dict(), save_path)
