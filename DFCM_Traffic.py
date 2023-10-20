import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

from sklearn import preprocessing
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras.models import load_model

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# 修改此部分为你的配置
configure = {
    'SourceFilePath': './5-Traffic/1-temp.csv',
    'InputFilePath': './5-Traffic/2plus-supervisedDataSet_zscore.csv',
    'OutputFilePath': './5-Traffic/6-DFCM.csv',
    'PltFilePath': './5-Traffic/6-DFCM/',
    'AllAttributes': 7,
    'TargetAttributes': 6,
    'InputAttributes': [1, 2, 3, 4, 5, 6],
    'OutputAttributes': [8, 9, 10, 11, 12, 13],
    'TimeAttributes': [0],
    'Length': 3602,
    'global_epochs': 100,
    'f_batch_size': 900,
    'f_epochs': 20,
    'hidden_layer': 12,
    'n_batch_size': 900,
    'n_epochs': 20,
    'LSTM_hiddenDim': 15
}

# 函数 - sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False

mkdir(configure['PltFilePath'])

# 加载数据集
dataset = pd.read_csv(configure['InputFilePath'])

# 构造训练集（70%）和测试集（30%）
values = dataset.values
n_train = int(0.7 * configure['Length'])

train = values[:n_train, :]
test = values[n_train:, :]

train_X ,train_Y = sigmoid( train[:,configure['InputAttributes']] ) , train[:,configure['OutputAttributes']]
test_X , test_Y = sigmoid( test[:,configure['InputAttributes']] ), test[:,configure['OutputAttributes']]


train_U = train[:, configure['TimeAttributes']]
train_U = train_U.reshape((train_U.shape[0], 1, train_U.shape[1]))
test_U = test[:, configure['TimeAttributes']]
test_U = test_U.reshape((test_U.shape[0], 1, test_U.shape[1]))

print('Train dataset length: ' + str(len(train)) + '.')
print('Test dataset length : ' + str(len(test)) + '.')
print('------')
print('X dim: ' + str(train_X.shape[1]) + '.')
print('Y dim: ' + str(train_Y.shape[1]) + '.')
print('------')
print('train_X shape: ' + str(train_X.shape))
print('train_Y shape: ' + str(train_Y.shape))
print('train_U shape: ' + str(train_U.shape))
print('------')
print('test_X shape: ' + str(test_X.shape))
print('test_Y shape: ' + str(test_Y.shape))
print('test_U shape: ' + str(test_U.shape))
def calculate_accuracy(y_pred, y_true, threshold=0.5):
    """
    Calculate accuracy for regression predictions by setting a threshold.
    If the difference between y_pred and y_true is less than the threshold, it's considered correct.
    """
    correct_predictions = np.sum(np.abs(y_pred - y_true) < threshold)
    total_predictions = len(y_pred)
    accuracy = correct_predictions / total_predictions
    return accuracy
# 1. 定义边索引
def create_edge_index(n):
    # 对于时间序列数据，我们将每个数据点与其前后的数据点连接起来
    edge_index = []
    for i in range(n - 1):
        edge_index.append((i, i + 1))
        edge_index.append((i + 1, i))
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# 2. 创建特征矩阵
x_train = torch.tensor(train[:, configure['InputAttributes']], dtype=torch.float)
x_test = torch.tensor(test[:, configure['InputAttributes']], dtype=torch.float)

# 3. 创建PyTorch Geometric数据对象
edge_index_train = create_edge_index(len(train))
edge_index_test = create_edge_index(len(test))

data_train = Data(x=x_train, edge_index=edge_index_train)
data_test = Data(x=x_test, edge_index=edge_index_test)

class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCNModel, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, input_dim)  # 输入层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, train_Y.shape[1])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.input_layer(x)  # 通过输入层
        x = F.relu(x)  # 添加激活函数
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 创建GCN模型
model_f = [GCNModel(input_dim=len(configure['InputAttributes']), hidden_dim=configure['hidden_layer']) for _ in range(configure['TargetAttributes'])]
optimizers = [torch.optim.Adam(model.parameters(), lr=0.01) for model in model_f]
train_Y_tensor = torch.tensor(train_Y, dtype=torch.float32)

model_u = [0 for i in range(configure['TargetAttributes'])]
for i in range(configure['TargetAttributes']):
    model_u[i] = Sequential()
    model_u[i].add(LSTM(configure['LSTM_hiddenDim'],  input_shape=(train_U.shape[1], train_U.shape[2])))
    model_u[i].add(Dense(1, input_dim=configure['LSTM_hiddenDim'], use_bias=True))
    model_u[i].compile(loss='mean_squared_error', optimizer='adam')
losses = []

for i in range(configure['global_epochs']):
    start = time.time()
    if i == 0:
        y_f = train_Y
    else:
        y_f = train_Y - y_u_predict


    y_f_predict = np.zeros_like(train_Y)
    for j in range(configure['TargetAttributes']):
        current_model = model_f[j]
        optimizer = optimizers[j]

        current_model.train()
        optimizer.zero_grad()
        out = current_model(data_train)

        total_loss = 0
    for j in range(configure['TargetAttributes']):
        current_model = model_f[j]
        optimizer = optimizers[j]

        current_model.train()
        optimizer.zero_grad()
        out = current_model(data_train)
        # 为每个模型选择相应的目标列
        target = train_Y_tensor[:, j].unsqueeze(1)

        loss = F.mse_loss(out[:, j], target.squeeze())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        average_loss = total_loss / configure['TargetAttributes']  # 计算平均损失
        losses.append(average_loss)

        # 使用GCN模型进行预测
        current_model.eval()
        with torch.no_grad():
            node_representations = current_model(data_train)
        y_f_predict[:, j] = node_representations[:, j]

    y_u = train_Y - y_f_predict

    for j in range(configure['TargetAttributes']):
        model_u[j].fit(train_U, y_u[:, j], configure['n_batch_size'], configure['n_epochs'], verbose=0)

    y_u_predict = DataFrame()
    for j in range(configure['TargetAttributes']):
        y_u_predict[str(j)] = model_u[j].predict(train_U).reshape(-1)
    y_u_predict = y_u_predict.values

    # evaluate
    yhat_f_predict = DataFrame()
    for j in range(configure['TargetAttributes']):
        yhat_f_predict[str(j)] = model_f[j](data_test).detach().numpy()[:, j]
    yhat_f_predict = yhat_f_predict.values

    yhat_u_predict = DataFrame()
    for j in range(configure['TargetAttributes']):
        yhat_u_predict[str(j)] = model_u[j].predict(test_U).reshape(-1)
    yhat_u_predict = yhat_u_predict.values


    predict_train = y_u_predict + y_f_predict
    predict_test = yhat_u_predict + yhat_f_predict
    real_train = train_Y
    real_test = test_Y
    train_accuracy = calculate_accuracy(predict_train, real_train)
    test_accuracy = calculate_accuracy(predict_test, real_test)
    print(f"Epoch {i+1} - Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")

    error_train = pow(abs(real_train - predict_train), 2)
    error_test = pow(abs(real_test - predict_test), 2)
    print(i + 1, error_train.mean().mean(), error_test.mean().mean())

    if (error_test.mean().mean() < 0.05):
        break

# 预测 & 输出

yhat_f_predict = DataFrame()
for j in range(configure['TargetAttributes']):
    yhat_f_predict[str(j)] = model_f[j](data_test).detach().numpy()[:, j]
yhat_f_predict = yhat_f_predict.values

yhat_u_predict = DataFrame()
for j in range(configure['TargetAttributes']):
    yhat_u_predict[str(j)] = model_u[j].predict(test_U).reshape(-1)
yhat_u_predict = yhat_u_predict.values

yhat = yhat_u_predict + yhat_f_predict
DataFrame(yhat).to_csv(configure['OutputFilePath'], index=False)

# 保存模型
for j in range(len(configure['OutputAttributes'])):
    torch.save(model_f[j].state_dict(), configure['PltFilePath'] + 'model_f_' + str(j + 1) + '.pt')
    model_u[j].save(configure['PltFilePath'] + 'model_u_' + str(j + 1) + '.h5')

# 数据概览 - 1
values = yhat
original = test_Y

# 指定要绘制的列
groups = list(range(configure['TargetAttributes']))
i = 1

# 绘制每一列
plt.figure(figsize=(15, 15))
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(original[:, group])
    plt.plot(values[:, group])
    i += 1
plt.savefig(configure['PltFilePath'] + 'performance.png')
plt.show()



values = yhat_f_predict
original = test_Y

# 指定要绘制的列
groups = list(range(configure['TargetAttributes']))
i = 1

# 绘制每一列
plt.figure(figsize=(15, 15))
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(original[:, group])
    plt.plot(values[:, group])
    i += 1
plt.savefig(configure['PltFilePath'] + 'GCN_performance.png')
plt.show()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()