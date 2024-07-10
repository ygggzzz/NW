import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Module
from torch.nn.init import xavier_uniform_
from torch.nn import MSELoss
from torch.optim import SGD
from sklearn.metrics import mean_squared_error
from numpy import vstack
import matplotlib.pyplot as plt

np.random.seed(0)
#数据集
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = np.sin(2 * np.pi * X)

class myDataset(Dataset):

    def __init__(self):
        #初始化为张量
        self.X = torch.tensor(X,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    #获得某一行的输入输出数据的list
    def __getitem__(self, index):
        return [self.X[index],self.y[index]]

    #划分训练集测试集
    def get_splits(self,n_test=0.2):
        #round四舍五入
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        #返回训练集测试集
        return random_split(self,[train_size,test_size])

# 为训练集和测试集创建 DataLoader
def prepare_data(myDataset):
    train,test = myDataset.get_splits()
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl,test_dl

class Mlp(Module):
    def __init__(self,inputs,outputs):
        super(Mlp,self).__init__()

        # 隐藏层
        self.hidden1 = Linear(inputs,50)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()

        self.hidden2 = Linear(50,100)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()

        self.hidden3 = Linear(100, 50)
        xavier_uniform_(self.hidden2.weight)
        self.act3 = ReLU()

        self.hidden4 = Linear(50,outputs)
        xavier_uniform_(self.hidden3.weight)

    def forward(self,X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X = self.act3(X)
        X = self.hidden4(X)
        return X

def train(train_dl,model,epochs=1000):
    #损失函数
    criterion = MSELoss()
    #优化器 SGD:随机梯度下降 parameters：要优化的模型参数 lr:learningrate momentnum：动量
    optimizer = SGD(model.parameters(), lr = 0.15, momentum=0.9)

    for epoch in range(epochs):
        #枚举minibatches
        for i,(inputs,targets) in enumerate(train_dl):
            #梯度清除
            optimizer.zero_grad()
            #计算模型输出
            y_hat = model(inputs)
            #计算损失
            loss = criterion(y_hat, targets)
            #反向
            loss.backward()
            #更新
            optimizer.step()

#评估模型
def evaluate(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        y_hat = model(inputs)
        #detach分离张量,获得张量的值,不会计算此张量的梯度
        y_hat = y_hat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        #加到预测列表和实际列表中
        predictions.append(y_hat)
        actuals.append(actual)
    #vstack 垂直方向堆叠原数组数据
    predictions, actuals = vstack(predictions), vstack(actuals)
    # 计算均方误差
    mse = mean_squared_error(actuals, predictions)
    return mse

dataset = myDataset()
model = Mlp(1,1)
train_dl, test_dl = prepare_data(dataset)
train(train_dl, model, epochs=1000)
mse = evaluate(test_dl, model)
print(f'mse',mse)

#输入原数据进行预测
predictions = model(torch.tensor(X, dtype=torch.float32))
predictions = predictions.detach().numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue')
plt.plot(X, predictions, color='red', linewidth=2)
plt.show()
