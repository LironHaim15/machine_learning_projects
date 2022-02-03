import numpy
import numpy as np
import torch
import sys
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim


# from sklearn.model_selection import KFold


class ModelAB(nn.Module):
    def __init__(self, image_size, ):
        super(ModelAB, self).__init__()
        self.image_size = image_size
        self.fc1 = nn.Linear(image_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)  # TODO check if 'dim=' is needed, check loss results
        return x


class ModelC(nn.Module):
    def __init__(self, image_size, drop1, drop2):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc1 = nn.Linear(image_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.dropout1 = nn.Dropout(drop1)
        self.dropout2 = nn.Dropout(drop2)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)  # TODO check if 'dim=' is needed, check loss results
        return x


class ModelD(nn.Module):
    def __init__(self, image_size):
        super(ModelD, self).__init__()
        self.image_size = image_size
        self.fc1 = nn.Linear(image_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)  # TODO check if 'dim=' is needed, check loss results
        return x


class ModelE(nn.Module):
    def __init__(self, image_size):
        super(ModelE, self).__init__()
        self.image_size = image_size
        self.fc1 = nn.Linear(image_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 10)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(10)
        self.bn4 = nn.BatchNorm1d(10)
        self.bn5 = nn.BatchNorm1d(10)
        self.bn6 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        # x = self.dropout1(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc3(x)
        # x = self.bn3(x)
        x = F.relu(x)
        # x = self.dropout3(x)
        x = self.fc4(x)
        # x = self.bn4(x)
        x = F.relu(x)
        # x = self.dropout4(x)
        x = self.fc5(x)
        # x = self.bn5(x)
        x = F.relu(x)
        # x = self.dropout5(x)
        x = self.fc6(x)
        x = F.log_softmax(x, dim=1)
        return x


class ModelF(nn.Module):
    def __init__(self, image_size):
        super(ModelF, self).__init__()
        self.image_size = image_size
        self.fc1 = nn.Linear(image_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 10)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(10)
        self.bn4 = nn.BatchNorm1d(10)
        self.bn5 = nn.BatchNorm1d(10)
        self.bn6 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.fc1(x)
        # x = self.bn1(x)
        x = torch.sigmoid(x)
        # x = self.dropout1(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = torch.sigmoid(x)
        # x = self.dropout2(x)
        x = self.fc3(x)
        # x = self.bn3(x)
        x = torch.sigmoid(x)
        # x = self.dropout3(x)
        x = self.fc4(x)
        # x = self.bn4(x)
        x = torch.sigmoid(x)
        # x = self.dropout4(x)
        x = self.fc5(x)
        # x = self.bn5(x)
        x = torch.sigmoid(x)
        # x = self.dropout5(x)
        x = self.fc6(x)
        x = F.log_softmax(x, dim=1)
        return x


class ModelMyModel(nn.Module):
    def __init__(self, image_size, drop1=0.1, drop2=0.1, drop3=0.1, drop4=0.1, drop5=0.1):
        super(ModelMyModel, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 10)
        self.fc7 = nn.Linear(8, 10)

        self.bn0 = nn.BatchNorm1d(512)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(16)
        self.bn6 = nn.BatchNorm1d(8)

        self.dropout1 = nn.Dropout(drop1)
        self.dropout2 = nn.Dropout(drop2)
        self.dropout3 = nn.Dropout(drop3)
        self.dropout4 = nn.Dropout(drop4)
        self.dropout5 = nn.Dropout(drop5)

    def forward(self, x):
        x = x.view(-1, self.image_size)

        x = self.fc0(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.fc5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        x = self.fc6(x)
        # x = self.bn6(x)
        # x = F.relu(x)
        # x = self.dropout5(x)
        # x = self.fc7(x)
        x = F.log_softmax(x, dim=1)
        return x


def train(train_loader, model, optimizer):
    model.train()
    training_loss = 0
    training_correct = 0
    for inputs, labels in train_loader:
        # inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs.float())
        loss = F.nll_loss(output, labels.long(), reduction='sum')
        training_loss += loss.item()
        loss.backward()
        optimizer.step()

        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        training_correct += pred.eq(labels.view_as(pred)).cpu().sum()
    size_train = len(train_loader.dataset)
    return training_loss / size_train, training_correct / size_train


def test(test_loader, model):
    model.eval()
    test_loss = 0
    test_correct = 0
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            # data,target = data.to(device), target.to(device)
            output = model(data.float())
            test_loss += F.nll_loss(output, target.long(), reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            predictions.extend(pred.tolist())
            test_correct += pred.eq(target.view_as(pred)).cpu().sum()
    size_test = len(test_loader.dataset)
    test_loss /= size_test
    return test_loss, test_correct, size_test, 100. * test_correct / size_test, predictions


def get_predictions(model, test_x):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_x:
            # data,target = data.to(device), target.to(device)
            output = model(data.float())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            predictions.extend(pred.tolist())
    return predictions


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # HYPER_PARAMETERS
    num_epochs = 10
    batch_size = 128

    train_x, train_y, test_x, test_y_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    train_x = numpy.loadtxt(train_x)
    train_y = numpy.loadtxt(train_y)
    test_x = numpy.loadtxt(test_x)

    train_size = int(len(train_x) * 0.8)
    validation_size = int(len(train_x) * 0.2)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    ten_train_y = torch.from_numpy(train_y)
    ten_test_x = torch.squeeze(trans(test_x))


    """create datasets & normalize"""
    train_all_dataset = data_utils.TensorDataset(torch.squeeze(trans(train_x)), ten_train_y)
    """split the dataset into train & validation"""
    train_loader = data_utils.DataLoader(dataset=train_all_dataset, batch_size=batch_size, shuffle=True)
    """load data from dataset"""


    models = {'A': ModelAB(28 * 28),
              'B': ModelAB(28 * 28),
              'C': ModelC(28 * 28, drop1=0.1, drop2=0.1),
              'D': ModelD(28 * 28),
              'E': ModelE(28 * 28),
              'F': ModelF(28 * 28),
              'MyModel': ModelMyModel(28 * 28, drop1=0.1, drop2=0.1, drop3=0.1, drop4=0.1, drop5=0.1), }

    optimizers = {
        'A': optim.SGD(models['A'].parameters(), lr=0.000001, ),
        'B': optim.Adam(models['B'].parameters(), lr=0.0001, betas=(0.9, 0.999)),
        'C': optim.Adam(models['C'].parameters(), lr=0.0001, betas=(0.9, 0.999)),
        'D': optim.Adam(models['D'].parameters(), lr=0.0001, betas=(0.9, 0.999)),
        'E': optim.Adam(models['E'].parameters(), lr=0.0001, betas=(0.9, 0.999)),
        'F': optim.Adam(models['F'].parameters(), lr=0.001, betas=(0.9, 0.999)),
        # 'MyModel': optim.Adam(models['MyModel'].parameters(), lr=0.00003, betas=(0.9, 0.99)), }
        'MyModel': optim.Adam(models['MyModel'].parameters(), lr=0.0001, betas=(0.9, 0.99)), }

    results = {}
    num_epochs = 50
    predictions = []
    # for char in models.keys():
    for char in ['MyModel']:
        loss_val_dict = {}
        loss_training_dict = {}
        acc_val_dict = {}
        acc_training_dict = {}
        for epoch in range(1, num_epochs + 1):
            loss_training, acc_training = train(train_loader, model=models[char], optimizer=optimizers[char])

    predictions = get_predictions(model=models[char], test_x=ten_test_x)
    """write predictions of test_x to file"""
    pred_file = open(test_y_name, 'w')
    for i in range(len(predictions) - 1):
        pred_file.write(str(predictions[i][0]) + '\n')
    pred_file.write(str(predictions[len(predictions) - 1][0]))
    pred_file.close()
