# Liron Haim 206234635
# Stav Lidor 207299785
import os.path
from matplotlib import pyplot as plt
from gcommand_dataset import GCommandLoader
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F


class MyModel(nn.Module):
    """
    a CNN model. create a convolutional layars by calling the method create_conv_layers,
    and FC layers after them.
    the input image channel should be 1 and the amount of output classifications is 30.
    """
    def __init__(self):
        super(MyModel, self).__init__()
        self.create_conv_layers()
        self.fcLayers = nn.Sequential(
            nn.Linear(7680, 4096),
            # nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(4096, 4096),
            # nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(4096, 30),
        )

    def create_conv_layers(self):
        """
        creating a sequence of convolution & max pooling layers, with BN & ReLU, according to the list of architecture.
        """
        layers = []
        channels = 1
        for layer_type in [64, 'MPL', 128, 'MPL', 256, 256, 'MPL', 512, 512, 'MPL', 512, 512, 'MPL']:  # VGG11
            if layer_type != 'MPL':
                layers.append(nn.Conv2d(channels, layer_type, kernel_size=(3, 3), padding=1))
                layers.append(nn.BatchNorm2d(layer_type))
                layers.append(nn.ReLU())
                channels = layer_type
            else:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        self.convLayers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.convLayers(x)          # conv layers
        x = x.view(x.size(0), -1)       # flattening
        x = self.fcLayers(x)            # fc layers
        x = F.log_softmax(x, dim=1)     # log softmax
        return x


def train(loaded_data, model, optimizer, epoch, cuda):
    """
   train the model
    Args:
        loaded_data: the loaded train dataset
        model: instance of the model
        optimizer: the model's optimizer
        epoch: current epoch (for prints)
        cuda: bool, True if cuda is available
    Returns: train loss and accuracy
    """
    model.train()
    training_epoch_loss = 0
    training_correct = 0
    size_train = len(loaded_data.dataset)
    for i, (data, label) in enumerate(loaded_data):
        loss = nn.CrossEntropyLoss(reduction='sum')
        if cuda:
            data, label = data.cuda(), label.cuda()
            loss = loss.cuda()
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, label.long(), reduction='sum')
        loss = loss(output, label)
        loss.backward()
        optimizer.step()
        training_epoch_loss += loss.item()

        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        training_correct += pred.eq(label.view_as(pred)).cpu().sum()
        # if i == 58 or i == 117 or i == 176 or i == 234:
        #     print('Train Epoch: {} ({}/{} |{:.0f}%|)\tLoss: {:.3f}'.format(
        #         epoch, i * len(data), size_train, 100. * i / len(loaded_data),
        #         loss.data.item()))

    return training_epoch_loss / size_train, training_correct / size_train


def test(loaded_data, model, cuda):
    """
       train the model
        Args:
            loaded_data: the loaded validation/test dataset
            model: instance of the model
            cuda: bool, True if cuda is available
        Returns: test loss and accuracy
        """
    model.eval()
    test_loss = 0
    correct = 0
    size_test = len(loaded_data.dataset)

    with torch.no_grad():
        for data, label in loaded_data:
            loss = nn.CrossEntropyLoss(reduction='sum')
            if cuda:
                data, label = data.cuda(), label.cuda()
                loss = loss.cuda()
            output = model(data)
            test_loss += loss(output, label).data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()

    test_loss /= size_test
    accuracy = float(correct) / size_test
    '''print validation testing for each epoch'''
    # print('\n{} set: Average loss: {:.6f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
    #     "Validation", test_loss, correct, size_test, 100 * accuracy))
    return test_loss, accuracy


def create_plots(model_name, dic_val_loss, dic_training_loss, dic_val_acc, dic_training_acc):
    """
    create two graph for further ana analysis.
    Args:
        model_name:
        dic_val_loss: dictionary of validation loss by epochs
        dic_training_loss: dictionary of train loss by epochs
        dic_val_acc: dictionary of validation accuracy by epochs
        dic_training_acc: dictionary of train accuracy by epochs
    Returns:
    """
    plt.subplot(1, 2, 1)
    plt.text(0.4, 0.4, "", fontsize=50)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Average Loss', fontsize=10)
    plt.title(f"Model {model_name} - Average Loss")

    lists = sorted(dic_val_loss.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y, label='Validation')
    lists = sorted(dic_training_loss.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y, label='Training')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.text(0.4, 0.4, "", fontsize=50)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Average Accuracy', fontsize=10)
    plt.title(f"Model {model_name} - Average Accuracy")

    lists = sorted(dic_val_acc.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two
    plt.plot(x, y, label='Validation')
    lists = sorted(dic_training_acc.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two
    plt.plot(x, y, label='Training')
    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(20, 5.5)
    plt.savefig(f'Model_{model_name}.png')
    # plt.show()
    plt.close()


def get_predictions(model, test_x):
    """
    get the predictions for a test dataset.
    Args:
        model: an instance of the trained model
        test_x: loaded test dataset

    Returns: predicions list

    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_x:
            if cuda:
                data = data.cuda()
            output = model(data.float())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            predictions.extend(pred.tolist())
    return predictions


if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    # print(cuda)
    # torch.manual_seed(1234)
    # if cuda:
        # torch.cuda.manual_seed(1234)
        # print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))

    """create data sets"""
    train_dataset = GCommandLoader('./data/train', test_mode=False)
    # valid_dataset = GCommandLoader('./data/valid', test_mode=False)
    test_dataset = GCommandLoader('./data/test', test_mode=True)

    """load datasets"""
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2,
                                               pin_memory=True, sampler=None)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=2,
    #                                            pin_memory=True, sampler=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2,
                                              pin_memory=True, sampler=None)

    """create model instance & optimizer"""
    model = MyModel()
    if cuda:
        model = torch.nn.DataParallel(model).cuda()
    optimizer = opt.Adam(model.parameters(), lr=0.0004)

    epochs = 25  # number of epochs

    """dictionaries for graphs"""
    # loss_valid_dict = {}
    # loss_training_dict = {}
    # acc_valid_dict = {}
    # acc_training_dict = {}

    """train the model"""
    for e in range(1, epochs+1):
        train_loss, train_accuracy = train(train_loader, model, optimizer, e, cuda)
        # test_loss, test_accuracy = test(valid_loader, model, cuda)

    """save loss and accuracy results in dictionaries and create graphs"""
        # acc_valid_dict.update({e: test_accuracy})
        # acc_training_dict.update({e: train_accuracy})
        # loss_valid_dict.update({e: test_loss})
        # loss_training_dict.update({e: train_loss})
    # create_plots("VGG11", loss_valid_dict, loss_training_dict, acc_valid_dict, acc_training_dict)

    """get predictions of test dataset"""
    predictions = get_predictions(model, test_loader)
    """write predictions of test_x to file"""
    pred_file = open("test_y", 'w')
    file_paths = test_dataset.spects
    for prediction, file_path in zip(predictions, file_paths):
        prediction_class = train_dataset.classes[prediction[0]]
        file_name = os.path.basename(file_path)
        pred_file.write(file_name + ',' + prediction_class + '\n')
    pred_file.close()
