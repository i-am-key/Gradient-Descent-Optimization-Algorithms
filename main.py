import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (3, 32, 32)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 32, 32)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(
            	kernel_size=2,
            	stride=2,
            	padding=0),    				# choose max value in 2x2 area, output shape (36, 16, 16)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 16, 16)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 16, 16)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2, 2),             # output shape (32, 8, 8)
        )
        self.fc1 = nn.Linear(32 * 8 * 8, 100)   # fully connected layer, output 10 classes
        self.fc2 = nn.Linear(100, 10) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 8 * 8)
        output = F.relu(self.fc1(x))
        output = self.fc2(output)
        return output, x    				# return x for visualization


with open("sampledCIFAR10", "rb") as f:
    data = pickle.load(f)
    train, val, test = data['train'], data['val'], data['test']
idxs = np.arange(len(train["data"]))
train_data_all = train["data"]
train_label_all = train["labels"]
batch_size = 64
training_losses = []
training_accs = []
validation_losses = []
validation_accs = []
test_losses = []
test_accs = []
cnn = Model()
in_channels = 3
# optimize all cnn parameters
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
# optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)   
# optimizer = torch.optim.Adagrad(cnn.parameters(), lr=0.001)
# optimizer = torch.optim.Adadelta(cnn.parameters(), lr=0.001)  # still have problems 
# optimizer = torch.optim.RMSprop(cnn.parameters(), lr=0.001)
# optimizer = torch.optim.Adamax(cnn.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

nepochs = 10
for e in range(nepochs):
    train_loss = []
    train_acc = []
    np.random.shuffle(idxs)
    train_data_all, train_label_all = train_data_all[idxs], train_label_all[idxs]
    for b in range(0, len(train["data"]), batch_size):
        train_data = train_data_all[b:b + batch_size]
        train_original_label = train_label_all[b:b + batch_size]
        if len(train_data) != batch_size:
            train_data = train_data.reshape(len(train_data), in_channels, 32, 32)
        else:
            train_data = train_data.reshape(batch_size, in_channels, 32, 32)
        train_data = torch.from_numpy(train_data).float()
        train_original_label = torch.from_numpy(train_original_label).long()
        train_output = cnn(train_data)[0]     				 # train output
        # train_original_label, train_predict_label: numpy.ndarray
        train_predict_label = torch.max(train_output, 1)[1].data.numpy()
        acc = float((train_predict_label == train_original_label.data.numpy()).astype(int).sum()) / float(train_original_label.size(0))
        train_acc.append(float(acc))
        loss = loss_func(train_output, train_original_label)   # train cross entropy loss
        train_loss.append(float(loss))
        optimizer.zero_grad()           				 # clear gradients for this training step
        loss.backward()                 				 # backpropagation, compute gradients
        optimizer.step()                				 # apply gradients
    print("epoch:", e + 1, "train loss: ", np.mean(train_loss), "train accuracy: ", np.mean(train_acc))
    training_losses.append(np.mean(train_loss))
    training_accs.append(np.mean(train_acc))
    val_loss = []
    val_acc = []
    for b in range(0, len(val["data"]), batch_size):
        val_data = val["data"][b:b + batch_size]
        val_original_label = val["labels"][b:b + batch_size]
        if len(val_data) != batch_size:
            val_data = val_data.reshape(len(val_data), in_channels, 32, 32)
        else:
            val_data = val_data.reshape(batch_size, in_channels, 32, 32)
        val_data = torch.from_numpy(val_data).float()
        val_original_label = torch.from_numpy(val_original_label).long()
        val_output = cnn(val_data)[0]   
        val_predict_label = torch.max(val_output, 1)[1].data.numpy()			  # val output
        acc = float((val_predict_label == val_original_label.data.numpy()).astype(int).sum()) / float(val_original_label.size(0))
        val_acc.append(float(acc))
        loss = loss_func(val_output, val_original_label)   # cross entropy loss
        val_loss.append(float(loss))
    print("epoch:", e + 1, "val loss: ", np.mean(val_loss), "val accuracy: ", np.mean(val_acc))
    validation_losses.append(np.mean(val_loss))
    validation_accs.append(np.mean(val_acc))
test_loss = []
test_acc = []
for b in range(0, len(test["data"]), batch_size):
    test_data = test["data"][b:b + batch_size]
    test_original_label = test["labels"][b:b + batch_size]
    if len(test_data) != batch_size:
        test_data = test_data.reshape(len(test_data), in_channels, 32, 32)
    else:
        test_data = test_data.reshape(batch_size, in_channels, 32, 32)
    test_data = torch.from_numpy(test_data).float()
    test_original_label = torch.from_numpy(test_original_label).long()
    test_output = cnn(test_data)[0]   
    test_predict_label = torch.max(test_output, 1)[1].data.numpy()			  # test output
    acc = float((test_predict_label == test_original_label.data.numpy()).astype(int).sum()) / float(test_original_label.size(0))
    test_acc.append(float(acc))
    loss = loss_func(test_output, test_original_label)   # cross entropy loss
    test_loss.append(float(loss))
test_losses = np.mean(test_loss)
test_accs = np.mean(test_acc)
print("training losses: ", training_losses)
print("validation losses: ", validation_losses)
print("test losses: ", test_losses)
print("training accuracy: ", training_accs)
print("validation accuracy: ", validation_accs)
print("test accuracy: ", test_accs)

        

