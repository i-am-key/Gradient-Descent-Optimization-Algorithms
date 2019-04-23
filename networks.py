import numpy as np
from layers import *
from utils import *


def convNet(train, val, test, nepochs, batch_size, in_channels, out_channels, 
            conv_kernel_size, lr, conv_stride, padding, pool_kernel_size, pool_stride, momentum=0):
    training_losses = []
    training_accs = []
    validation_losses = []
    validation_accs = []
    h1 = 1 + int((32 - conv_kernel_size + 2 * padding) / conv_stride)
    h2 = int(1 + (h1 - pool_kernel_size) / pool_stride)
    idxs = np.arange(len(train["data"]))
    conv = Conv2d(in_channels, out_channels, conv_kernel_size, lr, conv_stride, padding)
    pool = MaxPool2d(pool_kernel_size, pool_stride, 0)
    fc1 = FullyConnected(h2 * h2 * out_channels, 100, lr)
    fc2 = FullyConnected(100, 10, lr)
    loss_func = SoftmaxCrossEntropy()
    train_data_all = train["data"]
    train_label_all = train["labels"]
    t = 1
    for e in range(nepochs):
        train_loss = []
        train_acc = []
        # Shuffle
        np.random.shuffle(idxs)
        train_data_all, train_label_all = train_data_all[idxs], train_label_all[idxs]
        for b in range(0, len(train["data"]), batch_size):
            train_data = train_data_all[b:b + batch_size]
            train_original_label = train_label_all[b:b + batch_size]
            if len(train_data) != batch_size:
                train_data = train_data.reshape(len(train_data), in_channels, 32, 32)
            else:
                train_data = train_data.reshape(batch_size, in_channels, 32, 32)
            train_label = hot_vector(train_original_label, 10)
            conv.zero_grads()
            conv_out = conv.forward(train_data)
            relu1 = ReLU()
            relu1_out = relu1.forward(conv_out)
            pool_out = pool.forward(relu1_out)
            original_shape = pool_out.shape
            pool_out = pool_out.reshape(pool_out.shape[0], -1)
            fc1.zero_grads()
            fc1_out = fc1.forward(pool_out)
            relu2 = ReLU()
            relu2_out = relu2.forward(fc1_out)
            fc2.zero_grads()
            fc2_out = fc2.forward(relu2_out)
            loss = loss_func.forward(fc2_out, train_label).mean()
            train_predict_label = np.argmax(fc2_out, 1)
            acc = float((train_predict_label == train_original_label).astype(int).sum()) / float(train_original_label.shape[0])
            train_loss.append(loss)
            train_acc.append(acc)
            dout = loss_func.backward()
            d_fc2_out = fc2.backward(dout)
            d_relu2_out = relu2.derivative() * d_fc2_out
            d_fc1_out = fc1.backward(d_relu2_out)
            d_fc1_out  = d_fc1_out.reshape(*original_shape)
            d_pool_out = pool.backward(d_fc1_out)
            d_relu1_out = relu1.derivative() * d_pool_out
            conv.backward(d_relu1_out)
            # change update method
            # ##### (1) vanilla sgd, sgd with momentum or sgd with Nesterov momentum #####
            # fc1.SGD()
            # fc2.SGD()
            # conv.SGD()
            # ##### (2) adam #####
            fc1.adam(t)
            fc2.adam(t)
            conv.adam(t)
            # ##### (3) adagrad #####
            # fc1.adagrad()
            # fc2.adagrad()
            # conv.adagrad()
            # ##### (4) RMSprop #####
            # fc1.RMSprop()
            # fc2.RMSprop()
            # conv.RMSprop()
            t += 1

        print("epoch:", e + 1, "train loss: ", np.mean(train_loss), "train acc: ", np.mean(train_acc))
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
            val_label = hot_vector(val_original_label, 10)
            val_conv_out = conv.forward(val_data)
            val_relu1 = ReLU()
            val_relu1_out = val_relu1.forward(val_conv_out)
            val_pool_out = pool.forward(val_relu1_out)
            val_original_shape = val_pool_out.shape
            val_pool_out = val_pool_out.reshape(val_pool_out.shape[0], -1)
            val_fc1_out = fc1.forward(val_pool_out)
            val_relu2 = ReLU()
            val_relu2_out = val_relu2.forward(val_fc1_out)
            val_fc2_out = fc2.forward(val_relu2_out)
            loss = loss_func.forward(val_fc2_out, val_label).mean()
            val_predict_label = np.argmax(val_fc2_out, 1)              # val output
            acc = float((val_predict_label == val_original_label).astype(int).sum()) / float(val_original_label.shape[0])
            val_loss.append(loss)
            val_acc.append(acc)

        print("epoch:", e + 1, "val loss: ", np.mean(val_loss), "val acc: ", np.mean(val_acc))
        validation_losses.append(np.mean(val_loss))
        validation_accs.append(np.mean(val_acc))

    test_loss = []
    test_acc = []
    weight_matrix = conv.weight_matrix()
    for b in range(0, len(test["data"]), batch_size):
        test_data = test["data"][b:b + batch_size]
        test_original_label = test["labels"][b:b + batch_size]
        if len(test_data) != batch_size:
            test_data = test_data.reshape(len(test_data), in_channels, 32, 32)
        else:
            test_data = test_data.reshape(batch_size, in_channels, 32, 32)
        test_label = hot_vector(test_original_label, 10)
        test_conv_out = conv.forward(test_data)
        test_relu1 = ReLU()
        test_relu1_out = test_relu1.forward(test_conv_out)
        test_pool_out = pool.forward(test_relu1_out)
        test_original_shape = test_pool_out.shape
        test_pool_out = test_pool_out.reshape(test_pool_out.shape[0], -1)
        test_fc1_out = fc1.forward(test_pool_out)
        test_relu2 = ReLU()
        test_relu2_out = test_relu2.forward(test_fc1_out)
        test_fc2_out = fc2.forward(test_relu2_out)
        loss = loss_func.forward(test_fc2_out, test_label).mean()
        test_predict_label = np.argmax(test_fc2_out, 1)              # val output
        acc = float((test_predict_label == test_original_label).astype(int).sum())  / float(test_original_label.shape[0])
        test_loss.append(loss)
        test_acc.append(acc)
    test_losses = np.mean(test_loss)
    test_accs = np.mean(test_acc)
    return training_losses, training_accs, validation_losses, validation_accs, \
           test_losses, test_accs, e + 1, weight_matrix

def neuralNet(train, val, test, nepochs, batch_size, input_size, output_size, hiddens, lr):
    fc1 = FullyConnected(input_size, hiddens[0], lr)
    fc2 = FullyConnected(hiddens[0], hiddens[1], lr)
    fc3 = FullyConnected(hiddens[1], output_size, lr)
    loss_func = SoftmaxCrossEntropy()
    train_data_all = train["data"]
    train_label_all = train["labels"]
    idxs = np.arange(len(train["data"]))
    training_losses = []
    training_accs = []
    validation_losses = []
    validation_accs = []
    t = 1
    for e in range(nepochs):
        train_loss = []
        train_acc = []
        # Shuffle
        np.random.shuffle(idxs)
        train_data_all, train_label_all = train_data_all[idxs], train_label_all[idxs]
        for b in range(0, len(train["data"]), batch_size):
            train_data = train_data_all[b:b + batch_size]
            train_original_label = train_label_all[b:b + batch_size]
            train_label = hot_vector(train_original_label, output_size)
            fc1.zero_grads()
            fc2.zero_grads()
            fc3.zero_grads()
            fc1_out = fc1.forward(train_data)
            relu1 = Sigmoid()
            relu1_out = relu1.forward(fc1_out)
            fc2_out = fc2.forward(relu1_out)
            relu2 = Sigmoid()
            relu2_out = relu2.forward(fc2_out)
            fc3_out = fc3.forward(relu2_out)
            loss = loss_func.forward(fc3_out, train_label).mean()
            train_predict_label = np.argmax(fc3_out, 1)
            acc = float((train_predict_label == train_original_label).astype(int).sum()) / float(train_original_label.shape[0])
            train_loss.append(loss)
            train_acc.append(acc)
            dout = loss_func.backward()
            d_fc3_out = fc3.backward(dout)
            d_relu2_out = relu2.derivative() * d_fc3_out
            d_fc2_out = fc2.backward(d_relu2_out) 
            d_relu1_out = relu1.derivative() * d_fc2_out
            d_fc1_out = fc1.backward(d_relu1_out)
            # change update method
            # ##### (1) vanilla sgd, sgd with momentum or sgd with Nesterov momentum #####
            # fc1.SGD()
            # fc2.SGD()
            # fc3.SGD()
            # ##### (2) adam #####
            # fc1.adam(t)
            # fc2.adam(t)
            # fc3.adam(t)
            # ##### (3) adagrad #####
            # fc1.adagrad()
            # fc2.adagrad()
            # fc3.adagrad()
            # ##### (4) RMSprop #####
            fc1.RMSprop()
            fc2.RMSprop()
            fc3.RMSprop()
            t += 1


        print("epoch:", e + 1, "train loss: ", np.mean(train_loss), "train acc: ", np.mean(train_acc))
        training_losses.append(np.mean(train_loss))
        training_accs.append(np.mean(train_acc))

        val_loss = []
        val_acc = []
        for b in range(0, len(val["data"]), batch_size):
            val_data = val["data"][b:b + batch_size]
            val_original_label = val["labels"][b:b + batch_size]
            val_label = hot_vector(val_original_label, output_size)
            val_fc1_out = fc1.forward(val_data)
            val_relu1 = Sigmoid()
            val_relu1_out = val_relu1.forward(val_fc1_out)
            val_fc2_out = fc2.forward(val_relu1_out)
            val_relu2 = Sigmoid()
            val_relu2_out = val_relu2.forward(val_fc2_out)
            val_fc3_out = fc3.forward(val_relu2_out)
            loss = loss_func.forward(val_fc3_out, val_label).mean()
            val_predict_label = np.argmax(val_fc3_out, 1) 
            acc = float((val_predict_label == val_original_label).astype(int).sum()) / float(val_original_label.shape[0])
            val_loss.append(loss)
            val_acc.append(acc)

        print("epoch:", e + 1, "val loss: ", np.mean(val_loss), "val acc: ", np.mean(val_acc))
        validation_losses.append(np.mean(val_loss))
        validation_accs.append(np.mean(val_acc))

    test_loss = []
    test_acc = []
    for b in range(0, len(test["data"]), batch_size):
        test_data = test["data"][b:b + batch_size]
        test_original_label = test["labels"][b:b + batch_size]
        test_label = hot_vector(test_original_label, output_size)
        test_fc1_out = fc1.forward(test_data)
        test_relu1 = Sigmoid()
        test_relu1_out = test_relu1.forward(test_fc1_out)
        test_fc2_out = fc2.forward(test_relu1_out)
        test_relu2 = Sigmoid()
        test_relu2_out = test_relu2.forward(test_fc2_out)
        test_fc3_out = fc3.forward(test_relu2_out)
        loss = loss_func.forward(test_fc3_out, test_label).mean()
        test_predict_label = np.argmax(test_fc3_out, 1)              
        acc = float((test_predict_label == test_original_label).astype(int).sum())  / float(test_original_label.shape[0])
        test_loss.append(loss)
        test_acc.append(acc)
    test_losses = np.mean(test_loss)
    test_accs = np.mean(test_acc)
    return training_losses, training_accs, validation_losses, validation_accs, \
           test_losses, test_accs, e + 1