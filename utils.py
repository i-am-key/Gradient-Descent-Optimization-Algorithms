import numpy as np
import matplotlib.pyplot as plt


def hot_vector(data_label, k):
    label = []
    for value in data_label:
        p = [0 for _ in range(k)]
        p[int(value)] = 1
        label.append(p)
    label = np.array(label)
    label = label.astype(float)
    return label


def im2col(image, FH, FW, stride, padding):
    # image is a 4d tensor([batchsize, channel, height, width])
    N, C, H, W = image.shape
    H_out = (H + 2 * padding - FH) // stride + 1
    W_out = (W + 2 * padding - FW) // stride + 1
    data = np.pad(image, [(0, 0), (0, 0), (padding, padding), (padding, padding)], mode='constant')
    col = np.zeros((N, C, FH, FW, H_out, W_out))
    for y in range(FH):
        y_max = y + stride * H_out
        for x in range(FW):
            x_max = x + stride * W_out
            col[:, :, y, x, :, :] = data[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)
    return col


def col2im(col, input_shape, FH, FW, stride, padding):
    N, C, H, W = input_shape
    H_out = (H + 2 * padding - FH) // stride + 1
    W_out = (W + 2 * padding - FW) // stride + 1
    col = col.reshape(N, H_out, W_out, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1))
    for y in range(FH):
        y_max = y + stride * H_out
        for x in range(FW):
            x_max = x + stride * W_out
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    return img[:, :, padding:H + padding, padding:W + padding]

def plot_loss(plot, num_epoch, train_losses, val_losses):
    x = [i for i in range(1, num_epoch + 1)]
    plot.plot(x, train_losses, color='blue', label='training loss')
    plot.plot(x, val_losses, color='orange', label='validation loss')
    plot.title('Cross-entropy loss vs Epoch number')
    plot.xlabel('epoch number')
    plot.ylabel('average cross-entropy loss')
    plot.legend()
    plot.show()


def plot_acc(plot, num_epoch, train_accs, val_accs):
    x = [i for i in range(1, num_epoch + 1)]
    plot.plot(x, train_accs, color='blue', label='training accuracy')
    plot.plot(x, val_accs, color='orange', label='validation accuracy')
    plot.title('Classification accuracy vs Epoch number')
    plot.xlabel('epoch number')
    plot.ylabel('mean classification accuracy')
    plot.legend()
    plot.show()


def get_data(fr):
    data = [inst.strip().split(',')for inst in fr.readlines()]
    data_label = [a[-1] for a in data]
    data = [data[i][:(len(data[0])-1)] for i in range(len(data))]
    data = np.array(data)
    data = data.astype(float)
    data_label = np.array(data_label)
    data_label = data_label.astype(float)
    context = {}
    context["data"] = data
    context["labels"] = data_label
    return context
