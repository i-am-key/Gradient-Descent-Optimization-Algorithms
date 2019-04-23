import pickle
from networks import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open("sampledCIFAR10", "rb") as f:
        data = pickle.load(f)
        train, val, test = data['train'], data['val'], data['test']
    # ########### 36 5*5 CNN filter -> ReLU activation -> 2*2 max pooling layer -> fully connected ReLU ###########
    [training_losses, training_accs, validation_losses, validation_accs, test_losses, test_accs, epoch, w] = \
        convNet(train, val, test, nepochs=200, batch_size=64, in_channels=3, out_channels=36, conv_kernel_size=5,
                lr=0.001, conv_stride=1, padding=2, pool_kernel_size=2, pool_stride=2, apply_early_stop=False, momentum=0)
    print("training losses: ", training_losses)
    print("validation losses: ", validation_losses)
    print("test losses: ", test_losses)
    print("training accs: ", training_accs)
    print("validation accs: ", validation_accs)
    print("test accs: ", test_accs)
    print("epoch: ", epoch)
    plt.figure(1)
    plot_loss(plt, epoch, training_losses, validation_losses)
    plt.figure(2)
    plot_acc(plt, epoch, training_accs, validation_accs)

    