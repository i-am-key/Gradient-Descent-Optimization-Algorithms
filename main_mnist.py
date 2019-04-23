from networks import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_fr = open('train.txt')
    val_fr = open('val.txt')
    test_fr = open('test.txt')
    train = get_data(train_fr)
    val = get_data(val_fr)
    test = get_data(test_fr)
    [training_losses, training_accs, validation_losses, validation_accs, test_losses, test_accs, epoch] = \
        neuralNet(train, val, test, nepochs=10, batch_size=32, input_size=1568, output_size=19, hiddens=[200,100], lr=0.1)
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

