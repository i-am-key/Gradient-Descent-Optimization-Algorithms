from networks import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_fr = open('train.txt')

    val_fr = open('val.txt')
    test_fr = open('test.txt')
    train = get_data(train_fr)
    val = get_data(val_fr)
    test = get_data(test_fr)
    seeds = [5, 10, 15, 20, 25]
    sum_training_losses = []
    sum_training_accs = []
    sum_validation_losses = []
    sum_validation_accs = []
    sum_test_losses = []
    sum_test_accs = []
    for i in range(len(seeds)):
        np.random.seed(seeds[i])
        [training_losses, training_accs, validation_losses, validation_accs, test_losses, test_accs, epoch] = \
        neuralNet(train, val, test, nepochs=26, batch_size=32, input_size=1568, output_size=19, hiddens=[200,100], lr=0.01, apply_early_stop=False, momentum=0)
        sum_training_losses.append(training_losses)
        sum_training_accs.append(training_accs)
        sum_validation_losses.append(validation_losses)
        sum_validation_accs.append(validation_accs)
        sum_test_losses.append(test_losses)
        sum_test_accs.append(test_accs)
    mean_training_losses = np.mean(sum_training_losses, axis=0)
    mean_training_accs = np.mean(sum_training_accs, axis=0)
    mean_validation_losses = np.mean(sum_validation_losses, axis=0)
    mean_validation_accs = np.mean(sum_validation_accs, axis=0)
    mean_test_losses = np.mean(sum_validation_losses, axis=0)
    mean_test_accs = np.mean(sum_validation_accs, axis=0)
    print("training losses: ", mean_training_losses)
    print("validation losses: ", mean_validation_losses)
    print("test losses: ", mean_test_losses)
    print("training accs: ", mean_training_accs)
    print("validation accs: ", mean_validation_accs)
    print("test accs: ", mean_test_accs)
    print("epoch: ", epoch)

    # [training_losses, training_accs, validation_losses, validation_accs, test_losses, test_accs, epoch] = \
    #     neuralNet(train, val, test, nepochs=200, batch_size=32, input_size=1568, output_size=19, hiddens=[200,100], lr=0.01, apply_early_stop=True, momentum=0)
    # print("training losses: ", training_losses)
    # print("validation losses: ", validation_losses)
    # print("test losses: ", test_losses)
    # print("training accs: ", training_accs)
    # print("validation accs: ", validation_accs)
    # print("test accs: ", test_accs)
    # print("epoch: ", epoch)
    # plt.figure(1)
    # plot_loss(plt, epoch, training_losses, validation_losses)
    # plt.figure(2)
    # plot_acc(plt, epoch, training_accs, validation_accs)

