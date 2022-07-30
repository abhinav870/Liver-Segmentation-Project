from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm
from preprocess import prepare

def dice_metric(predicted, target):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coefficient then we use it
    to calculate a metric value for the training and the validation/testing.
    '''
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value

# eg prob of bg = 0.9 & prob of fg = 0.1
# now when we calculate error (weighted cross entropy) of bg, we multiply error by 0.1 . Soo error will be small
# So in Back prop, when the model tries to update the weights, there won't be a drastic change in the weights

# when we calculate error (weighted cross entropy) of fg, we multiply error by 0.9 . Soo error will be big
# So in Back prop, when the model tries to update the weights, weights would be modified significantly
def calculate_weights(val1, val2):
    '''
    In this function we take the number of the background and the foreground pixels to return the `weights`
    for the cross entropy loss values.
    '''
    count = np.array([val1, val2])
    summ = count.sum()
    weights = count / summ
    weights = 1 / weights
    summ = weights.sum()
    weights = weights / summ
    return torch.tensor(weights, dtype=torch.float32)

"""

def train()

model = U-Net Model Architecture for training 
data_in = directory where data is stored
loss = loss_fn to be used
optim = optimizer
max_epochs = max no of epochs
model_dir = directory to save the trained model

test_intervals = controls in which interval/epoch do we want to save the weights
So test_interval = 1 means in each 2 epochs they calculate new dice for all the testing data.
When we get higher dice value, save the model. Doing this speeds up the training a little bit  

"""

def train(model, data_in, loss, optim, max_epochs, model_dir, test_interval=2, device=torch.device("cuda:0")):
    best_metric = -1
    best_metric_epoch = -1

    save_loss_train = []
    save_loss_test = []

    save_metric_train = []
    save_metric_test = []

    """
    We will get 4 graphs:
    (a) loss_training
    (b) metric_training
    (c) loss_testing
    (b) metric_testing  
    """

    train_loader, test_loader = data_in

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0
        for batch_data in train_loader:
            train_step += 1


            # get the key (volume) & value (label)
            volume = batch_data["vol"]
            label = batch_data["seg"]
            label = label != 0
            volume, label = (volume.to(device), label.to(device))

            optim.zero_grad()
            # for every mini-batch during the training phase,set the gradients to zero before backpropagation
            outputs = model(volume)

            train_loss = loss(outputs, label)

            train_loss.backward()# calculates the training loss
            optim.step()
            # optimizers iterate over all parameters (tensors) it is supposed to update and use their internally stored
            # gradients to update their values.

            train_epoch_loss += train_loss.item()
            print(
                f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
                f"Train_loss: {train_loss.item():.4f}")

            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric
            print(f'Train_dice: {train_metric:.4f}')

        print('-' * 20)

        train_epoch_loss /= train_step
        print(f'Epoch_loss: {train_epoch_loss:.4f}')
        save_loss_train.append(train_epoch_loss)

        # Saves new value in each epoch. It creates a list containing all the values for each epoch
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)

        epoch_metric_train /= train_step
        print(f'Epoch_metric: {epoch_metric_train:.4f}')

        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)

        # we will update,store & display model parameters when below condition is satisfied
        if (epoch + 1) % test_interval == 0:

            model.eval()
            # model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training
            # eg- Dropouts Layers, BatchNorm Layers etc

            # It means any tensor with gradient currently attached with the current computational graph
            # is now detached from the current graph.
            # We no longer be able to compute the gradients with respect to this tensor.
            with torch.no_grad():
                test_epoch_loss = 0
                test_metric = 0
                epoch_metric_test = 0
                test_step = 0

                for test_data in test_loader:
                    test_step += 1

                    test_volume = test_data["vol"]
                    test_label = test_data["seg"]
                    test_label = test_label != 0
                    test_volume, test_label = (test_volume.to(device), test_label.to(device),)

                    test_outputs = model(test_volume)

                    test_loss = loss(outputs, test_label)
                    test_epoch_loss += test_loss.item()
                    test_metric = dice_metric(test_outputs, test_label)
                    epoch_metric_test += test_metric

                test_epoch_loss /= test_step
                print(f'test_loss_epoch: {test_epoch_loss:.4f}')
                save_loss_test.append(test_epoch_loss)
                np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)

                epoch_metric_test /= test_step
                print(f'test_dice_epoch: {epoch_metric_test:.4f}')
                save_metric_test.append(epoch_metric_test)
                np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)

                # if curr_metric performance is better than previous best_metric, then we save this & update the
                # best_metric performance
                if epoch_metric_test > best_metric:
                    best_metric = epoch_metric_test

                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "best_metric_model.pth"))

                print(
                    f"current epoch: {epoch + 1} current mean dice: {test_metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")


def show_patient(data, SLICE_NUMBER=1, train=True, test=False):
    """
    This function is to show one patient from your datasets, so that you can see if it is okay or you need
    to change/delete something.

    `data`: this parameter should take the patients from the data loader, which means you need to call the function
    prepare first and apply the transforms that you want after that pass it to this function so that you visualize
    the patient with the transforms that you want.

    `SLICE_NUMBER`: this parameter will take the slice number that you want to display/show
    `train`: this parameter is to say that you want to display a patient from the training data (by default it is true)
    `test`: this parameter is to say that you want to display a patient from the testing patients.
    """

    check_patient_train, check_patient_test = data

    # first() function will load the 1st patient record out of all the patients
    view_train_patient = first(check_patient_train)
    view_test_patient = first(check_patient_test)

    if train:
        plt.figure("Visualization Train", (12, 6))
        plt.subplot(1, 2, 1) # plot the 1st row, 2nd column & 1st slice (out of 65) of the volume of the image
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_train_patient["vol"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2) # plot the 1st row, 2nd column & 2nd slice (out of 65) of the volume of the image
        plt.title(f"seg {SLICE_NUMBER}")
        plt.imshow(view_train_patient["seg"][0, 0, :, :, SLICE_NUMBER])
        plt.show()

    if test:
        plt.figure("Visualization Test", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_test_patient["vol"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"seg {SLICE_NUMBER}")
        plt.imshow(view_test_patient["seg"][0, 0, :, :, SLICE_NUMBER])
        plt.show()

# give data as I/P to this function (i.e. the O/P of prepare function is fed as I/P too this function)
# this function will calculate the number of pixels in background and foreground
# hence this will return an array of 2 values- pixels in background + pixels in foreground
# O/P of this function will be fed to def calculate_weights [val1 = background, val2 = foreground]
def calculate_pixels(data):
    val = np.zeros((1, 2))

    for batch in tqdm(data):
        batch_label = batch["seg"] != 0
        _, count = np.unique(batch_label, return_counts=True)

        if len(count) == 1:
            count = np.append(count, 0)
        val += count

    print('The last values:', val)
    return val

in_dir = 'C:\\Users\\Abhinav Garg\\PycharmProjects\\FirstProg\\Liver Segmentation Project\\Liver Segmentation Dataset\\Task03_Liver\\Data_Train_Test'
patient = prepare(in_dir)
show_patient(patient,40)


