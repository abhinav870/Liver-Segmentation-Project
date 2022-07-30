from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
import torch
from preprocess import prepare
from utilities import train


data_dir = 'C:\\Users\\Abhinav Garg\\PycharmProjects\\FirstProg\\Liver Segmentation Project\\Liver Segmentation Dataset\\Task03_Liver\\Data_Train_Test'
model_dir = 'C:\\Users\\Abhinav Garg\\PycharmProjects\\FirstProg\\Liver Segmentation Project\\Liver Segmentation Dataset\\results\\model'
data_in = prepare(data_dir, cache=True)

device = torch.device("cuda:0")
model = UNet(
    dimensions = 3,
    in_channels = 1,
    out_channels = 2,
    channels = (16, 32, 64, 128, 256),
    strides = (2, 2, 2, 2),
    num_res_units = 2,
    norm=Norm.BATCH,
).to(device)

"""

(a) For 3D volume segmentation, we need dimensions = 3
(b) I/P channel = 1 (we have mask with only one channel i.e. each slice has only one channel comprising of 0s and 1s)
(c) O/P channels = 2 (this is equal to number of classes)
    First Channel = Pixel probabilities for the background
    Second Channel = Pixel probabilities for the foreground
(d) channels = Kernels/Filters in CNN 
(e) strides
(f) residual units
(g) Performing Batch Normalization

"""

# ce_weights = ?, these weights are found using calculate_weights() function
#loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

"""

Loss Function = Dice Loss
Activation Function = Sigmoid
Optimizer = Adam with learning rate = 1e-5

"""

if __name__ == '__main__':
    # train(model, data_in, loss_function, optimizer, 600, model_dir)
    # We will train the model for 100 epochs
    train(model, data_in, loss_function, optimizer, 100, model_dir)

