import os
from glob import glob
import shutil
from tqdm import tqdm
import dicom2nifti
import numpy as np
import nibabel as nib

from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from monai.data import DataLoader, Dataset, CacheDataset

"""

This file is for preprocessing only, it contains the functions needed
to make data ready for training.

"""

def create_groups(in_dir, out_dir, Number_slices):
    '''
    This function is to get the last part of the path so that we can use it to name the folder.
    `in_dir`: the path to our folders that contain dicom files
    `out_dir`: the path where we want to put the converted nifti files
    `Number_slices`: here we put the number of slices that you need for our project and it will
    create groups with this number.
    '''

    for patient in glob(in_dir + '/*'):# returns a list of all directories in the in_dir. We iterate over all such directories
        patient_name = os.path.basename(os.path.normpath(patient))
        # os.path.normpath(patient) normalizes the path i.e. if there is any error in the path then it corrects it
        # os.path.basename() provides us the base name of the path i.e. name of the last item of the path (labels in our case)

        # Here we need to calculate the number of folders which mean into how many groups we will divide the number of slices
        # So divide the no of slices of a particular patient by fixed_no_of_slices to get the no of grps or folders
        # Here we need to calculate the number of folders which mean into how many groups we will divide the number of slices
        number_folders = int(len(glob(patient + '/*')) / Number_slices)

        # we need to move slices of the image to the folders so that you will save memory in your device
        for i in range(number_folders): # creating folders where the slices will be moved
            output_path = os.path.join(out_dir, patient_name + '_' + str(i))
            # joining the two paths together then adding patient name
            # os.path.join(path1,path2) joins the two paths together as path1
            os.makedirs(output_path)# make directory with output_path name

            # enumerate() adds a counter to an iterable and returns it in a form of enumerating object(tuple)
            # this tuple contains the count of the iterable & contents of the iterable
            # Move the slices into folder created above
            for i, file in enumerate(glob(patient + '/*')):
                if i == Number_slices + 1: #(if i== 64 + 1)
                    break # we need to stop when we reach 65 slices
                # shutil.move will move the dicom files to the required directory
                shutil.move(file, output_path)

in_dir = 'C:\\Users\\Abhinav Garg\\PycharmProjects\\FirstProg\\Liver Segmentation Project\\Liver Segmentation Dataset\\dicom_files\\images'
out_dir = 'C:\\Users\\Abhinav Garg\\PycharmProjects\\FirstProg\\Liver Segmentation Project\\Liver Segmentation Dataset\\dicom_grps\\images'

in_dir1 = 'C:\\Users\\Abhinav Garg\\PycharmProjects\\FirstProg\\Liver Segmentation Project\\Liver Segmentation Dataset\\dicom_files\\labels'
out_dir1 = 'C:\\Users\\Abhinav Garg\\PycharmProjects\\FirstProg\\Liver Segmentation Project\\Liver Segmentation Dataset\\dicom_grps\\labels'

Number_slices = 64

create_groups(in_dir,out_dir,Number_slices)
create_groups(in_dir1,out_dir1,Number_slices)

print(len(glob(in_dir +'/*')))
print(len(glob(in_dir1 +'/*')))

def dcm2nifti(in_dir, out_dir):
    '''
    This function will be used to convert dicoms into nifti files after creating the groups with
    the number of slices that you want.
    `in_dir`: the path to the folder where you have all the patients (folder of all the groups).
    `out_dir`: the path to the output, which means where you want to save the converted nifties.
    '''

    for folder in tqdm(glob(in_dir + '/*')):
        patient_name = os.path.basename(os.path.normpath(folder))
        # folder = path1 / src path, os.path.join() = path2 / destination path
        # using '.gz' gives us compressed format nifti files
        dicom2nifti.dicom_series_to_nifti(folder, os.path.join(out_dir, patient_name + '.nii.gz'))
        # add .gz to .nii to get the nifti files in the compressed format

in_path_images = 'C:\\Users\\Abhinav Garg\\PycharmProjects\\FirstProg\\Liver Segmentation Project\\Liver Segmentation Dataset\\dicom_grps\\images'
in_path_labels = 'C:\\Users\\Abhinav Garg\\PycharmProjects\\FirstProg\\Liver Segmentation Project\\Liver Segmentation Dataset\\dicom_grps\\labels'

out_path_images = 'C:\\Users\\Abhinav Garg\\PycharmProjects\\FirstProg\\Liver Segmentation Project\\Liver Segmentation Dataset\\nifti_files\\images'
out_path_labels = 'C:\\Users\\Abhinav Garg\\PycharmProjects\\FirstProg\\Liver Segmentation Project\\Liver Segmentation Dataset\\nifti_files\\labels'

dcm2nifti(in_path_images,out_path_images)
dcm2nifti(in_path_labels,out_path_labels)


def find_empy(in_dir):
    '''
    This function helps to find the empty volumes that are not needed for training
    so instead of opening all the files and search for the empty ones, them use this function to make it quick.
    The following pixels represent:
    0 = Background
    1 = liver
    2 = other objects
    '''

    list_patients = []
    for patient in glob(os.path.join(in_dir, '*')):
        img = nib.load(patient) # this will load one particular patient at a time in the form of array

        # fdata() means frame data. It gives an array of 65 item. Each item is an array of image containing pixel values
        # this means that the array contains background, foreground and liver. So don't need to delete it
        img_f_data = img.get_fdata()
        if len(np.unique(img_f_data)) > 2: # means there is something in the background & foreground both
            print(os.path.basename(os.path.normpath(patient)))
            list_patients.append(os.path.basename(os.path.normpath(patient)))
        # if len(np.unique(img_f_data)) ==1, then we skip & don't do anything

    return list_patients

# the values of a_min=-200, a_max=200 were on an average found to be the best. Hence they were chosen accordingly
def prepare(in_dir, pixdim=(1.5, 1.5, 1.0), a_min=-200, a_max=200, spatial_size=[128, 128, 64], cache=False):
    """
    Monai documentation to refer more transforms
    https://monai.io/docs.html
    """

    # set_determinism(seed=0)
    #

    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainSegmentation", "*.nii.gz")))

    path_test_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nii.gz")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "TestSegmentation", "*.nii.gz")))

    # We are creating 2 dictionaries over here with keys & value as = vol, seg (for training and testing purpose)
    # Volumes = Patients & Segmentation = Labels
    # eg- 1st row, 1st column represents the path to the volume of the 1st patient &
    # 1st row, 2nd column represents segmentation of 1st Patient

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                   zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                  zip(path_test_volumes, path_test_segmentation)]

# Compose allows us to use multiple transforms at the same time
    train_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]), # To load the image
            AddChanneld(keys=["vol", "seg"]), # add a channel to our image and label
            # First channel = mask / pixels for background
            # Second channel = mask / pixels for foreground.

            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            # changing the width, height, depth of medical images

            # a_min = intensity original range min
            # a_max = intensity original range max
            # b_min = intensity target range min
            # a_max = intensity target range max

            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            # the first will be to change the contrast from that dense vision into something more visible,
            # and the second will be to normalize the voxel values and place them between 0 and 1 so that the training will be faster.

            CropForegroundd(keys=["vol", "seg"], source_key="vol"), # crops out the empty regions of the image

            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            # if we do not add an operation to give the same dimensions to all patients, our model will not work
            # We need to take spatial size such that we don’t lose a significant image data & also training doesn’t take much longer time.

            ToTensord(keys=["vol", "seg"]),
            # After performing all the transforms, we need to convert them to tensors
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            AddChanneld(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            ToTensord(keys=["vol", "seg"]),

        ]
    )

    if cache:
        # CacheDataset() loads the data into GPU memory. So training will be faster (5-10 times)
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader