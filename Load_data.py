#============================================================
#
#   Script to
#   - Prepare the hdf5 dataset of the input images
#
#============================================================


import os
import h5py
import numpy as np
from PIL import Image



def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


# Path of the input images
original_imgs_test = "./Data/images/"

# Parameters of the input images
Number = 3
channels = 3
height = 2048
width = 2048
dataset_path = "./Data/"


def get_datasets(imgs_dir):
    imgs = np.empty((Number,height,width,channels))
    for path, subdirs, files in os.walk(imgs_dir):  # list all files, directories in the path
        for i in range(len(files)):
            print("original image: " +files[i])
            names,formatt=files[i].split(".")
            name,num=names.split("e")
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
    print("test images range (min-max): " +str(np.min(imgs)) +' - '+str(np.max(imgs)))
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Number,channels,height,width))
    return imgs

# Preparing the testing datasets
imgs_test = get_datasets(original_imgs_test)
print("saving test datasets")
write_hdf5(imgs_test,dataset_path + "Testing Dataset.hdf5")

