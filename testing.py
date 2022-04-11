#============================================================
#
#   Script to
#   - Obtain the destriping result of the input
#
#============================================================

#Python
import numpy as np
import tensorflow as tf
import h5py
import scipy.io as scio
#Keras
from keras.models import load_model
from keras import backend as K

# load hdf5 datasets
def load_hdf5(infile):
    with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
        return f["image"][()]

# convert RGB image in black and white, channel from 3 to 1
def rgb2gray(data):
    assert (len(data.shape)==4)  # 4D arrays: batch_size*channel*img_hight*img_width
    assert (data.shape[1]==3)
    gray_img = (data[:,0,:,:] + data[:,1,:,:] + data[:,2,:,:])/3
    gray_img = np.reshape(gray_img,(data.shape[0],1,data.shape[2],data.shape[3]))
    return gray_img

# dataset normalization
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  # 4D arrays: batch_size*channel*img_hight*img_width
    assert (imgs.shape[1]==1)    # gray image: channel=1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized

def data_preprocess(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  # check the input channel
    train_imgs = rgb2gray(data) # optional
    train_imgs = dataset_normalized(train_imgs)  # optional
    train_imgs = train_imgs/255.  # image reduced to 0-1 
    return train_imgs

# extend the border of input image for dividing into patches
def extend_border(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape)==4)  # 4D arrays: batch_size*channel*img_hight*img_width
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3] # width of the full image
    leftover_h = (img_h-patch_h)%stride_h  # leftover on the h dim
    leftover_w = (img_w-patch_w)%stride_w  # leftover on the w dim
    if (leftover_h != 0):  # change dimension of img_h
        print("\nthe side H is not compatible with the selected stride of " +str(stride_h))
        print("img_h " +str(img_h) + ", patch_h " +str(patch_h) + ", stride_h " +str(stride_h))
        print("(img_h - patch_h) MOD stride_h: " +str(leftover_h))
        print("So the H dim will be padded with additional " +str(stride_h - leftover_h) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_h+(stride_h-leftover_h),img_w))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:img_h,0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    if (leftover_w != 0):   # change dimension of img_w
        print("the side W is not compatible with the selected stride of " +str(stride_w))
        print("img_w " +str(img_w) + ", patch_w " +str(patch_w) + ", stride_w " +str(stride_w))
        print("(img_w - patch_w) MOD stride_w: " +str(leftover_w))
        print("So the W dim will be padded with additional " +str(stride_w - leftover_w) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],full_imgs.shape[2],img_w+(stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:full_imgs.shape[2],0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    print("new full images shape: \n" +str(full_imgs.shape))
    return full_imgs

# divide the full input image into patches according to patch_h/w and stride_h/w
def extract_patches(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape)==4)   # 4D arrays: batch_size*channel*img_hight*img_width
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3] # width of the full image
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  # division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]
    print("Number of patches on h : " +str(((img_h-patch_h)//stride_h+1)))
    print("Number of patches on w : " +str(((img_w-patch_w)//stride_w+1)))
    print("number of patches per image: " +str(N_patches_img) +", totally for this dataset: " +str(N_patches_tot))
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
    iter_tot = 0   # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                patch = full_imgs[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                patches[iter_tot]=patch
                iter_tot +=1   # total
    assert (iter_tot==N_patches_tot)
    return patches  # array with all the full_imgs divided in patches

# load the original datasets and return the divided patches
def get_testint_data(original_test_imgs_path, patch_height, patch_width, stride_height, stride_width): 
    test_imgs_original = load_hdf5(original_test_imgs_path)
    test_imgs = data_preprocess(test_imgs_original)
    test_imgs = extend_border(test_imgs, patch_height, patch_width, stride_height, stride_width)

    print("test images shape:"+str(test_imgs.shape))
    print("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))

    patches_imgs_test = extract_patches(test_imgs,patch_height,patch_width,stride_height,stride_width)
    print("\ntest PATCHES images shape:"+str(patches_imgs_test.shape))
    print("test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))

    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3]

# recover full images with the patches
def recompone_patches(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape)==4)   # 4D arrays: batch_size*channel*img_hight*img_width
    assert (preds.shape[1]==1 or preds.shape[1]==3) 
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    print("N_patches_h: " +str(N_patches_h))
    print("N_patches_w: " +str(N_patches_w))
    print("N_patches_img: " +str(N_patches_img))
    assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    print("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
    full_value = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))
    full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))
    preds = preds/np.max(preds)
    print(preds.shape)
    k = 0 # iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_value[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k]
                full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1
                k+=1
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)  # at least one
    print(np.max(full_value),np.min(full_value),np.max(full_sum),np.min(full_sum))
    # final_avg = final_avg/K.max(final_avg)
    final_avg = full_value/full_sum
    # print(final_avg.shape)
    return final_avg

def m_psnr(y_true,y_pred):
    return 10.0 * K.log(1.0 / (K.mean(K.square(y_pred/K.max(y_pred) - y_true)))) / K.log(10.0)
def m_mae(y_true,y_pred):
    return  5*K.mean(K.abs(y_true-y_pred/K.max(y_pred)))
def m_mse(y_true,y_pred):
    return  10*K.mean(K.square(y_true-y_pred/K.max(y_pred)))


#================ Load original input ================
test_data_path = './Data/Testing Dataset.hdf5'
test_imgs_orig = load_hdf5(test_data_path)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]

#================ Load the testing datasets and divide into patches ================
test_patches = None
new_height = None
new_width = None
masks_test  = None
# adjustable parameters which decide the number of patches
# It is suggested to adjust the parameters according to the testing datasets.
Patch_Height = 512
Patch_Width = 512
Stride_Height=32
Stride_Width=32

test_patches, new_height, new_width = get_testint_data(original_test_imgs_path = test_data_path, 
        patch_height = Patch_Height,
        patch_width = Patch_Width,
        stride_height = Stride_Height,
        stride_width = Stride_Width)

n_ch = test_patches.shape[1]
patch_height = test_patches.shape[2]
patch_width = test_patches.shape[3]

#================ Test the data with well-trained model ================
model=load_model('./Model/model2.h5',custom_objects={'tf': tf,'m_psnr':m_psnr,'m_mae':m_mae,'m_mse':m_mse}) # Select a model to test.
test_patches=test_patches.transpose(0,2,3,1)
print(test_patches.shape)
results,lossout = model.predict(test_patches, batch_size=32, verbose=2)
print("predicted images size :"+str(results.shape))
results=results.transpose(0,3,1,2)
print("final images:"+str(results.shape))

#================ Save the results ================
imgs_output = None
imgs_output = recompone_patches(results, new_height, new_width, Stride_Height, Stride_Width)
print(np.max(imgs_output),np.min(imgs_output))
imgs_output = imgs_output[:,:,0:full_img_height,0:full_img_width]
print("results shape: " +str(imgs_output.shape))

scio.savemat('./Results.mat',{'Results':imgs_output}) # Save the results directly to mat format. The results can be saved in other way.

