import os
import matplotlib.image as mpimg
import cv2
import numpy as np


zeros='./dataTry/0_4/'
ones='./dataTry/1/'


width=252
height=252

images_zeros=[]
images_ones=[]
label_zeros=[]
label_ones=[]
#print(images.sheape)

i=0

for filename in os.listdir(zeros):
    if filename.endswith(".jpg"):
        i=i+1
        #img = mpimg.imread(zeros + filename)
        img=cv2.imread(zeros + filename)
        #print(img)
        images_zeros.append(img)
        label_zeros.append(0)

        #print(filename)
        #print('0 ',np.array(images_zeros).shape)

for filename in os.listdir(ones):
    if filename.endswith(".jpg"):
        #img = mpimg.imread(ones + filename)
        img = cv2.imread(ones + filename)
        #print(len(img))
        images_ones.append(img)
        label_ones.append(1)
        #print('1 ',np.array(images_ones).shape)


images_zeros=np.array(images_zeros)
print(images_zeros.shape)
#images_zeros=np.random.shuffle(images_zeros)

images_ones=np.array(images_ones)
print(images_ones.shape)
#images_ones=np.random.shuffle(images_ones)
#print(images_zeros.shape)
#print(images_ones.shape)
label_zeros=np.array(label_zeros)
label_ones=np.array(label_ones)
images_all=np.concatenate((images_zeros,images_ones),axis=0)
label_all=np.concatenate((label_zeros,label_ones),axis=0)
print(images_all.shape)
print(label_all.shape)
np.save("./data/images_ones.npy",images_ones)
np.save("./data/images_zeros.npy",images_zeros)
np.save("./data/labels_ones.npy",label_ones)
np.save("./data/labels_zeros.npy",label_zeros)