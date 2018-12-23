import os
import matplotlib.image as mpimg
import cv2
import numpy as np
zeros='./dataTry/0_test/'
ones='./dataTry/1_test/'

images_zeros=[]
images_ones=[]
label_zeros=[]
label_ones=[]

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

images_zeros = np.array(images_zeros)
print(images_zeros.shape)
# images_zeros=np.random.shuffle(images_zeros)

images_ones = np.array(images_ones)
print(images_ones.shape)
images_all=np.concatenate((images_zeros,images_ones),axis=0)
np.save("./data/images_ones_test.npy",images_ones)
np.save("./data/images_zeros_test.npy",images_zeros)