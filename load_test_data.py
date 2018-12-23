import os
import matplotlib.image as mpimg

import numpy as np


test='./data/p2/'



width=252
height=252

images_zeros=[]
images_ones=[]
label_zeros=[]
label_ones=[]
#print(images.sheape)

i=0

for filename in os.listdir(test):
    if filename.endswith(".jpg"):
        i=i+1
        img = mpimg.imread(test + filename)
        images_zeros.append(img)
        print(filename)
        a = [int(s) for s in filename.split("_") if s.isdigit()]
        print(a)
        label_zeros.append((a[-2],a[-1]))

        #print(filename)
        #print('0 ',np.array(images_zeros).shape)



images_zeros=np.array(images_zeros)
label_zeros=np.array(label_zeros)


np.save("./data/images_test.npy",images_zeros)

np.save("./data/labels_test.npy",label_zeros)