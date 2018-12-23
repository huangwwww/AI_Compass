import numpy
import pylab as plb
import PIL.Image as Image
import matplotlib.pyplot as plt
import os
import shutil

dirname = "./data/p1"
new_dir = "./data/p2"


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("create p2")


    else:
        print("There is p2")


shutil.rmtree(new_dir)
mkdir(new_dir)
for root, dirs, files in os.walk(dirname):
    print(root)
    for filename in files:
        # print(root+"/"+filename)
        img = Image.open(root+"/"+filename)
        print(img.size) # (4032, 2268)
        length = img.size[0]
        wide = img.size[1]
        # if (length/16 != wide/9 and length/9 != wide/16  ):
        #     print("Image Shape Error")
        #     break

        mul = 1 # How many pieces to cut
        gap = 252
        i = 0
        # print(gap)
        stride = 252
        while (i <= img.size[1]):
            j = 0
            while (j <= img.size[0]):
                new_img = img.crop((j,i,j+gap,i+gap))
                print("Top_Right",j+gap,"Bottom_Left", i+gap)

                new_img.save("{}/{}_length_{}_wide_{}_.jpg".format(new_dir, filename[0:-4] ,i,j))



                # plb.imshow(new_img)
                # plt.show()
                j += stride

            i += stride
