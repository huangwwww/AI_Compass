import numpy
import pylab as plb
import PIL.Image as Image
import matplotlib.pyplot as plt
import os

dirname = "./data/p1"
new_dir = "./data/p2"
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
        length = 0
        print(gap)

        while (length <= img.size[0]-gap):
            wide = 0
            while (wide <= img.size[1]-gap):
                new_img = img.crop((length,wide,length+gap,wide+gap))
                print("Top_Right",length+gap,"Bottom_Left", wide+gap)

                new_img.save("{}/{}_length_{}_wide_{}.jpg".format(new_dir, filename[0:-4] ,length,wide))



                # plb.imshow(new_img)
                # plt.show()
                wide += gap

            length += gap
