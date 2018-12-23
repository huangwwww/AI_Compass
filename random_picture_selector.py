import numpy
import pylab as plb
import PIL.Image as Image
import matplotlib.pyplot as plt
import os
import random
from shutil import copyfile


dirname = "./data/1_2"
new_dir = "./data/1_3"

for root, dirs, files in os.walk(dirname):
    # print(files)
    randomlist = random.sample(files,300)

# print(randomlist)

for filenames in randomlist:
    copyfile(dirname+"/"+filenames, new_dir+"/"+filenames)