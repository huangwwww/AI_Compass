import cv2
import os
import numpy as np
import torch
from ImageDataset import MyDataset

from torch.utils.data import DataLoader
import math
import pandas as pd


dirname = "./data/v_r"
cdir = "Testresult.csv"
newdir = "./data/v_c/"
chart = pd.read_csv(cdir)




for root, dirs, files in os.walk(dirname):
    i = 0
    for filename in files:
        print(i)
        frame = cv2.imread(root + "/" + filename)
        # print(filename)
        name = chart["Name"][i]
        center_1 = chart["Center_1"][i]
        center_2 = chart["Center_2"][i]
        # print(name,center)
        #

        cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 500, (255, 0, 255), 5)
        # # # cv2.imshow('windows', frame)
        cv2.line(frame, (int(frame.shape[1] // 2), int(frame.shape[0] // 2)),((int(center_2)), int(center_1)), (155,155,155), 5)
        cv2.imwrite(newdir+filename, frame)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        i +=1