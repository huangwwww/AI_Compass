import cv2
import os
import numpy as np
import torch
from ImageDataset import MyDataset

from torch.utils.data import DataLoader
import math

dirname="./data/v1"

for root, dirs, files in os.walk(dirname):

    for filename in files:
        videoCapture = cv2.VideoCapture(root + "/" + filename)

        # 获得码率及尺寸
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        print("Size:", size)
        # 读帧
        success, frame = videoCapture.read()
        count = 0
        while success:
            # cv2.imshow("Oto Video", frame)  # display this frame
            cv2.imwrite("./data/v_r/{}.jpg".format(count),frame)

            cv2.waitKey(int(fps))  # delay
              # write one frame into the output video
            success, frame = videoCapture.read()  # get the next frame of the video
            count += 1
        # some process after finish all the program
    cv2.destroyAllWindows()  # close all the widows opened inside the program
    videoCapture.release  # release the video read/write handler





