import os
import cv2
import pandas as pd
dirname="./data/v_p"
fps =  10   #保存视频的FPS，可以适当调整

#可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
videoWriter = cv2.VideoWriter('saveVideo.avi',fourcc,fps,(640,480))#最后一个是保存图片的尺寸
cdir = "Testresult.csv"
chart = pd.read_csv(cdir)
x = 450
y = 10
w = 200
h = 45
for root, dirs, files in os.walk(dirname):
    i=0
    for filename in files:
        frame = cv2.imread(root + "/" + filename)
        name = chart["Name"][i]
        center_1 = chart["Center_1"][i]
        center_2 = chart["Center_2"][i]
        ins = chart["Instruction"][i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,0), 1)
        cv2.putText(frame, name, (x+3, y+10), 2, 0.3, (0,0,0), 1)
        cv2.putText(frame, "Centroid: x:"+str(center_2)+"   ,   y:"+str(center_1), (x + 3, y + 20), 2, 0.3, (0, 0, 0), 1)
        if ins == "Continue":
            cv2.putText(frame, ins, (x + 50, y + 40), 2, 0.5, (255, 0, 0), 1)
        else:
            cv2.putText(frame, ins, (x + 50, y + 40), 2, 0.5, (0, 255, 0), 1)
        videoWriter.write(frame)
        i+=1
videoWriter.release()