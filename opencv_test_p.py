import cv2
import os
import numpy as np
import torch
from ImageDataset import MyDataset

from torch.utils.data import DataLoader
import math
dirname="./data/v_r"
newdir = "./data/v1"
stride = 256
gap =252
model = torch.load("./data/model2.pt")
for root, dirs, files in os.walk(dirname):

    for filename in files:

        # print(filename)
        frame = cv2.imread(root + "/" + filename)
        print("!!!", frame.shape)

        newframe = cv2.copyMakeBorder(frame,0,stride-frame.shape[0]%stride,0,stride-frame.shape[1]%stride,cv2.BORDER_CONSTANT,value=0)
        print(stride-frame.shape[0]%stride)
        print(stride - frame.shape[1] % stride)
        # newframe = cv2.resize(newframe, (frame.shape[1]+ stride-frame.shape[1]%stride,frame.shape[0]+ stride-frame.shape[0]%stride))
        print("!!!",newframe.shape)

        # cv2.imshow("windows",newframe)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        index = []
        crop_image = []
        i = 0
        while (i + gap <= newframe.shape[0]):
            j = 0
            while (j + gap <= newframe.shape[1]):
                print(i,j)
                new_img = np.zeros((252, 252, 3))
                new_img = newframe[ i:i + gap,j:j + gap]
                # print("new",new_img.shape)
                # print("Top_Right", j + gap, "Bottom_Left", i + gap)
                # cv2.imshow("windows", new_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # cv2.imwrite(newdir + "/" + filename, frame)
                print("size:",new_img.shape)
                crop_image.append(new_img)

                index.append((i, j))


                j += stride

            i += stride

        images_zeros = np.array(crop_image)
        label_zeros = np.array(index)
        batch_size = 1
        valid_dataset = MyDataset(X=images_zeros, y=label_zeros)
        val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        total_corr = 0
        matrix_0 = np.zeros((newframe.shape[0] // stride , newframe.shape[1] // stride))  ### 这里
        # matrix_0 = np.zeros((newframe.shape[0], newframe.shape[1] ))  ### 这里
        matrix = np.zeros((newframe.shape[0] // stride , newframe.shape[1] // stride ))
        # matrix = np.zeros((newframe.shape[0], newframe.shape[1]))

        for i, data in enumerate(val_loader):
            images, label = data
            print("label", label)
            images = images.float()
            print(images.shape)
            outputs = model(images)


            # matrix_0[label.detach().numpy()[0][1]:label.detach().numpy()[0][1]+stride,label.detach().numpy()[0][0]:label.detach().numpy()[0][0]+stride    ]= outputs.item()
            # matrix[label.detach().numpy()[0][1]:label.detach().numpy()[0][1] + stride,label.detach().numpy()[0][0]:label.detach().numpy()[0][0] + stride] = (outputs > 0.5).squeeze().long()
            matrix_0[int(label.detach().numpy()[0][0] / stride), int(label.detach().numpy()[0][1] / stride)] = outputs.item()
            matrix[int(label.detach().numpy()[0][0] / stride), int(label.detach().numpy()[0][1] / stride)] = (outputs > 0.5).squeeze().long()
            print("output:",(outputs > 0.5).squeeze().long())
        print("matrix", matrix)
        print("matrix_0",matrix_0)
        print(matrix_0.shape)
        # cv2.imshow("frame", matrix)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        n = 2
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                list = np.zeros((4, 1))
                if matrix[i][j] == 1:
                    if i - 1 < 0:  # left
                        list[0] = 0
                    else:
                        list[0] = matrix[i - 1][j]
                    if i + 1 >= matrix.shape[0]:  # right
                        list[1] = 0
                    else:
                        list[1] = matrix[i + 1][j]
                    if j - 1 <= 0:  # up
                        list[2] = 0
                    else:
                        list[2] = matrix[i][j - 1]
                    if j + 1 >= matrix.shape[1]:  # bottom
                        list[3] = 0
                    else:
                        list[3] = matrix[i][j + 1]
                    if np.sum(list) <= 0:
                        matrix[i][j] = 0
                    else:
                        if list[0] == 0 and list[2] == 0:
                            matrix[i][j] = n
                            n += 1
                        elif list[0] != 0 and list[2] != 0:
                            matrix[i][j] = min(list[0], list[2]).item()
                        else:
                            matrix[i][j] = sum(list[0], list[2]).item()
        # print(matrix)
        # cv2.imshow("windows", matrix)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite(newdir + "/" + filename, matrix)

        for i in range(matrix.shape[0] - 2, -1, -1):

            for j in range(matrix.shape[1] - 2, -1, -1):
                print(i, j)
                if matrix[i][j] > 1 and matrix[i][j + 1] > 1:
                    matrix[i][j] = matrix[i][j + 1]
                if matrix[i][j] > 1 and matrix[i + 1][j] > 1:
                    matrix[i][j] = matrix[i + 1][j]
        print(matrix)

        i = 2
        m_max = 0
        max_list = []
        while i < n:
            # print(i)
            list = []
            for j in range(matrix.shape[0]):
                for k in range(matrix.shape[1]):
                    if matrix[j][k] == i:
                        print(j)
                        list.append((j, k))
            if len(list) > m_max:
                # if len(list) != 0:
                #     if matrix_0[list[len(list) // 2][0]][list[len(list) // 2][1]] > m_max:
                m_max = len(list)
                # m_max = matrix_0[list[0][0]][list[0][1]]
                max_list = list
                print(i)

            i += 1

        print(max_list)
        x = []
        y = []
        for i in max_list:
            x.append(i[0])
            y.append(i[1])

        center = ((min(x) + max(x) + 0.5) / 2, (min(y) + max(y) + 0.5) / 2)
        print(center)
        cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 500, (255, 0, 255), 5)
        cv2.arrowedLine(frame, (int(frame.shape[1] // 2), int(frame.shape[0] // 2)), (int(center[1]*stride), int(center[0]*stride)), (0, 0, 255),5)
        # cv2.resizeWindow("windows", 640, 480);
        # cv2.imshow('windows', frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite(newdir+"/"+filename, frame)