import cv2
import os
import numpy as np
import torch
from ImageDataset import MyDataset

from torch.utils.data import DataLoader
import math

dirname="./data/v1"
model = torch.load("./data/model2.pt")
for root, dirs, files in os.walk(dirname):

    for filename in files:
        videoCapture = cv2.VideoCapture(root + "/" +filename)

        # 获得码率及尺寸
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        print("Size:",size)
        # 读帧
        success, frame = videoCapture.read()
        # print(success)
        # print(frame)
        count = 0
        while success:
            if count%100 == 0:
                cv2.imwrite("./data/v_r_testbak_{}.jpg".format(count), frame)
            # cv2.circle(frame, (size[0]//2, size[1]//2), 500, (255, 0, 255), 5)

            # cv2.imshow('windows', frame)  # 显示
            # cv2.waitKey(int(1000 / int(fps))) # 延迟

            print(frame.shape)

            # cv2.imshow("frame",frame)
            #
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            index = []
            crop_image = []
            gap = 252
            stride = 120
            i = 0
            while (i+gap <= size[0]):
                j = 0
                while (j+gap <= size[1]):
                    new_img = np.zeros((252, 252, 3))
                    new_img = frame [j:j + gap, i:i + gap]
                    # print("new",new_img.shape)
                    # print("Top_Right", j + gap, "Bottom_Left", i + gap)

                    crop_image.append(new_img)

                    index.append((i,j))
                    # plb.imshow(new_img)
                    # plt.show()
                    j += stride

                i += stride
            # print(crop_image)
            # print(index)
            images_zeros = np.array(crop_image)
            label_zeros = np.array(index)
            batch_size = 1
            valid_dataset = MyDataset(X=images_zeros, y=label_zeros)
            val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            total_corr = 0
            matrix_0 = np.zeros((size[0] // stride + 1, size[1] // stride + 1))  ### 这里
            # matrix_0 = np.zeros((size[0], size[1] ))  ### 这里
            matrix = np.zeros((size[0] // stride + 1, size[1] // stride + 1))
            # matrix = np.zeros((size[0], size[1]))
            for i, data in enumerate(val_loader):
                images, label = data
                print("label",label)
                images = images.float()
                print(images.shape)
                outputs = model(images)
                # matrix_0[label.detach().numpy()[0][1]:label.detach().numpy()[0][1]+stride,label.detach().numpy()[0][0]:label.detach().numpy()[0][0]+stride    ]= outputs.item()
                # matrix[label.detach().numpy()[0][1]:label.detach().numpy()[0][1] + stride,label.detach().numpy()[0][0]:label.detach().numpy()[0][0] + stride] = (outputs > 0.5).squeeze().long()
                matrix_0[int(label.detach().numpy()[0][0] / stride), int(label.detach().numpy()[0][1] / stride)] = outputs.item()
                matrix[int(label.detach().numpy()[0][0] / stride), int(label.detach().numpy()[0][1] / stride)] = (outputs > 0.5).squeeze().long()
            print("matrix",matrix_0)
            print(matrix_0.shape)
            # cv2.imshow("frame", matrix)
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

            for i in range(matrix.shape[0] - 1, -1, -1):

                for j in range(matrix.shape[1] -1, -1, -1):
                    print(i,j)
                    if matrix[i][j] > 1 and matrix[i][j + 1] > 1:
                        matrix[i][j] = matrix[i][j + 1]
                    if matrix[i][j]>1 and matrix[i+1][j]>1:
                        matrix[i][j]= matrix[i+1][j]
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
            dist = math.sqrt((center[0] - matrix.shape[0] / 2) ** 2 + (center[1] - matrix.shape[1] / 2) ** 2)
            print("dist:", dist)

            if dist * stride <= 500:
                instruction = "Plument."
            else:
                instruction = "Continue"

            print("Instruction: {}".format(instruction))
            cv2.circle(frame, (size[0] // 2, size[1] // 2), 300, (255, 0, 255), 5)
            cv2.arrowedLine(frame,(int(size[0] // 2),int(size[1] // 2)) ,(int(center[0]),int(center[1])) , (0, 0, 255), 5)
            # cv2.resizeWindow("windows", 640, 480);
            cv2.imshow('windows', frame)  # 显示

            cv2.waitKey(100) # 延迟


            success, frame = videoCapture.read()  # 获取下一帧
            count += 1
        videoCapture.release()
