import numpy as np
import pylab as plb
import PIL.Image as Image
import matplotlib.pyplot as plt
import os
import shutil
import matplotlib.image as mpimg
import torch
from torch.utils.data import DataLoader

from ImageDataset import MyDataset
import math
import pandas as pd
import time

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("create p2")


    else:
        print("There is p2")

# def dotproduct(v1, v2):
#   return sum((a*b) for a, b in zip(v1, v2))
#
# def length(v):
#   return math.sqrt(dotproduct(v, v))
#
# def angle(v1, v2):
#   return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def angle(v1, v2, acute):
# v1 is your firsr vector
# v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if (acute == True):
        return angle
    else:
        return 2 * np.pi - angle


dirname = "./data/v_r"
new_dir = "./data/p2"
new_dir_2 = "./data/v_p"
mkdir(new_dir)
test_result_name = []
test_result_center_1 = []
test_result_center_2 = []
ins = []
for root, dirs, files in os.walk(dirname):
    step = 1
    for filename in files:
        print("Step:",step)
        start_time = time.time()
        shutil.rmtree(new_dir)
        mkdir(new_dir)
        img_0 = Image.open(root + "/" + filename)

        # print(img_0.size)  # (4032, 2268)
        length = img_0.size[0]
        wide = img_0.size[1]
        mul = 1  # How many pieces to cut
        gap = 252
        i = 0
        # print(gap)
        stride = 252
        while (i <= img_0.size[1]):
            j = 0
            while (j <= img_0.size[0]):
                new_img = img_0.crop((j, i, j + gap, i + gap))
                # print("Top_Right", j + gap, "Bottom_Left", i + gap)

                new_img.save("{}/{}_length_{}_wide_{}_.jpg".format(new_dir, filename[0:-4], i, j))

                # plb.imshow(new_img)
                # plt.show()
                j += stride

            i += stride

        width = 252
        height = 252

        images_zeros = []
        images_ones = []
        label_zeros = []
        label_ones = []
        # print(images.sheape)

        i = 0

        for filename2 in os.listdir(new_dir):
            if filename2.endswith(".jpg"):
                i = i + 1
                img = mpimg.imread(new_dir +"/"+ filename2)
                images_zeros.append(img)
                # print(filename2)
                a = [int(s) for s in filename2.split("_") if s.isdigit()]
                # print(a)
                label_zeros.append((a[-2], a[-1]))

                # print(filename)
                # print('0 ',np.array(images_zeros).shape)

        images_zeros = np.array(images_zeros)
        label_zeros = np.array(label_zeros)

        np.save("./data/images_test.npy", images_zeros)

        np.save("./data/labels_test.npy", label_zeros)

        c = np.array(img_0)
        c = np.transpose(c, (1, 0, 2))
        # plt.imshow(c)
        # plt.gray()
        # plt.show()

        model = torch.load("./data/model4.pt")

        feat_valid = np.load("./data/images_test.npy")
        label_valid = np.load("./data/labels_test.npy")


        # feat_valid = feat_valid.transpose((0,2,1))

        batch_size = 1
        valid_dataset = MyDataset(X=feat_valid, y=label_valid)
        val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        total_corr = 0
        matrix_0 = np.zeros((img_0.size[0]//252+1,img_0.size[1]//252+1)) ### 这里
        matrix = np.zeros((img_0.size[0]//252+1,img_0.size[1]//252+1))
        for i, data in enumerate(val_loader):
            images, label = data
            images = images.float()
            # print(label)
            # print("data: ",inputs.shape)
            # print("label", label.detach().numpy()[0][0])
            # print("label", label.detach().numpy()[0][1])

            outputs = model(images)
            # print("outputs:{}".format(outputs.item()))
            # print("!!!!:{}".format((outputs> 0.5).squeeze().long()))
            # total_corr += sum(label.long() == torch.max(outputs, 1)[1]).item()
            # total_corr += (label.long() == torch.max(outputs, 1)[1]).sum().item()
            # print("detach",label.detach().numpy()[0][1],label.detach().numpy()[0][0])
            # matrix_0[label.detach().numpy()[0][1]:label.detach().numpy()[0][1]+stride,label.detach().numpy()[0][0]:label.detach().numpy()[0][0]+stride    ]= outputs.item()
            # print(label.detach().numpy()[0][1]/252)
            # print(label.detach().numpy()[0][0]/252)
            matrix_0[int(label.detach().numpy()[0][1]/252),int(label.detach().numpy()[0][0]/252) ]= outputs.item()

            # print("Label!!!!:",label.detach().numpy()[0][1])


            # matrix[label.detach().numpy()[0][1]:label.detach().numpy()[0][1] + stride,label.detach().numpy()[0][0]:label.detach().numpy()[0][0] + stride] = (outputs > 0.5).squeeze().long()
            matrix[int(label.detach().numpy()[0][1]/252),int(label.detach().numpy()[0][0]/252) ] = (outputs > 0.5 ).squeeze().long()
        # print(matrix.shape)
        # for i in range(matrix.shape[0]):
        #     for j in range(matrix.shape[1]):
        #         print(matrix[0][1])
        # print(np.array(matrix))
        # plt.imshow(np.array(matrix_0))
        # plt.gray()
        # plt.show()
        # val_acc = total_corr / len(val_loader.dataset)

        # print("Validation Accuracy for Model: ",val_acc)

        # print(c.shape)
        print(matrix.shape)
        np.save("./data/white_m.npy", np.array(matrix))
        np.save("./data/white_m_0.npy",np.array(matrix_0))
        #####
        matrix = np.load("./data/white_m.npy")
        matrix_0 = np.load("./data/white_m_0.npy")

        # print(matrix)
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
        print(matrix.shape[0])
        print(matrix.shape[1])
        for i in range(matrix.shape[0]-1,0,-1):
            for j in range(matrix.shape[1]-2,0,-1):
                print(j)
                if matrix[i][j]>1 and matrix[i][j+1]>1:
                    matrix[i][j]=matrix[i][j+1]

        #
        # plt.imshow(np.array(matrix))
        # plt.gray()
        # plt.show()

        # print(n-2)
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
            #     if matrix_0[list[len(list)//2][0]][list[len(list)//2][1]]> m_max:
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

        center = ((min(x) + max(x)+0.5) / 2, (min(y) + max(y)+0.5) / 2)
        print(center)

        dist = math.sqrt(( center[0]- matrix.shape[0] / 2) ** 2 + (center[1]- matrix.shape[1] / 2) ** 2)
        print("dist:",dist)

        if dist*stride <=500:
            instruction = "Plument."
        else:
            instruction = "Continue"
        #  instruction = angle((matrix.shape[0] / 2-center[0],matrix.shape[1] / 2-center[1]),(0,1),True)/math.pi * 180
        ins.append(instruction)
        print("Instruction: {}".format(instruction))
        test_result_name.append(filename)
        test_result_center_1.append(center[0]*stride)
        test_result_center_2.append(center[1]*stride)
        print("Test: ", test_result_name, " | ",test_result_center_1,test_result_center_2)
        if instruction == "Continue":
            circle1 = plt.Circle((img_0.size[1] / 2,img_0.size[0] / 2), 500, color='b',fill=False)
        else:
            circle1 = plt.Circle((img_0.size[1] / 2, img_0.size[0] / 2), 500, color='g', fill=False)
        img = Image.open(root + "/" + filename)
        c = np.array(img)
        fig, ax = plt.subplots()
        ax = plt.axes()
        c = np.transpose(c, (1, 0, 2))
        ax.arrow(img_0.size[1] / 2, img_0.size[0] / 2, center[1]*stride - img_0.size[1] / 2,
                 center[0]*stride - img_0.size[0] / 2, head_width=100, head_length=100, fc='k', ec='k')
        # ax.arrow(0, 0, center[0], center[1], head_width=0.05, head_length=0.1,fc='k', ec='k')
        ax.add_artist(circle1)

        plt.imshow(c)
        plt.savefig(new_dir_2+"/"+filename)
        plt.gray()
        # plt.show()


        end_time = time.time()

        time_difference = end_time - start_time
        print("Time: ",time_difference)
        step+=1

df = pd.DataFrame({"Name": test_result_name, "Center_1": test_result_center_1,"Center_2":test_result_center_2,"Instruction":ins})
df.to_csv("Testresult.csv", index=False)