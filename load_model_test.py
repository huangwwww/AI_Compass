import numpy as np
import torch
from torch.utils.data import DataLoader

from ImageDataset import MyDataset

import pylab as plb
import PIL.Image as Image
import matplotlib.pyplot as plt
import os
from scipy.misc import imread, imresize
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



dirname = "./data/p1"
for root, dirs, files in os.walk(dirname):
    print(root)
    for filename in files:
        img = Image.open(root + "/" + filename)
        c=np.array(img)
        c = np.transpose(c,(1, 0, 2))
        plt.imshow(c)
        plt.gray()
        plt.show()
        # print(img.size)  # (4032, 2268)


model = torch.load("./data/model2.pt")

feat_valid = np.load("./data/images_test.npy")
label_valid = np.load("./data/labels_test.npy")

stride = 252
# feat_valid = feat_valid.transpose((0,2,1))

batch_size = 1
valid_dataset= MyDataset(X =feat_valid,y = label_valid)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

total_corr = 0

matrix = np.zeros(img.size)
for i, data in enumerate(val_loader):
    images, label = data
    images = images.float()
    # print(label)
    # print("data: ",inputs.shape)
    # print("label",label.detach().numpy()[0][0])
    # print("label",label.detach().numpy()[0][1])

    outputs = model(images)
    # print("outputs:{}".format(outputs.item()))
    # print("!!!!:{}".format((outputs> 0.5).squeeze().long()))
    # total_corr += sum(label.long() == torch.max(outputs, 1)[1]).item()
    # total_corr += (label.long() == torch.max(outputs, 1)[1]).sum().item()
    # print("detach",label.detach().numpy()[0][1],label.detach().numpy()[0][0])
    matrix[label.detach().numpy()[0][1]:label.detach().numpy()[0][1]+stride,label.detach().numpy()[0][0]:label.detach().numpy()[0][0]+stride    ]= outputs.item()
    # matrix[label.detach().numpy()[0][1]:label.detach().numpy()[0][1]+stride,label.detach().numpy()[0][0]:label.detach().numpy()[0][0]+stride]= (outputs> 0.5).squeeze().long()
# print(matrix.shape)
# for i in range(matrix.shape[0]):
#     for j in range(matrix.shape[1]):
#         print(matrix[0][1])
# print(np.array(matrix))
plt.imshow(np.array(matrix))
# plt.gray()
plt.show()
# val_acc = total_corr / len(val_loader.dataset)


# print("Validation Accuracy for Model: ",val_acc)

# print(c.shape)
# print(matrix.shape)
white = []
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        if matrix[i][j] == 1:
            white.append((i,j))

# print(len(white))

np.save("./data/white.npy",np.array(white))
np.save("./data/white_m.npy",np.array(matrix))
