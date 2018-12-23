
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import os

matrix = np.load("./data/white_m.npy")
print(matrix)
n = 2
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        list = np.zeros((4,1))
        if matrix[i][j] == 1:
            if i-1 <0:#left
                list[0] = 0
            else:
                list[0] = matrix[i-1][j]
            if i+1 >= matrix.shape[0]:# right
                list[1] = 0
            else:
                list[1] = matrix[i+1][j]
            if j - 1 <=0: # up
                list[2] = 0
            else:
                list[2]= matrix[i][j-1]
            if j + 1 >= matrix.shape[1]: #bottom
                list[3] = 0
            else:
                list[3] = matrix[i][j+1]
            if np.sum(list) <= 1:
                matrix[i][j] = 0
            else:
                if list[0] == 0 and list[2] == 0:
                    matrix[i][j] = n
                    n +=1
                elif list[0] != 0 and list[2]!= 0:
                    matrix[i][j] = min(list[0],list[2]).item()
                else:
                    matrix[i][j] = sum(list[0],list[2]).item()

plt.imshow(np.array(matrix))
plt.gray()
plt.show()

# print(n-2)
i = 2
m_max = 0
max_list = []
while i<n:
    # print(i)
    list = []
    for j in range(matrix.shape[0]):
        for k in range(matrix.shape[1]):
            if matrix[j][k] == i:
                list.append((j,k))
    if len(list) > m_max:
        m_max = len(list)
        max_list = list
        print(i)

    i+=1

print(max_list)
x =[]
y =[]
for i in max_list:
     x.append(i[0])
     y.append(i[1])

center = ((min(x)+max(x))/2,(min(y)+max(y))/2)
print(center)


dirname = "./data/p1"
for root, dirs, files in os.walk(dirname):
    print(root)
    for filename in files:
        img = Image.open(root + "/" + filename)
        c=np.array(img)
        ax = plt.axes()
        c = np.transpose(c,(1, 0, 2))
        ax.arrow(matrix.shape[1] / 2, matrix.shape[0] / 2, center[1]-matrix.shape[1] / 2, center[0]-matrix.shape[0] / 2, head_width=100, head_length=100,fc='k', ec='k')
        # ax.arrow(0, 0, center[0], center[1], head_width=0.05, head_length=0.1,fc='k', ec='k')
        plt.imshow(c)
        plt.gray()
        plt.show()


# plt.imshow(np.array(matrix))
# plt.show()
