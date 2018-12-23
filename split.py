import numpy as np

seed=2
data=30

images_ones=np.load("./data/images_ones.npy")
images_zeros=np.load("./data/images_zeros.npy")

images_zeros=np.random.permutation(images_zeros)
images_ones=np.random.permutation(images_ones)
#images_zeros=images_zeros[:data,:,:,:]
#images_ones=images_ones[:data,:,:,:]
print(images_ones.shape)

#train_one, val_one, test_one = np.split(images_ones, [int(.64*len(images_ones)), int(.8*len(images_ones))])
#train_zero, val_zero, test_zero = np.split(images_zeros, [int(.64*len(images_zeros)), int(.8*len(images_zeros))])

train_one, val_one = np.split(images_ones, [int(.8*len(images_ones))])
train_zero, val_zero = np.split(images_zeros, [int(.8*len(images_zeros))])


train_label_ones=np.ones(train_one.shape[0])
val_label_ones=np.ones(val_one.shape[0])
#test_label_ones=np.ones(test_one.shape[0])

train_label_zeros=np.zeros(train_zero.shape[0])
val_label_zeros=np.zeros(val_zero.shape[0])
#test_label_zeros=np.zeros(test_zero.shape[0])

train_images=np.concatenate((train_one,train_zero))
val_images=np.concatenate((val_one,val_zero))
#test_images=np.concatenate((test_one,test_zero))


train_label=np.concatenate((train_label_ones,train_label_zeros))
val_label=np.concatenate((val_label_ones,val_label_zeros))
#test_label=np.concatenate((test_label_ones,test_label_zeros))

np.save("./data/train_images.npy",train_images)
np.save("./data/val_images.npy",val_images)
#np.save("./data/test_images.npy",test_images)

np.save("./data/train_label.npy",train_label)
np.save("./data/val_label.npy",val_label)
#np.save("./data/test_label.npy",test_label)


