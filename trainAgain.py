import torch
import numpy as np
from ImageDataset import MyDataset
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

def load_data(batch_size):

    train_images=np.load("./data/train_images.npy")
    val_images=np.load("./data/val_images.npy")
    #test_images=np.load("./data/test_images.npy")

    train_label=np.load("./data/train_label.npy")
    val_label =np.load("./data/val_label.npy")
    #test_label =np.load("./data/test_label.npy")


    train_dataset=MyDataset(train_images,train_label)
    val_dataset=MyDataset(val_images,val_label)
    #test_dataset = MyDataset(test_images, test_label)
    train_loader=DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_dataset, batch_size=batch_size,shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader#, test_loader

def load_model(lr,func,model):


    loss_fnc = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0.0001)

    return loss_fnc, optimizer

def evaluate(model, data_loader):
    total_corr = 0
    total=0
    for i,batch in enumerate(data_loader):
        images, label = batch
        images = images.float()
        prediction = model(images)
        corr = (prediction > 0.5).squeeze().long() == label.long()
        total_corr += int(corr.sum())
        total+=len(label)

    return total_corr/total

def main():
    model = torch.load("./saved_models/9605.pt")
    torch.manual_seed(1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--eval_every', type=int)
    parser.add_argument('--function',type=str)
    args = parser.parse_args()


    # train_loader, val_loader, test_loader = load_data(args.batch_size)
    train_loader, val_loader = load_data(args.batch_size)
    loss_fnc, optimizer = load_model(args.lr, args.function,model)

    N=0
    val_accuracy_list = []
    epoch_list = []


    for epoch in range(args.epochs):

        accum_loss = 0.0
        total_corr = 0
        total_predictions = 0

        for i, batch in enumerate(train_loader):

            images, label =batch
            optimizer.zero_grad()
            images = images.float()
            predictions = model(images)
            #print(predictions)
            batch_loss = loss_fnc(predictions, label.float())
            accum_loss += batch_loss
            batch_loss.backward()
            optimizer.step()

            corr = (predictions > 0.5).squeeze().long() == label.long()
            #print('pred:',(predictions > 0.5).squeeze().long())
            #print('label:',label.long())
            total_corr += int(corr.sum())
            total_predictions += len(label)
            print("batch ", i)

        N += 1

        if (N % (args.eval_every) == 0):

            epoch_list.append(epoch)
            if len(val_accuracy_list) > 0:
                best_last_time = max(val_accuracy_list)
            train_accuracy = total_corr/total_predictions
            val_accuracy = evaluate(model, val_loader)
            if len(val_accuracy_list) > 0 and val_accuracy > best_last_time:
                torch.save(model, "./saved_models/model_lr_{}_bs_{}_epochs_{}_function_{}_new_2.pt".format(args.lr,args.batch_size,args.epochs,args.function))
            val_accuracy_list.append(val_accuracy)
            #test_accuracy = evaluate(model,test_loader)
            print("Epoch {}: train accuracy: {}, validation accuracy: {}".format(epoch,train_accuracy, val_accuracy))
    print("highest validation accuracy: {}, at epoch {}".format(max(val_accuracy_list), epoch_list[val_accuracy_list.index(max(val_accuracy_list))]))
    #print("test accuracy: {}".format(test_accuracy))

if __name__ == "__main__":

    main()