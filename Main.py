import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from Datasets import IRNN2023
from module.model2 import IRNN
from torch.optim import Adam
from torchvision import transforms

parser=argparse.ArgumentParser()
parser.add_argument("--epoches",type=int,default=60)
parser.add_argument("--device",type=str,default=("cuda:0" if torch.cuda.is_available() else "cpu"))
parser.add_argument("--batch_size",type=int,default=60)
parser.add_argument("--lr",type=float,default=1e-4)
parser.add_argument("--weight_decay",type=float,default=5e-4)
parser.add_argument("--h_RNN_layers",type=int,default=1)
parser.add_argument("--drop_p",type=float,default=0.4)
parser.add_argument("--num_classes",type=int,default=3)
arg=parser.parse_args()

def train():
    acc_all=0.0
    transform=transforms.Compose([
        transforms.Resize([28,28]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    criterion= nn.CrossEntropyLoss()
    for idx in range(1,27):
        model=IRNN(arg.h_RNN_layers,arg.drop_p,arg.num_classes).to(arg.device)
        model.train()
        optim=Adam(model.parameters(),lr=arg.lr,weight_decay=arg.weight_decay)
        dataset=IRNN2023(pth=f"data/2023/sub{idx}_train.csv",transform=transform)
        dataloader=DataLoader(dataset=dataset,batch_size=arg.batch_size,shuffle=True)
        for epoch in range(arg.epoches):
            total_loss=0.0
            for images,labels in dataloader:
                images,labels=images.to(arg.device),labels.to(arg.device)
                optim.zero_grad()
                _,output=model(images,None)
                loss=criterion(output,labels)
                loss.backward()
                optim.step()
                total_loss+=loss.item()
            print(f"[epoch{epoch}_loss]:{total_loss/len(dataloader)}")
        acc=evaluate(model=model,pth=f"data/2023/sub{idx}_test.csv",transform=transform)
        print(f"sub{idx}_acc:{acc}")
        acc_all+=acc
        torch.save(model,f"./data/model_path/model{idx}.pth")
    print(f"all_acc:{acc_all}")

def evaluate(model,pth,transform):
    model.eval()
    dataset = IRNN2023(pth=pth, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=arg.batch_size, shuffle=True)
    ground_truth=[]
    predicts=[]
    for images, labels in dataloader:
        images= images.to(arg.device)
        _,output=model(images,None)
        _,predict=torch.max(output,dim=1)
        predicts.extend(predict.tolist())
        ground_truth.extend(labels.tolist())
    preds_tensor = torch.Tensor(predicts)
    labels_tensor= torch.Tensor(ground_truth)
    acc = torch.sum(preds_tensor == labels_tensor).double() / len(predicts)
    return acc

if __name__=='__main__':
    train()


