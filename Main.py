import torch
import torch.nn as nn
import argparse
import random
import Metrics as metrics
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from Datasets import IRNN2023
from module.model2 import IRNN
from torch.optim import Adam
from torchvision import transforms
from util.loss import INFONCELoss

parser=argparse.ArgumentParser()
parser.add_argument("--epoches",type=int,default=60)
parser.add_argument("--device",type=str,default=("cuda:0" if torch.cuda.is_available() else "cpu"))
parser.add_argument("--batch_size",type=int,default=60)
parser.add_argument("--lr",type=float,default=1e-4)
parser.add_argument("--weight_decay",type=float,default=5e-4)
parser.add_argument("--h_RNN_layers",type=int,default=1)
parser.add_argument("--drop_p",type=float,default=0.4)
parser.add_argument("--num_classes",type=int,default=3)
parser.add_argument("--ratio",type=float,default=0.1)
arg=parser.parse_args()

random.seed(1)
torch.manual_seed(1)

def train():
    pred_all_tensor = torch.Tensor([])
    ground_truth_all_tensor = torch.Tensor([])
    acc_all=0.0
    transform=transforms.Compose([
        transforms.Resize([64,64]),
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
        loss_all=[]
        epoch_acc=[]
        for epoch in range(arg.epoches):
            total_loss=0.0
            running_corrects=0.0
            for images,labels,fin in dataloader:
                images,labels,fin=images.to(arg.device),labels.to(arg.device),fin.to(arg.device)
                optim.zero_grad()
                out1,out2,output=model(images,fin)
                loss_pred=criterion(output,labels)
                loss_sim=INFONCELoss(out1,out2)
                loss=loss_pred+arg.ratio*loss_sim
                loss.backward()
                optim.step()
                total_loss+=loss.item()
                _, predicted = torch.max(output, 1)
                running_corrects += torch.sum(predicted == labels)
            epoch_acc.append(running_corrects.double().cpu() / len(dataloader.dataset))
            loss_all.append(total_loss/len(dataloader))
            print(f"[epoch{epoch}_loss]:{total_loss/len(dataloader)}")
        x=[i for i in range(1,arg.epoches+1)]
        plt.plot(x,loss_all,color='black',label='loss')
        plt.plot(x,epoch_acc,color='green',label='acc')
        plt.legend(["loss","acc"])
        plt.savefig(f"data/pic/sub{idx}_loss_acc.png")
        plt.cla()
        acc,pred_all_tensor,ground_truth_all_tensor=evaluate(model=model,pth=f"data/2023/sub{idx}_test.csv",
                                                          transform=transform,pred_all_tensor=pred_all_tensor,ground_truth_all_tensor=ground_truth_all_tensor)
        print(f"sub{idx}_acc:{acc}")
        acc_all+=acc
        torch.save(model,f"./data/model_path/model{idx}.pth")
    eval_acc = metrics.accuracy()
    eval_f1 = metrics.f1score()
    acc_w, acc_uw = eval_acc.eval(pred_all_tensor,ground_truth_all_tensor)
    f1_w, f1_uw = eval_f1.eval(pred_all_tensor,ground_truth_all_tensor)
    # 添加log日志
    pd.DataFrame({"train_date":[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],"acc":[acc_w],"uar":[acc_uw],"uf1":[f1_uw]}).to_csv("log.csv",mode='a',index=False,header=False)
    print("The dataset has the ACC:{:.4f} and UAR and UF1:{:.4f} and {:.4f}".format(acc_w, acc_uw, f1_uw))

def evaluate(model,pth,transform,pred_all_tensor,ground_truth_all_tensor):
    model.eval()
    dataset = IRNN2023(pth=pth, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=arg.batch_size, shuffle=True)
    ground_truth=[]
    predicts=[]
    for images, labels,fin in dataloader:
        images,fin= images.to(arg.device),fin.to(arg.device)
        out1,out2,output=model(images,fin)
        _,predict=torch.max(output,dim=1)
        predicts.extend(predict.tolist())
        ground_truth.extend(labels.tolist())
    preds_tensor = torch.Tensor(predicts)
    labels_tensor= torch.Tensor(ground_truth)
    pred_all_tensor = torch.cat([pred_all_tensor,preds_tensor.cpu()],dim=0)
    ground_truth_all_tensor = torch.cat([ground_truth_all_tensor,labels_tensor.cpu()],dim=0)
    acc = torch.sum(preds_tensor == labels_tensor).double() / len(predicts)
    return acc,pred_all_tensor,ground_truth_all_tensor

if __name__=='__main__':
    train()


