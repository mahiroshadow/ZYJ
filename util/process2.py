import os
import pandas as pd
from matplotlib import pyplot as plt



if __name__=='__main__':
    # for index in range(1,27):
    #     df=pd.read_csv(f"../data/MEGC2019/v_cde_flow/MEGC2019_all/sub{index if index>=10 else '0'+str(index)}_test.txt",sep=" ",names=["pth","class_label","db_label"])
    #     class_label=[]
    #     db_label=[]
    #     pth=[]
    #     for idx in range(len(df)):
    #         p = df["pth"][idx][25:-8]
    #         pth.append("./videoslice/"+p[:6]+p[9:])
    #         class_label.append(df["class_label"][idx])
    #         db_label.append(df["db_label"][idx])
    #     pd.DataFrame({"pth":pth,"class_label":class_label,"db_label":db_label}).to_csv(f"sub{index}_test.csv",index=False)
    x=[1,2,3,4,5]
    y=[1.2,0.98,0.76,0.54,0.55]
    y1=[1,2,2,3,4]
    plt.plot(x,y,color='black',label='loss')
    plt.plot(x,y1,color='green',label='acc')
    plt.legend(["loss","acc"])
    plt.title("sub_xxx_loss&acc")
    plt.savefig("xxx.png")