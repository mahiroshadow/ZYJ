from tqdm import tqdm
import pandas as pd
'''
地址修改
'''
def data_convert(num:int,prefix:str,path:str):
    df=pd.read_csv(path,sep=' ',header=None,names=["x1","x2","x3"],skiprows=0)
    df["x1"]=df["x1"].apply(lambda x:prefix+x[num:])
    df.to_csv(path,index=False,header=None,sep=' ')

if __name__=='__main__':
    subs=26
    num=len("D:/yanyan/deeplearning/micro_expression/DATASETS/CASME2/")
    prefix="./"
    print("data process...")
    for sub in tqdm(range(subs)):
        sub+=1
        path=f"../data/MEGC2019/v_cde_flow/MEGC2019_all/sub{sub if sub>=10 else '0'+str(sub)}_test.txt"
        data_convert(num=num,prefix=prefix,path=path)