import cv2
import os
from tqdm import tqdm


per_frame_save=2

def batch_extract_frame():
    prefix='../CASME2_Compressed video/CASME2_Compressed video/CASME2_compressed/'
    print("正在视频切片:")
    for index in tqdm(range(1,27)):
        path=prefix+f"sub{index if index>=10 else '0'+str(index)}"
        files=os.listdir(path)
        for file in files:
            frames=0
            video=cv2.VideoCapture(path+'/'+file)
            os.makedirs(f"../videoslice/sub{index if index>=10 else '0'+str(index)}/{file[:-4]}")
            while True:
                _,f=video.read()
                if f is None or frames//2==5:
                    break
                if frames%2==0:
                    cv2.imwrite(f"../videoslice/sub{index if index>=10 else '0'+str(index)}/{file[:-4]}/{frames//2}.jpg",f)
                frames+=1




if __name__=='__main__':
    batch_extract_frame()
