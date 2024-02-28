import cv2
import os
from tqdm import tqdm
from decimal import *

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
                if f is None or frames//1==5:
                    break
                if frames%1==0:
                    cv2.imwrite(f"../videoslice/sub{index if index>=10 else '0'+str(index)}/{file[:-4]}/{frames//1}.jpg",f)
                frames+=1


def batch_extract_frame_modify():
    prefix = '../CASME2_Compressed video/CASME2_Compressed video/CASME2_compressed/'
    print("正在视频切片:")
    for index in tqdm(range(1, 27)):
        path = prefix + f"sub{index if index >= 10 else '0' + str(index)}"
        files = os.listdir(path)
        for file in files:
            file_video_slice=[]
            video = cv2.VideoCapture(path + '/' + file)
            os.makedirs(f"../videoslice/sub{index if index >= 10 else '0' + str(index)}/{file[:-4]}")
            while True:
                _, f = video.read()
                if f is None:
                    break
                file_video_slice.append(f)
            video_slice_len=len(file_video_slice)
            stride = int(Decimal(video_slice_len / 5).quantize(Decimal(0), rounding=ROUND_HALF_UP))
            idx = 0
            nums=0
            while idx < video_slice_len:
                cv2.imwrite(f"../videoslice/sub{index if index>=10 else '0'+str(index)}/{file[:-4]}/{nums}.jpg"
                            ,file_video_slice[idx])
                nums+=1
                if nums == 5:
                    break
                if (idx + stride) < video_slice_len - 1:
                    idx += stride
                else:
                    cv2.imwrite(f"../videoslice/sub{index if index >= 10 else '0' + str(index)}/{file[:-4]}/{nums}.jpg"
                                , file_video_slice[-1])
                    break




if __name__=='__main__':
    # batch_extract_frame()
    batch_extract_frame_modify()

