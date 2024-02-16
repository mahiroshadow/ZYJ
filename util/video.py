import cv2

def extract_frame(video_path):
    video=cv2.VideoCapture("C://Users//ASUS//Desktop//测试.mp4")
    r,f=video.read()
    print(f)


if __name__=='__main__':
    extract_frame()