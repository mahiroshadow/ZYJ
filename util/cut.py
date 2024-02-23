import math
import cv2
import dlib
import numpy as np
import os

predictor_model = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_model)

def image_cut(img_pth,save_pth):
    img = cv2.imread(img_pth)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rects = detector(img_gray, 0)
    landmarks = np.array([[p.x, p.y] for p in predictor(img,rects[0]).parts()])
    le_h=landmarks[37][1]-landmarks[19][1]
    le_l=landmarks[21][0]-landmarks[17][0]
    left_eye_crop=img[int(landmarks[19][1]-0.25*le_h):int(landmarks[27][1]-0.25*le_h),
            int(landmarks[17][0]-0.25*le_l):int(landmarks[21][0]+0.25*le_l)]
    re_h=landmarks[44][1]-landmarks[24][1]
    re_l=landmarks[26][0]-landmarks[22][0]
    right_eye_crop=img[int(landmarks[24][1]-0.25*re_h):int(landmarks[44][1]-0.25*re_h),
            int(landmarks[22][0]-0.25*re_l):int(landmarks[26][0]+0.25*re_l)]
    m_h=landmarks[51][1]-landmarks[33][1]
    m_l=landmarks[54][0]-landmarks[48][0]
    mouse_crop=img[int(landmarks[51][1]-0.5*m_h):int(landmarks[57][1]+0.5*m_h),
           int(landmarks[48][0]-0.25*m_l):int(landmarks[54][0]+0.25*m_h)]
    mouse_crop=cv2.resize(mouse_crop,(mouse_crop.shape[1]*2,mouse_crop.shape[0]*2))
    width=min(left_eye_crop.shape[1]*2,right_eye_crop.shape[1]*2,mouse_crop.shape[1])
    height=min(left_eye_crop.shape[0],right_eye_crop.shape[0])+mouse_crop.shape[0]
    channel=3
    new_img=np.zeros((height,width,channel),dtype=np.uint8)
    new_img[0:height-mouse_crop.shape[0],0:left_eye_crop.shape[1],:]=left_eye_crop[0:height-mouse_crop.shape[0],:,:]
    new_img[0:height-mouse_crop.shape[0],left_eye_crop.shape[1]:width,:]=right_eye_crop[0:height-mouse_crop.shape[0],0:width-left_eye_crop.shape[1],:]
    new_img[height-mouse_crop.shape[0]:height,:,:]=mouse_crop[:,0:width,:]
    cv2.imwrite(save_pth, new_img)



if __name__=='__main__':
    prefix="../videoslice/sub"
    for idx in range(2,27):
        for (root,dirs,files) in os.walk(prefix+(str(idx) if idx>=10 else "0"+str(idx))):
            if len(files)!=0:
                for file in files:
                    image_cut(root+"\\"+file,root+"\\"+file)



