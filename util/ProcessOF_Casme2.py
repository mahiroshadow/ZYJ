# 作者：张鑫
# 时间：2022/10/23 10:50
#准备这个文件只是用于获取指定图片的路径，但是写着写着就写一块了
#生成的光流图存到CASME2-OpticalFlow（CASME2-OpticalFlow-num-eyeMask）（CASME2-OpticalFlow-num）文件夹下
import os
import xlrd
import pandas as pd
import cv2
import numpy as np
import dlib
def pol2cart(rho,phi):#从极坐标转换为笛卡尔坐标，来计算光流应变
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x,y)
def computeStrain(u,v):
    u_x = u - pd.DataFrame(u).shift(-1,axis=1)
    v_y = v - pd.DataFrame(v).shift(-1,axis=0)
    u_y = u - pd.DataFrame(u).shift(-1,axis=0)
    v_x = v - pd.DataFrame(v).shift(-1,axis=1)
    os = np.array(np.sqrt(u_x**2 + v_y**2 + 1/2 * (u_y+v_x)**2).ffill(1).ffill(0))
    return os

filename = 'CASME2-coding-train.xlsx'
book_wind = xlrd.open_workbook(filename=filename)
wind_sheet1 = book_wind.sheets()[0]
Subjects = wind_sheet1.col_values(0,1) #subject string型 01 02
Filename = wind_sheet1.col_values(1, 1)  # Filename EP02_04f
OnsetFrames = wind_sheet1.col_values(3,1) #起始帧
ApexFrames = wind_sheet1.col_values(4, 1) #峰值帧
Emotion = wind_sheet1.col_values(8, 1)  # 情绪，类型 fear
ClassInfor = wind_sheet1.col_values(9, 1)  # 对应的类别 消极0  积极1  惊讶2
print(type(ClassInfor[0]))

#获得sub01这类的名字
subName = []
for i in range(0,len(Subjects)):
    subName.append("sub"+Subjects[i])
#现在subName+'/'+Filename,就是sub01/EP02_01f 这个目录下

ApexsStr,OnsetsStr,ApexsInt,OnsetsInt = [],[],[],[] #读取的是float类型，需要转换为int型->string型
for i in range(0,len(OnsetFrames)):
    OnsetsStr.append(str(int(OnsetFrames[i])))
    ApexsStr.append(str(int(ApexFrames[i])))
    ApexsInt.append(int(ApexFrames[i]))
    OnsetsInt.append(int(OnsetFrames[i]))

file_path = 'CASME2-Cropped/'
OF_path = 'CASME2-OpticalFlow-num/'# CASME2-OpticalFlow/
#创建CASME2-OpticalFlow/sub01这类文件夹
subs = os.listdir(file_path)
for i in range(0,len(subs)):
    if not os.path.exists(OF_path+subs[i]):
        os.makedirs(OF_path+subs[i])

#图片是以reg_img01.jpg 这种命名的
Onsets,Apexs = [],[]
#imgs = []
for i in range(0,len(subName)):
    print(i)
    imgs = os.listdir(file_path+subName[i]+'/'+Filename[i])
    #现在有个问题，imgs读取的图片顺序不对，比如reg_img99.jpg排在reg_img100.jpg后面，它是根据第一个数字进行排序的这样提取的峰值帧和起始帧不对
    #而且还出错'numpy.ndarray' object has no attribute 'listdir'
    #发现错误在哪了，下面的OS = computeStrain(u, v)，返回的os与os库重名了，于是改成OS了
    #print(imgs[0],imgs[ApexsInt[i]-OnsetsInt[i]]) #reg_img46.jpg reg_img59.jpg
    #imgs[0],imgs[ApexsInt[i]-OnsetsInt[i]]为起始帧和峰值帧
    #上面一行注释不对了
    img_onset,img_apex = '',''
    for j in range(0,len(imgs)):
        name,jpg = imgs[j].split('.')#reg_img46 jpg
        num = name[7:] # str 46
        if num==OnsetsStr[i]:
            img_onset = imgs[j]
        if num==ApexsStr[i]:
            img_apex = imgs[j]
    print(img_onset,img_apex)
    Onset = file_path+subName[i]+'/'+Filename[i]+'/'+img_onset
    Apex = file_path+subName[i]+'/'+Filename[i]+'/'+img_apex
    img1 = cv2.imread(Onset)
    img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('img1_gray', img1_gray)
    #cv2.waitKey(0)

    img2 = cv2.imread(Apex)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('img2_gray', img2_gray)
    #cv2.waitKey(0)

    #optical_flow = cv2.DualTVL1OpticalFlow_create()
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(img1_gray, img2_gray, None)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])


    u, v = pol2cart(magnitude, angle)
    OS = computeStrain(u, v)
    u = cv2.resize(u, (224, 224))
    v = cv2.resize(v, (224, 224))
    OS = cv2.resize(OS, (224, 224))
    final = np.zeros((224,224,3))
    final[:, :, 0] = u
    final[:, :, 1] = v
    final[:, :, 2] = OS
    #cv2.imshow('final', final)
    #cv2.waitKey(0)
    '''
    #下面代码是为了切除眼睛区域
    # 消除全局头部运动用鼻子区域
    x61, y61 = 0, 0  # nose landmark
    # 眼睛部分
    x11, y11, x12, y12, x13, y13, x14, y14, x15, y15, x16, y16 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    x21, y21, x22, y22, x23, y23, x24, y24, x25, y25, x26, y26 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    faceDetector = dlib.get_frontal_face_detector()
    landmarkpred = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = faceDetector(img1_gray, 1)
    for face in faces:
        landmarks = landmarkpred(img1, face)
        # 鼻子
        x61 = landmarks.part(28).x
        y61 = landmarks.part(28).y
        # 左眼
        x11 = max(landmarks.part(36).x - 15, 0)
        y11 = landmarks.part(36).y
        x12 = landmarks.part(37).x
        y12 = max(landmarks.part(37).y - 15, 0)
        x13 = landmarks.part(38).x
        y13 = max(landmarks.part(38).y - 15, 0)
        x14 = min(landmarks.part(39).x + 15, 224)  # 256是像素(resize)最大为256
        y14 = landmarks.part(39).y
        x15 = landmarks.part(40).x
        y15 = min(landmarks.part(40).y + 15, 224)
        x16 = landmarks.part(41).x
        y16 = min(landmarks.part(41).y + 15, 224)
        # 右眼
        x21 = max(landmarks.part(42).x - 15, 0)
        y21 = landmarks.part(42).y
        x22 = landmarks.part(43).x
        y22 = max(landmarks.part(43).y - 15, 0)
        x23 = landmarks.part(44).x
        y23 = max(landmarks.part(44).y - 15, 0)
        x24 = min(landmarks.part(45).x + 15, 224)
        y24 = landmarks.part(45).y
        x25 = landmarks.part(46).x
        y25 = min(landmarks.part(46).y + 15, 224)
        x26 = landmarks.part(47).x
        y26 = min(landmarks.part(47).y + 15, 224)

    final[:, :, 0] = abs(final[:, :, 0] - final[y61 - 5:y61 + 6, x61 - 5:x61 + 6, 0].mean())
    final[:, :, 1] = abs(final[:, :, 1] - final[y61 - 5:y61 + 6, x61 - 5:x61 + 6, 1].mean())
    final[:, :, 2] = final[:, :, 2] - final[y61 - 5:y61 + 6, x61 - 5:x61 + 6, 2].mean()

    # 遮眼
    left_eye = [(x11, y11), (x12, y12), (x13, y13), (x14, y14), (x15, y15), (x16, y16)]
    right_eye = [(x21, y21), (x22, y22), (x23, y23), (x24, y24), (x25, y25), (x26, y26)]
    cv2.fillPoly(final, [np.array(left_eye)], 0)
    cv2.fillPoly(final, [np.array(right_eye)], 0)
    '''

    cv2.imwrite(OF_path + subName[i]+'/'+Subjects[i]+'-'+Filename[i]+'-'+str(ClassInfor[i])+'.jpg', final * 255)
'''
for i in range(0,len(imgs)):
    name,jpg = imgs[i].split('.') #reg_img46 jpg
    num = name[7:] # str 46...
print(imgs[0],imgs[13])#reg_img46.jpg reg_img59.jpg
'''