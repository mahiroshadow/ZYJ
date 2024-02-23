import cv2
import pandas as pd
import numpy as np



def pol2cart(rho,phi):#从极坐标转换为笛卡尔坐标，来计算光流应变
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x,y)
def computeStrain(u,v):
    u_x = u - pd.DataFrame(u).shift(-1,axis=1)
    v_y = v - pd.DataFrame(v).shift(-1,axis=0)
    u_y = u - pd.DataFrame(u).shift(-1,axis=0)
    v_x = v - pd.DataFrame(v).shift(-1,axis=1)
    os = np.array(np.sqrt(u_x**2 + v_y**2 + 1/2 * (u_y+v_x)**2).ffill(axis=1).ffill(axis=0))

    return os

img1 = cv2.imread("..//videoslice//sub01//EP02_01f//0.jpg")
img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)


img2 = cv2.imread("..//videoslice//sub01//EP02_01f//1.jpg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('img2_gray', img2_gray)
    #cv2.waitKey(0)

#optical_flow = cv2.DualTVL1OpticalFlow_create()
optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
flow = optical_flow.calc(img1_gray, img2_gray,None)
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


cv2.imwrite('1.jpg', final * 255)