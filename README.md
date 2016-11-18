# -*- coding: utf-8 -*-
# BackgroundSubtractorMOG()方法提取前景
import argparse
import cv2
import numpy as np
# 创建参数解析器并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
cap = cv2.VideoCapture('8.avi')
# 背景对象
fgbg = cv2.BackgroundSubtractorMOG()
while True:
    ret, frame = cap.read()
    # apply()方法得到前景的掩模
    fgmask = fgbg.apply(frame)
    th=cv2.threshold(fgmask.copy(),244,255,cv2.THRESH_BINARY)[1]
    dilated=cv2.dilate(th,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=2)
    (contours, _)=cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # 遍历轮廓
    for c in contours:
        if cv2.contourArea(c)>500:
            #  boundingRect用一个最小的矩形，把找到的形状包裹起来！
            (x,y,w,h)=cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


    cv2.imshow('mog', fgmask)
    cv2.imshow('thresh', th)
    cv2.imshow('detection', frame)
    # # 键盘绑定函数，时间尺度是毫秒级别的。如果设置参数为0，他将会无限期的等待键盘的输入
    k = cv2.waitKey(30) & 0xff
    if k == 27 :
        break
cap.release()
cv2.destroyAllWindows()
