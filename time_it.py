#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 02:34:16 2019

@author: nicolas
"""


#import the necessary modules
import freenect
import cv2
import numpy as np
import time
from timeit import timeit
#import matplotlib.pyplot as plt
 
threshold = 250
current_depth = 630
i=0

#function to get RGB image from kinect
#def get_video():
#    array,_ = freenect.sync_get_video()
#    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
#    return array
def f1():
    global raw_depth
    raw_depth = get_raw_depth()

def f2():
    global depth_thr
    depth_thr = 255 * np.logical_and(raw_depth >= current_depth - threshold,
                                     raw_depth <= current_depth + threshold)
    depth_thr = depth_thr.astype(np.uint8)
    
def f3():
    global crop_img_izq 
    global crop_img_cen 
    global crop_img_der 
    crop_img_izq =  depth_thr[0:480, 0:213]
    cv2.imshow(winizq,crop_img_izq)
    crop_img_cen =  depth_thr[0:480, 213:426]
    cv2.imshow(wincen,crop_img_cen)
    crop_img_der =  depth_thr[0:480, 426:640]
    cv2.imshow(winder,crop_img_der)
    
    
"""function to read frames"""
def read_frame():
    for i in range(480):
        for j in range(213):
            if crop_img_cen[i][j]!=0:                       
#                    cv2.putText(img, 'OpenCV Python', (5,260), font, 3, (255,255,255), 1, cv2.LINE_AA)
#                    cv2.imshow("COMANDO", img)
                print("Girar en un sentido")
        
            if crop_img_der[i][j]!=0:                        
#                    cv2.putText(img, 'OpenCV Python', (5,260), font, 3, (255,255,255), 1, cv2.LINE_AA)
#                    cv2.imshow("COMANDO", img)
                print("girar izquierda")
        
            if crop_img_izq[i][j]!=0:                        
#                    cv2.putText(img, 'OpenCV Python', (5,260), font, 3, (255,255,255), 1, cv2.LINE_AA)
#                    cv2.imshow("COMANDO", img)
                print("girar derecha")
 

"""function to get raw depth image from kinect"""
def get_raw_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(np.uint16)
    return array

"""function to get raw depth image from kinect"""
def get_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return array

def rawDepthToMeters(depthValue):
    if(depthValue < 2047):
        return (1.0 / ((depthValue) * -0.0030711016 + 3.3309495161))
    return 0
 
if __name__ == "__main__":       
#    winname='Depth image'
#    winname1='Depth image Filter'
    winizq='FRAME IZQ'
    wincen='FRAME CEN'
    winder='FRAME DER'
    kernel = np.ones((5,5),np.uint8)
    img = np.zeros((600, 800, 3), np.uint8)
    font = cv2.FONT_HERSHEY_PLAIN 
    cv2.namedWindow(winizq)
    cv2.moveWindow(winizq,10,50)
    cv2.namedWindow(wincen)
    cv2.moveWindow(wincen,300,50)
    cv2.namedWindow(winder)
    cv2.moveWindow(winder,570,50)
    
    #get a frame from RGB camera
    #     frame = get_video()
    while 1:
        #-----------get a frame from depth sensor        
    #        depth = get_depth()
        f1()
        #print("-->ADQUISICION IMAGEN:")
        #print(timeit(get_raw_depth()))
        """-------------------- APPLY FILTER ---------------------"""
        #erosion=cv2.erode(depth,kernel,iterations=3)
        #erosion=cv2.morphologyEx(depth,cv2.MORPH_OPEN,kernel,iterations=4)
        """-------------------- SEGMENTACION ---------------------"""
        print("-->SEGMENTACION:")
        t1=timeit(f2)
        print(t1)
        
        
    	    #	Median filter	
        #medianblur= cv2.medianBlur(depth,27)
        
        
        #----------Dibujando ellipse
        
    #        overlay = erosion.copy()
    #        output = erosion.copy()
    #        circle = cv2.circle(overlay,(255,255), 50, (0,0,255), -1)
    #        	#apply the overlay
    #        cv2.addWeighted(overlay, 1, output, 1 - 1,0, output)
    #        
    #        cv2.imshow(winname1,output)
        
        #display RGB image
    	    #       cv2.imshow('RGB image',frame)
        
        
    #       ---------- RECORTANDO IMAGEN --------------------------
        # NOTE: its img[y: y + h, x: x + w] 
        print("--> RECORTANDO FRAMES:")
        t3=timeit(f3)
        print(t3)
                
        #imgplot = plt.imshow(crop_img_izq)
        #plt.colorbar()
        #plt.show()
        
        
        
        
    #        imgplot = plt.imshow(crop_img_cen)
    #        plt.colorbar()
    #        plt.show()
        
        
        
    
        """ LECTURA DE FRAMES """
        print("--> LECTURA DE FRAMES:")
        t9=timeit(read_frame())
            
                           
        
    #       --------- CALCULANDO PROFUNDIDAD PROMEDIO ----------
    #        depthmm = raw_depth[1,1];
    #        millimeters = rawDepthToMeters(depthmm)+0.0365
    #        print("meters: "+str(millimeters));        
    
        
        #--------- LEER IMAGEN COMPLETA
    #        for i in range(len(image_data2)):
    #    for j in range(len(image_data1[0])):
    #        if image_data1[i][j]==image_data2[i][j]:
    #          imagen_negra[i][j]=[0,0,0]
    #        elif image_data1[i][j]>image_data2[i][j]:
    #          imagen_negra[i][j]=[0.5,0.5,0.5]
    #        else:
    #          imagen_negra[i][j]=[1,1,1]
                
        
        #--------display depth image
        #cv2.imshow(winname1,erosion)
        #cv2.imshow(winname,depth)      
    #        print(crop_img_der.shape)
        
        # quit program when 'esc' key is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            
            cv2.destroyAllWindows()
        
