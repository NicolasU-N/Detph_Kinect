#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 23:11:33 2020

@author: nicolas
"""

import freenect
import cv2
import numpy as np
import math 
import matplotlib.pyplot as plt
from linetimer import CodeTimer

"""function to get depth image from kinect"""
def get_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return array

if __name__ == "__main__":
    
    kernel = np.ones((5,5),np.uint8)
    while 1:
        
        depth = get_depth()
        
        erosion=cv2.morphologyEx(depth,cv2.MORPH_OPEN,kernel,iterations=4)
        
        n_blanco_sin=0;
        n_blanco_con=0;
        
        
        
        for i in range(640):
            for j in range(480):
                if depth[j,i]==255:
                    n_blanco_sin+=1;
                    
        for k in range(640):
            for p in range(480):
                if erosion[p,k]==255:
                    n_blanco_con+=1;
        
        
        print("Sin filtro: "+str(n_blanco_sin)+"    Con filtro: "+str(n_blanco_con));
        
        #display depth image                    
        cv2.imshow('Depth image',erosion)
        #display RGB image
        cv2.imshow('RGB image',depth)
                # quit program when 'esc' key is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()