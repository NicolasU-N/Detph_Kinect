#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:28:05 2020

@author: nicolas
"""

#import the necessary modules
import freenect
import cv2
import numpy as np
import math 
import matplotlib.pyplot as plt
 
#function to get RGB image from kinect
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array

"""function to get raw depth image from kinect"""
def get_raw_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(np.uint16)
    return array
 
#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return array

def rawDepthToMeters(depthValue):
    if(depthValue < 2047):
        return (1.0 / ((depthValue) * -0.0030711016 + 3.3309495161))
    return 0
 
"""    
/ Only needed to make sense of the ouput depth values from the kinect
PVector depthToWorld(int x, int y, int depthValue) {

  final double fx_d = 1.0 / 5.9421434211923247e+02;
  final double fy_d = 1.0 / 5.9104053696870778e+02;
  final double cx_d = 3.3930780975300314e+02;
  final double cy_d = 2.4273913761751615e+02;

// Drawing the result vector to give each point its three-dimensional space
  PVector result = new PVector();
  double depth =  depthLookUp[depthValue];//rawDepthToMeters(depthValue);
  result.x = (float)((x - cx_d) * depth * fx_d);
  result.y = (float)((y - cy_d) * depth * fy_d);
  result.z = (float)(depth);
  return result;
}
"""
#     We'll use a lookup table so that we don't have to repeat the math over and over
depthLookUp= np.zeros(2048);


#   Lookup table for all possible depth values (0 - 2047)
for i in range(len(depthLookUp)):
    depthLookUp[i] = rawDepthToMeters(i);


def depthToWorld( x, y, depthValue):
    fx_d = 1.0 / 5.9421434211923247e+02;
    fy_d = 1.0 / 5.9104053696870778e+02;
    cx_d = 3.3930780975300314e+02;
    cy_d = 2.4273913761751615e+02;
    
#    Drawing the result vector to give each point its three-dimensional space           
    
    depth = depthLookUp[depthValue]; #rawDepthToMeters(depthValue);
    result=float((x - cx_d) * depth * fx_d);
    result1=np.append(result,float((y - cy_d) * depth * fy_d));
    result2=np.append(result1,float(depth));       
    return result2;
    
    



if __name__ == "__main__":
    
    kernel = np.ones((5,5),np.uint8)
    vector = [];
    dis=[];
#           ---------------------IZQUIERDA------------------------------------------------------------CENTRO----------------------------------------DERECHA
    coor=[-305,-290,-275,-260,-245,-230,-215,-200,-185,-170,-155,-140,-125,-110,-95,-80,-65,-50,-35,-20,0,20,35,50,65,80,95,110,125,140,155,170,185,200,215,230,245,260,275,290,305];
    
#           COORDENADAS PLANO PROFUNDIDAD    
#    coor_depth= [];
#    
#    for i in range(20,621,15):
#        coor_depth.append(i);
        
    
    while 1:
        #get a frame from RGB camera
        frame = get_video()
        
        #get a frame from depth sensor
       # depth = get_depth()
        raw_depth = get_raw_depth()
        
        #get a frame from depth sensor
        #depth = get_depth()
        """-------------------- APPLY FILTER ---------------------"""
#        erosion=cv2.morphologyEx(depth,cv2.MORPH_OPEN,kernel,iterations=4)
        raw_erosion=cv2.morphologyEx(raw_depth,cv2.MORPH_OPEN,kernel,iterations=4)
        """-------------------- CIRCLES ---------------------"""
        img_c = cv2.circle(frame,(320,240),3,(0,255,0),-1);        
        img_c1 = cv2.circle(img_c,(335,240),3,(0,0,255),-1);
        img_c2 = cv2.circle(img_c1,(350,240),3,(0,0,255),-1);
        img_c3 = cv2.circle(img_c2,(365,240),3,(0,0,255),-1);
        img_c4 = cv2.circle(img_c3,(380,240),3,(0,0,255),-1);
        img_c5 = cv2.circle(img_c4,(395,240),3,(0,0,255),-1);
        img_c6 = cv2.circle(img_c5,(410,240),3,(0,0,255),-1);
        img_c7 = cv2.circle(img_c6,(425,240),3,(0,0,255),-1);
        img_c8 = cv2.circle(img_c7,(440,240),3,(0,0,255),-1);
        img_c9 = cv2.circle(img_c8,(455,240),3,(0,0,255),-1);
        img_c10 = cv2.circle(img_c9,(470,240),3,(0,0,255),-1);
        img_c11 = cv2.circle(img_c10,(485,240),3,(0,0,255),-1);
        img_c12 = cv2.circle(img_c11,(500,240),3,(0,0,255),-1);
        img_c13 = cv2.circle(img_c12,(515,240),3,(0,0,255),-1);
        img_c14 = cv2.circle(img_c13,(530,240),3,(0,0,255),-1);
        img_c15 = cv2.circle(img_c14,(545,240),3,(0,0,255),-1);
        img_c16 = cv2.circle(img_c15,(560,240),3,(0,0,255),-1);
        img_c17 = cv2.circle(img_c16,(575,240),3,(0,0,255),-1);
        img_c18 = cv2.circle(img_c17,(590,240),3,(0,0,255),-1);
        img_c19 = cv2.circle(img_c18,(605,240),3,(0,0,255),-1);
        img_c20 = cv2.circle(img_c19,(620,240),3,(0,0,255),-1);
        
        img_c21 = cv2.circle(img_c20,(305,240),3,(0,0,255),-1);
        img_c22 = cv2.circle(img_c21,(290,240),3,(0,0,255),-1);
        img_c23 = cv2.circle(img_c22,(275,240),3,(0,0,255),-1);
        img_c24 = cv2.circle(img_c23,(260,240),3,(0,0,255),-1);
        img_c25 = cv2.circle(img_c24,(245,240),3,(0,0,255),-1);
        img_c26 = cv2.circle(img_c25,(230,240),3,(0,0,255),-1);
        img_c27 = cv2.circle(img_c26,(215,240),3,(0,0,255),-1);
        img_c28 = cv2.circle(img_c27,(200,240),3,(0,0,255),-1);
        img_c29 = cv2.circle(img_c28,(185,240),3,(0,0,255),-1);
        img_c30 = cv2.circle(img_c29,(170,240),3,(0,0,255),-1);
        img_c31 = cv2.circle(img_c30,(155,240),3,(0,0,255),-1);
        img_c32 = cv2.circle(img_c31,(140,240),3,(0,0,255),-1);
        img_c33 = cv2.circle(img_c32,(125,240),3,(0,0,255),-1);
        img_c34 = cv2.circle(img_c33,(110,240),3,(0,0,255),-1);
        img_c35 = cv2.circle(img_c34,(95,240),3,(0,0,255),-1);
        img_c36 = cv2.circle(img_c35,(80,240),3,(0,0,255),-1);
        img_c37 = cv2.circle(img_c36,(65,240),3,(0,0,255),-1);
        img_c38 = cv2.circle(img_c37,(50,240),3,(0,0,255),-1);
        img_c39 = cv2.circle(img_c38,(35,240),3,(0,0,255),-1);
        img_c40 = cv2.circle(img_c39,(20,240),3,(255,0,0),-1);
#        

#        
        """-------------------- VECTOR ---------------------"""
        
        for i in range(20,621,15):
            profundidad=raw_erosion[240,i];
            vector.append(profundidad);            
            
        maximo =max(vector);
#        minimo =min(vector);
        index_max=vector.index(max(vector));
        
        coordinate_x = coor[index_max];
        
        
        
        """-------------------- ANGULO DE GIRO ---------------------"""
#        tetha=np.arctan(coordinate_x/(max(vector)));
        s=float(coordinate_x)/float(maximo);        
        s2=math.radians(s)
        tetha=math.atan(float(s2));
        tetha1=math.degrees(float(tetha));
        tetha2=float(tetha1)*100;
#        print(tetha2);
#        print(s);
        """-------------------- DISTANCIA ---------------------"""
#        print("Máximo: "+str(maximo)+" pos: "+str(vector.index(max(vector)))+"------ Mínimo: "+str(minimo)+" pos: "+str(vector.index(minimo)));
#        
#        for i in range(len(vector)): # IMPRIMIENDO VECTOR DISTANCIAS 
#            distance= rawDepthToMeters(float(vector[i]));
#            dis.append(distance);
#        
#        plt.plot(dis);
#        plt.show();       
        
#        z1= raw_depth[240,320]
#        z2= raw_depth[240,20]
        """--------------------MEDIR ZONA---------------------"""
        p1 = depthToWorld(320,240,raw_erosion[240,320]);
        p2 = depthToWorld(20,240,raw_erosion[240,20]);
    
        result = np.subtract(p1,p2);
        norma= np.linalg.norm(result);
        print(norma+0.1);
        
        
        
        
        #display RGB image
        cv2.imshow('RGB image',img_c40)
        #display depth image
        cv2.imshow('Depth image',raw_erosion)
 
    
        vector=[];
        dis=[];
        maximo=0;
        minimo=0;
        
        # quit program when 'esc' key is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()