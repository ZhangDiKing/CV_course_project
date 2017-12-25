#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#read data from file
class eight_point_method:
    def __init__(self,pts_2d_left,pts_2d_right):
        #data normalization
        Y1, T1=self.normlized_2d_points(pts_2d_left)
        Y2, T2=self.normlized_2d_points(pts_2d_right)
        
        #construct matrix A
        A = self.compute_A_matrix(Y1,Y2)
        
        #use SVD of A to compute F
        U, D, V=np.linalg.svd(A)
        e      =V[-1].reshape((3,3))
        F      =e
        
        #for the constrain of rank(F)==2
        if not np.linalg.matrix_rank(F)==2:
            print("further approximation")
            U, D, V=np.linalg.svd(F)
            D[2]   =0
            F      =np.dot(np.dot(U,np.diag(D)),V)
            
        #denormalization to recover F
        F=np.dot(np.dot(T2.T,F),T1)
        self.F=F/F[2,2]
    
    def getF(self):
        return self.F
    #data normalization
    def normlized_2d_points(self,pts_2d):
        #shift points to center
        center=np.mean(pts_2d,axis=1)
        p0=pts_2d[0,:]-center[0]
        p1=pts_2d[1,:]-center[1]
        
        #construct T matrix
        mean_dis=np.mean(np.sqrt(p0**2+p1**2))
        scale_factor=np.sqrt(2)/mean_dis
        
        T=np.eye(3)
        T[0,0]=scale_factor
        T[1,1]=scale_factor
        T[0,2]=-scale_factor*center[0]
        T[1,2]=-scale_factor*center[1]
        
        return np.dot(T,pts_2d), T

    #construct matrix A
    def compute_A_matrix(self,Y1,Y2):
        _,N=Y1.shape
        A=np.c_[Y1[0]*Y2[0], Y1[0]*Y2[1], Y1[0], Y1[1]*Y2[0],\
                Y1[1]*Y2[1], Y1[1],       Y2[0], Y2[1],\
                np.ones((N,1))]
        return A
        
def read_2d_points(path):
    f=open(path)
    lines=f.readlines()
    N=len(lines)
    pts_2d=np.ones((3, N))
    for i in range(0,N):
        line=lines[i].split()
        pts_2d[0,i],pts_2d[1,i] = float(line[1]), float(line[2])
    f.close()
    return pts_2d        

def main():
    path        ="/Users/zhangdi/Documents/course computer vision/project3/project3_data/calibration/"
    pts_2d_left =read_2d_points(path+"pts_2D_left.txt")
    pts_2d_right=read_2d_points(path+"pts_2D_right.txt")
    method=eight_point_method(pts_2d_left,pts_2d_right)
    
    F=method.getF();
    #print result and error
    print('F=',F)
    error=np.transpose(pts_2d_left).dot(F.dot(pts_2d_right))
    print('error=',np.mean(error*error))
    
    return F
F=main()
np.savetxt('/Users/zhangdi/Dropbox/computer_vision_group/F_8method.txt',F)