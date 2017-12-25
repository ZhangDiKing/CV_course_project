#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import cv2
class StructureFromMotion(object):
    def __init__(path):
        '''
        :para path: the path of the data file
        '''
        matfile =path+'M1.mat'
        data    = sio.loadmat(matfile)
        T,n_f=30,10
        #W is the 2d position of the featured points of different frame in continued time
        W       =data['M']
        
        #svd decomposition of W and make rank(W)=3
        U, s, V = np.linalg.svd(W)
        s_3=np.diag(s[0:3])
        U_3=U[:,0:3]
        V_3=np.transpose(V)[:,0:3]
        D_half=np.sqrt(s_3)
        R_1=U_3.dot(D_half)
        S_1=D_half.dot(np.transpose(V_3))
            
        M=np.zeros((3*T,6))
        b=np.zeros((3*T,1))
        for i in range(T):
            R_i1=R_1[2*i,:]
            R_i2=R_1[2*i+1,:]
            '''
            A=[a b c
               b d e
               c e f]
            v=[a b c d e f]^t
            M*v=b
            '''
            M[3*i,0]=R_i1[0]*R_i1[0]
            M[3*i,1]=R_i1[0]*R_i1[1]*2
            M[3*i,2]=R_i1[0]*R_i1[2]*2
            M[3*i,3]=R_i1[1]*R_i1[1]
            M[3*i,4]=R_i1[1]*R_i1[2]*2
            M[3*i,5]=R_i1[2]*R_i1[2]
            
            M[3*i+1,0]=R_i2[0]*R_i2[0]
            M[3*i+1,1]=R_i2[0]*R_i2[1]*2
            M[3*i+1,2]=R_i2[0]*R_i2[2]*2
            M[3*i+1,3]=R_i2[1]*R_i2[1]
            M[3*i+1,4]=R_i2[1]*R_i2[2]*2
            M[3*i+1,5]=R_i2[2]*R_i2[2]
            
            M[3*i+2,0]=R_i2[0]*R_i1[0]
            M[3*i+2,1]=R_i2[0]*R_i1[1]+R_i1[0]*R_i2[1]
            M[3*i+2,2]=R_i2[0]*R_i1[2]+R_i1[0]*R_i2[2]
            M[3*i+2,3]=R_i2[1]*R_i1[1]
            M[3*i+2,4]=R_i2[1]*R_i1[2]+R_i1[1]*R_i2[2]
            M[3*i+2,5]=R_i2[2]*R_i1[2]
            
            b[3*i,0]=1
            b[3*i+1,0]=1
        #get v
        v=np.linalg.pinv(M).dot(b)
        
        #recover A
        A=np.zeros((3,3))
        A[0,0]=v[0]
        A[0,1]=A[1,0]=v[1]
        A[0,2]=A[2,0]=v[2]
        A[1,1]=v[3]
        A[1,2]=A[2,1]=v[4]
        A[2,2]=v[5]
        
        #svd decomposition of A to get R and S
        U_A, s_A, V_A = np.linalg.svd(A)
        Q=U_A.dot(np.sqrt(np.diag(s_A)))
        R=R_1.dot(Q)
        S=np.linalg.inv(Q).dot(S_1)
        
        #plot and save R,S as videos
        self.rotationVideoWrite(R,T)
        self.structureVideoWrite(S,path,n_f)
        
    def structureVideoWrite(self,S,path,n_f):
        '''
        :para S: the 2d posiiton of the object
        :para path: the path of the data file
        :para n_f: number of feature points
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=10., azim=0)
        ax.scatter(S[0,:], S[1,:], S[2,:], marker="*")
        ax.plot(S[0,:], S[1,:], S[2,:], color = 'r')
        for i in range(n_f):
            ax.text(S[0,i], S[1,i], S[2,i], str(i+1))
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        fig.savefig(path+'movie.jpg',dpi=fig.dpi)
        frame = cv2.imread(path+'movie.jpg')
        height,width,ch=frame.shape
        video = cv2.VideoWriter(path+'video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10,(width,height))
        
        #rotation of the viewing angle
        for i in range(0,180,5):
            ax.view_init(elev=0., azim=i)
            fig.savefig(path+'movie.jpg',dpi=fig.dpi)
            frame = cv2.imread(path+'movie.jpg')
            video.write(frame)
        for i in range(0,181,5):
            ax.view_init(elev=i, azim=175)
            fig.savefig(path+'movie.jpg',dpi=fig.dpi)
            frame = cv2.imread(path+'movie.jpg')
            video.write(frame)
        for i in range(360,-1,-5):
            ax.view_init(elev=180, azim=i)
            fig.savefig(path+'movie.jpg',dpi=fig.dpi)
            frame = cv2.imread(path+'movie.jpg')
            video.write(frame)
        for i in range(180,-1,-5):
            ax.view_init(elev=i, azim=0)
            fig.savefig(path+'movie.jpg',dpi=fig.dpi)
            frame = cv2.imread(path+'movie.jpg')
            video.write(frame)
        ax.view_init(elev=0, azim=0)
        frame = cv2.imread(path+'movie.jpg')
        video.release()
        cv2.destroyAllWindows()
        plt.close(fig)
    def plotRotation(self,R):
        '''
        :para R: the rotation matrix 2*3
        '''
        #get the third row of rotation matrix
        Ri3=np.cross(R[0,:],R[1,:])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.view_init(elev=10., azim=11)
        zero_pos=np.array([0,0,0])
        
        ax.text(R[0,0], R[0,1], R[0,2], 'r1')
        ax.quiver(zero_pos[0], zero_pos[1], zero_pos[2],\
                  R[0,0],      R[0,1],      R[0,2],\
                  arrow_length_ratio=0.2,\
                  color='r')
        
        ax.text(R[1,0], R[1,1], R[1,2], 'r2')
        ax.quiver(zero_pos[0], zero_pos[1], zero_pos[2],\
                  R[1,0],      R[1,1],      R[1,2],\
                  arrow_length_ratio=0.2,\
                  color='g')
        
        ax.text(Ri3[0], Ri3[1], Ri3[2], 'r3')
        ax.quiver(zero_pos[0], zero_pos[1], zero_pos[2],\
                  Ri3[0],      Ri3[1],      Ri3[2],\
                  arrow_length_ratio=0.2,\
                  color='b')
        
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_zlim(-1.3, 1.3)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        fig.savefig("/Users/zhangdi/Documents/course computer vision/r.jpg",dpi=fig.dpi)
        plt.close(fig)
        
    def RotationVideoWrite(self,R,T):
        '''
        :para R: the rotation matrix through time
        :para T: the total time
        '''
        self,plotRotation(R[0:2,:])
        frame = cv2.imread('/Users/zhangdi/Documents/course computer vision/r.jpg')
        height,width,ch=frame.shape
        video = cv2.VideoWriter('/Users/zhangdi/Documents/course computer vision/video2.avi',\
                                cv2.VideoWriter_fourcc('M','J','P','G'), 12,(width,height))
        
        #get the video of how the rotation matrix change over time
        for i in range(T):
            self.plotRotation(R[2*i:2*(i+1),:])
            frame = cv2.imread('/Users/zhangdi/Documents/course computer vision/r.jpg')
            video.write(frame)
        video.release()

def main():
    path='your path for W'
    SFM=StructureFromMotion(path)
main()
