#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
class Recitification(object):
    def compute_R(self,T)-> expression:
        """
        :param T: the transanction matrix between two images 
        
        :return R: the rotation matrix for the recitified left image
        """
        T     =T/np.sqrt(np.transpose(T).dot(T))
        R     =np.zeros((3,3))
        R[:,0]=-np.transpose(T)
        R[:,1]=np.array([T[1], (-1)*T[0], 0])/((T[0]**2+T[1]**2)**0.5)
        R[:,2]=np.cross(R[:,0],R[:,1])
        return R
    def recitified_points(self,w_l,w_r,T,direction, R_b,pts_2d):
        """
        :param w_l: the W matrix of the left image 
        :param w_r: the W matrix of the right image 
        :param T: the transanction matrix between two images 
        :param direction: str 'l' for left image, 'r' for right image 
        :param R_b: the transanction matrix between two images 
        :param pts_2d: the position of 2d points of the images
        
        :return pos: 2d position for the corresponding points in the recitified image
        """
        if direction=='l':
            R=self.compute_R(T)
        else:
            R_1=self.compute_R(T)
            R=np.matmul(np.transpose(R_b),R_1)
        
        #forward maping
        map_f=w_l.dot(np.transpose(R)).dot(np.linalg.inv(w_r))
        pos=map_f.dot(pts_2d)
        
        #divide the scale factor lambda
        pos=pos/pos[2,:]
        return pos
    
    def recitified_process(self,w_l,w_r,T,img,direction, R_b):
        """
        :param w_l: the W matrix of the left image 
        :param w_r: the W matrix of the right image 
        :param T: the transanction matrix between two images 
        :param direction: str 'l' for left image, 'r' for right image 
        :param R_b: the transanction matrix between two images 
        :param pts_2d: the position of 2d points of the images
        
        :return pos: 2d position for the corresponding points in the recitified image
        """
        if direction=='l':
            R=self.compute_R(T)
        else:
            R_1=self.compute_R(T)
            R=np.matmul(np.transpose(R_b),R_1)
        rows, cols, channels = img.shape
        
        #backward maping
        map_r=w_r.dot(np.transpose(np.linalg.inv(R))).dot(np.linalg.inv(w_l))
        
        #get corner points to resize
        corner=np.array([[int(cols/2)-1],\
                         [int(rows/2)-1],\
                         [1]])
        corner_recitified=np.linalg.inv(map_r).dot(corner)
        corner_recitified=np.round(corner_recitified/corner_recitified[2,:])
        min_c=int(np.min(corner_recitified[0,:]))-int(cols/2)
        min_r=int(np.min(corner_recitified[1,:]))-int(rows/2)
        max_c=int(np.max(corner_recitified[0,:]))+int(cols/2)
        max_r=int(np.max(corner_recitified[1,:]))+int(rows/2)
        
        #save parameter for future use
        para=[min_c,max_c,min_r,max_r]
        
        #initialize the recitified image
        im_recitified=np.zeros((int(max_r-min_r), int(max_c-min_c), channels))
         
        real_r,real_c,_=im_recitified.shape
        r=np.arange(int(min_r),int(max_r))
        c=np.arange(int(min_c),int(max_c))
        x,y=np.meshgrid(c,r)
        total_length=real_r*real_c
        
        #get the matrix of [c r 1]
        p=np.vstack((x.reshape((1,total_length)),y.reshape((1,total_length)),\
                                  np.ones((1,total_length))))
        
        #backmapping 
        pos=map_r.dot(p)
        pos=pos/pos[2,:]
        
        #prepare useful data
        p=np.int64(p)
        p[0,:]=p[0,:]-min_c
        p[1,:]=p[1,:]-min_r
        p_c=np.int64(np.floor(pos[0,:]))
        p_r=np.int64(np.floor(pos[1,:]))
        v  =pos[0,:]-p_c
        u  =pos[1,:]-p_r
        
        #get the index where to fill the new position
        index=np.where((0<=pos[0,:]) & (pos[0,:]<cols-1) & (0<=pos[1,:]) & (pos[1,:]<rows-1))
        
        #fill each channel with linear interpolation
        for i in range(3):
            im_recitified[p[1,index],p[0,index],i]=(1- u[index])*(1-v[index])*img[p_r[index],p_c[index],i] +\
                                               (1-v[index])*u[index]*img[p_r[index],p_c[index]+1,i] + \
                                                v[index]*(1-u[index])*img[p_r[index]+1,p_c[index],i] + \
                                                u[index]*v[index]*img[p_r[index]+1,p_c[index]+1,i]
        return np.round(im_recitified[:,:,:]),para
    
    def scale_w(self,w):
        """
        :param w: the W matrix :(w_l+w_r)/2 
        
        :return w: rescaled W so that the recitified image can be smaller
        """
        w[0,0]=w[0,0]/1.2
        w[1,1]=w[1,1]/1.2
        return w

    def read_2d_points(self,path):
        """
        :param path: the path for the txt
        
        :return pts_2d: the position of 2d points in txt
        """
        f=open(path)
        lines=f.readlines()
        N=len(lines)
        pts_2d=np.ones((3, N))
        for i in range(0,N):
            line=lines[i].split()
            pts_2d[0,i],pts_2d[1,i] = float(line[1]), float(line[2])
        f.close()
        return pts_2d
    def recitify_two_image(self,w_l,w_r,F,T,R_b,im_path):
        """
        :param w_l: the W matrix of the left image 
        :param w_r: the W matrix of the right image 
        :param T: the transanction matrix between two images
        :param F: the fundamental matrix between two images
        :param R_b: the transanction matrix between two images 
        :param im_path: the path of the images
        
        """
        
        #read image pairs and 2d points
        im_l=cv2.imread(im_path+'left_face.jpg')
        im_r=cv2.imread(im_path+'right_face.jpg')
        pts_2d_left =self.read_2d_points(im_path+"pts_left.txt")
        pts_2d_right=self.read_2d_points(im_path+"pts_right.txt")
        
        #get w
        w=0.5*(w_l+w_r)
        w=scale_w(w)
        
        #recitifiy image pairs and 2d points
        im_l_recitified,para_l=self.recitified_process(w,w_l,T,im_l,'l',R_b)
        im_r_recitified,para_r=self.recitified_process(w,w_r,T,im_r,'r',R_b)
        
        #print out the error 
        error_mat=np.abs(pos_left[1,:]-pos_right[1,:])
        error=np.mean(error_mat)
        print('parallel error=',error)
        
        #save data
        cv2.imwrite(im_path+'left_face_recitified.jpg',im_l_recitified)
        cv2.imwrite(im_path+'right_face_recitified.jpg',im_r_recitified)
        np.savetxt(im_path+'p_l.txt',np.transpose(pos_left))
        np.savetxt(im_path+'p_r.txt',np.transpose(pos_right))
        im_full=np.hstack([im_l_recitified,im_r_recitified])
        cv2.imwrite(im_path+'full_recitified.jpg',im_full[:,:,:])

        #show the epipoles on the orignal image pairs
        _,N=pts_2d_left.shape
        r,c,_=im_r.shape
        
        #the epipolar lines on the left image
        for i in range(N):
            color = tuple(np.random.randint(0,255,3).tolist())
            l=np.transpose(F).dot(pts_2d_right[:,i])
            l=np.reshape(l,(3,1))
            x0,y0 = map(int, [0, -l[2]/l[1] ])
            x1,y1 = map(int, [c, -(l[2]+l[0]*c)/l[1] ])
            cv2.line(im_l, (x0,y0), (x1,y1), color, 2)
            cv2.circle(im_l,tuple(np.int64(pts_2d_left[0:2,i])),5,color,-1)
        cv2.imwrite(im_path+'left_face_with_line.jpg',im_l)
        
        #the epipolar lines on the right image
        r,c,_=im_r.shape
        for i in range(N):
            color = tuple(np.random.randint(0,255,3).tolist())
            l=(F).dot(pts_2d_left[:,i])
            l=np.reshape(l,(3,1))
            x0,y0 = map(int, [0, -l[2]/l[1] ])
            x1,y1 = map(int, [c, -(l[2]+l[0]*c)/l[1] ])
            cv2.line(im_r, (x0,y0), (x1,y1), color, 2)
            cv2.circle(im_r,tuple(np.int64(pts_2d_right[0:2,i])),5,color,-1)
        cv2.imwrite(im_path+'right_face_with_line.jpg',im_r)
        
        #recitify 2d points on the image pairs
        pos_left=self.recitified_points(w,w_l,T,'l',R_b,pts_2d_left)
        pos_right=self.recitified_points(w,w_r,T,'r',R_b,pts_2d_right)
        
        #move the points according to the parameters
        pos_left[0,:]=pos_left[0,:]-para_l[0]
        pos_left[1,:]=pos_left[1,:]-para_l[2]
        
        #the epipolar lines on the left recitified image
        E=np.array([[0,0,0],[0,0,-1],[0,1,0]])
        F=np.transpose(np.linalg.inv(w)).dot(E).dot(np.linalg.inv(w))
        _,N=pts_2d_left.shape
        r,c,_=im_l_recitified.shape
        for i in range(N):
            color = tuple(np.random.randint(0,255,3).tolist())
            l=np.transpose(F).dot(pos_right[:,i])
            l=np.reshape(l,(3,1))
            x0,y0 = map(int, [0, -l[2]/l[1]-para_l[2] ])
            x1,y1 = map(int, [c, -(l[2]+l[0]*c)/l[1]-para_l[2] ])
            cv2.line(im_l_recitified, (x0,y0), (x1,y1), color, 2)
            cv2.circle(im_l_recitified,tuple(np.int64(pos_left[0:2,i])),5,color,-1)
        cv2.imwrite(im_path+'left_face_with_line_recitified.jpg',im_l_recitified)
        
        #the epipolar lines on the right recitified image
        _,N=pts_2d_right.shape
        r,c,_=im_r_recitified.shape
        for i in range(N):
            color = tuple(np.random.randint(0,255,3).tolist())
            l=(F).dot(pos_left[:,i])
            l=np.reshape(l,(3,1))
            x0,y0 = map(int, [0, -l[2]/l[1]-para_r[2] ])
            x1,y1 = map(int, [c, -(l[2]+l[0]*c)/l[1]-para_r[2] ])
            cv2.line(im_r_recitified, (x0,y0), (x1,y1), color, 2)
            cv2.circle(im_r_recitified,tuple(np.int64(pos_right[0:2,i])),5,color,-1)
        cv2.imwrite(im_path+'right_face_with_line_recitified.jpg',im_r_recitified)

def main():
    #setting up parameters
    w_l=np.array([[1634.6892, 0,         1055.4498],\
                  [0,         1600.2453, 753.0086],\
                  [0,         0,         1]])
    w_r=np.array([[1656.0795, 0,         1007.4288],\
                  [0,         1637.1629, 719.0341],\
                  [0,         0,         1]])
    F= np.array([[ 3.78105e-07, -9.32121e-06,  0.00889751],\
                 [-2.59749e-06,  2.48323e-06,  0.042666],\
                 [0.00338136,   -0.0301453,   -10.1607]])
    P_l=np.array([[-55.8639,  -1940.2605,  135.8357,  -145595.8959],\
                  [-856.8643, -482.6269,  -1469.9190, -45923.3761],\
                  [-0.8477,   -0.5091,     0.1490,    -122.6100]])
    P_r=np.array([[ 914.611,  -1696.21,  209.453,  -157876],\
                  [-570.564,  -909.191, -1430.08,  -60966.1],\
                  [-0.500099, -0.835448, 0.227879, -122.355]])
    R_b=np.array([[0.8825,  0.1084,  0.4574],\
                  [-0.0730, 0.9919, -0.1052],\
                  [-0.4651, 0.0601,  0.8832]])
    path="your path for two images"
    method=Recitification()
    method.recitify_two_image(w_l,w_r,F,P_l,P_r,R_b,path)
main()
