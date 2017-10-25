#%%
import scipy.io as sio
import numpy as np
import math
import cv2

#read data from file
matfile ='/Users/zhangdi/Documents/course computer vision/project1/CV1_data.mat'
data    = sio.loadmat(matfile)

Nx = data['Nx']
Ny = data['Ny']
Nz = data['Nz']
X  = data['X']
Y  = data['Y']
Z  = data['Z']

#varied parameters for project 1
R1   =np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
R2   =np.array([[0.9848, 0, 0.17360],[0, 1, 0],[-0.1736, 0, 0.9848]])
L1   =np.array([0, 0, -1])
L2   =np.array([0.5774, - 0.5774, -0.5774])
f1   =40
f2   =30
#%%
def threeD_to_twoD_porjection(R,L,f,outputname):    
    #input parameters other than R and T and f
    (N,c)=X.shape
    T    =np.array([[-14],[-71],[1000]])
    Sx   =8
    Sy   =8
    c0   =50
    r0   =50
    Alpha=0
    Beta =1
    Rho  =1
    d    =33
    
    #compute location and I
    P        =np.dot(np.array([[Sx*f, 0, c0],[0, Sy*f, r0],[0, 0, 1]]),np.hstack((R,T)))    
    real_pos =np.hstack((X,np.hstack((Y,np.hstack((Z, np.ones((N,1))))))))   
    pos_array=np.dot(P,real_pos.T)
    
    pos_array[0,:]=pos_array[0,:]/pos_array[2,:]
    pos_array[1,:]=pos_array[1,:]/pos_array[2,:]
    const_para    =Beta*Rho*math.pi/4.0*((d/f)**2)*(math.cos(Alpha)**4)
    Norm=np.hstack((Nx,np.hstack((Ny,Nz))))
    I=const_para*np.dot(L.T,Norm.T)*255
    
        
    #generate the 2d_projection array
    I.astype(np.uint8)
    (_,N)=pos_array.shape
    r    =int(round(max(pos_array[1,:])))+1
    c    =int(round(max(pos_array[0,:])))+1
    projection_2d=np.zeros([r,c],np.uint8)
    for i in range(N):
        if int(I[i])>=0 :
            projection_2d[int(round(pos_array[1,i])),int(round(pos_array[0,i]))]=int(I[i])
    
    #write image
    path='/Users/zhangdi/Documents/course computer vision/project1/'
    cv2.imwrite(path+outputname+'.jpg',projection_2d)
    return projection_2d
# %%
def threeD_to_twoD_porjection_weak(R,L,f,outputname):    
    #input parameters other than R and T and f
    (N,c)   =X.shape
    T       =np.array([[-14],[-71],[1000]])
    xyz     =np.hstack((X,np.hstack((Y,Z))))  
    R_w     =R.copy()
    zc_avg  =np.mean(np.dot(R_w[2,:],xyz.T)+T[2])
    Sx      =8
    Sy      =8
    c0      =50
    r0      =50
    Alpha   =0
    Beta    =1
    Rho     =1
    d       =33    
    
    R_w[2,:]=0
    T       =np.array([[-14],[-71],[zc_avg/f]])

    #compute location and I
    w1       =np.array([[Sx, 0, c0],[0, Sy, r0],[0, 0, 1]])
    P        =f/zc_avg*np.dot(w1,np.hstack((R_w,T))) 
    real_pos =np.hstack((X,np.hstack((Y,np.hstack((Z, np.ones((N,1))))))))   
    pos_array=np.dot(P,real_pos.T)


    const_para=Beta*Rho*math.pi/4.0*((d/f)**2)*(math.cos(Alpha)**4)
    Norm=np.hstack((Nx,np.hstack((Ny,Nz))))
    I=const_para*np.dot(L.T,Norm.T)*255


    #generate the 2d_projection array
    I.astype(np.uint8)
    (_,N)=pos_array.shape
    r    =int(round(max(pos_array[1,:])))+1
    c    =int(round(max(pos_array[0,:])))+1
    projection_2d=np.zeros([r,c],np.uint8)
    for i in range(N):
        if int(I[i])>=0 :
            projection_2d[int(round(pos_array[1,i])),int(round(pos_array[0,i]))]=int(I[i])
    
    #write image
    path='/Users/zhangdi/Documents/course computer vision/project1/'
    cv2.imwrite(path+outputname+'.jpg',projection_2d)
    return pos_array
# %%   
def threeD_to_twoD_porjection_orth(R,L,f,outputname):    
    #input parameters other than R and T and f
    (N,c)   =X.shape
    T       =np.array([[-14],[-71],[1000]])
    xyz     =np.hstack((X,np.hstack((Y,Z))))  
    R_w     =R.copy()
    zc_avg  =f
    Sx      =8
    Sy      =8
    c0      =50
    r0      =50
    Alpha   =0
    Beta    =1
    Rho     =1
    d       =33    
    
    R_w[2,:]=0
    T       =np.array([[-14],[-71],[zc_avg/f]])

    #compute location and I
    w1       =np.array([[Sx, 0, c0],[0, Sy, r0],[0, 0, 1]])
    P        =f/zc_avg*np.dot(w1,np.hstack((R_w,T)))
    real_pos =np.hstack((X,np.hstack((Y,np.hstack((Z, np.ones((N,1))))))))   
    pos_array=np.dot(P,real_pos.T)


    const_para=Beta*Rho*math.pi/4.0*((d/f)**2)*(math.cos(Alpha)**4)
    Norm=np.hstack((Nx,np.hstack((Ny,Nz))))
    I=const_para*np.dot(L.T,Norm.T)*255


    #generate the 2d_projection array
    I.astype(np.uint8)
    (_,N)=pos_array.shape
    #k=11
    k    =11
    r    =int(round(max(pos_array[1,:])))+k
    c    =int(round(max(pos_array[0,:])))+k
    r_min=int(round(min(pos_array[1,:])))-k
    c_min=int(round(min(pos_array[0,:])))-k
    print([r-r_min,c-c_min])
    projection_2d=np.zeros([r-r_min,c-c_min],np.uint8)
    for i in range(N):
        if int(I[i])>=0:
            pos_x=int(round(pos_array[1,i]))-r_min
            pos_y=int(round(pos_array[0,i]))-c_min
            projection_2d[pos_x-k:pos_x+k,pos_y-k:pos_y+k]=int(I[i])
    
    #write image
    path='/Users/zhangdi/Documents/course computer vision/project1/'
    cv2.imwrite(path+outputname+'.jpg',projection_2d)
    return pos_array
#%%
a=threeD_to_twoD_porjection(R1,L1,f1,'1')
b=threeD_to_twoD_porjection(R2,L1,f1,'2')
c=threeD_to_twoD_porjection(R1,L2,f1,'3')
d=threeD_to_twoD_porjection(R2,L2,f1,'4')
e=threeD_to_twoD_porjection(R1,L1,f2,'5')
f=threeD_to_twoD_porjection(R2,L1,f2,'6')
g=threeD_to_twoD_porjection(R1,L2,f2,'7')
h=threeD_to_twoD_porjection(R2,L2,f2,'8')
#%%
a=threeD_to_twoD_porjection_weak(R1,L1,f1,'w1')
b=threeD_to_twoD_porjection_weak(R2,L1,f1,'w2')
c=threeD_to_twoD_porjection_weak(R1,L2,f1,'w3')
d=threeD_to_twoD_porjection_weak(R2,L2,f1,'w4')
e=threeD_to_twoD_porjection_weak(R1,L1,f2,'w5')
f=threeD_to_twoD_porjection_weak(R2,L1,f2,'w6')
g=threeD_to_twoD_porjection_weak(R1,L2,f2,'w7')
h=threeD_to_twoD_porjection_weak(R2,L2,f2,'w8')
#%%
a=threeD_to_twoD_porjection_orth(R1,L1,f1,'o1')
b=threeD_to_twoD_porjection_orth(R2,L1,f1,'o2')
c=threeD_to_twoD_porjection_orth(R1,L2,f1,'o3')
d=threeD_to_twoD_porjection_orth(R2,L2,f1,'o4')
e=threeD_to_twoD_porjection_orth(R1,L1,f2,'o5')
f=threeD_to_twoD_porjection_orth(R2,L1,f2,'o6')
g=threeD_to_twoD_porjection_orth(R1,L2,f2,'o7')
h=threeD_to_twoD_porjection_orth(R2,L2,f2,'o8') 
