#Di Zhang Project 2
#%%
import numpy as np
import math
import cv2
import random
import matplotlib.pyplot as plt
#%%
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

#calculate parameters
def parameter_computation(P):
    tz=P[2,3]
    c0=P[0,0:3].dot(np.transpose(P[2,0:3]))
    r0=P[1,0:3].dot(np.transpose(P[2,0:3]))
    Sxf=np.sqrt(P[0,0:3].dot(np.transpose(P[0,0:3]))-c0*c0)
    Syf=np.sqrt(P[1,0:3].dot(np.transpose(P[1,0:3]))-r0*r0)
    #print(Sxf,Syf)
    r3=P[2,0:3]
    tx=(P[0,3]-c0*tz)/Sxf
    ty=(P[1,3]-r0*tz)/Syf
    r1=(P[0,0:3]-c0*r3)/Sxf
    r2=(P[1,0:3]-r0*r3)/Syf
    M=np.ones((3,4))
    M[0,0:3]=r1
    M[1,0:3]=r2
    M[2,0:3]=r3
    M[0,3]=tx
    M[1,3]=ty
    M[2,3]=tz
    W=np.array([[Sxf,0,c0],[0,Syf,r0],[0,0,1]])
    print('W=',W)
    print('M=',M)
    return W,M

#%% compute projection error
def projection_error_computation(P,p_3d):
    data_3d=np.ones((4,N))
    data_3d[0:3,:]=p_3d
    data_2d=P.dot(data_3d)
    data_2d=np.around(data_2d/data_2d[2,:])
    error_mat=(data_2d[0:2,:]-points_2d)*(data_2d[0:2,:]-points_2d)
    #error_mat=(data_2d[0:2,:]-data_2d_batch)*(data_2d[0:2,:]-data_2d_batch)
    P_error=np.sqrt(error_mat.sum(axis=0))
    return np.mean(P_error)    

#%% compute the gradient of P
def computeGradientP(P,points_2d,points_3d):
    lamda1=0.0001
    lamda2=0.0001
    k=6
    g_p=np.zeros((3,4))
    g_p=np.float64(g_p)
    
    nums = [x for x in range(N)]
    random.shuffle(nums)
    p1=np.reshape(P[0,0:3],(3,1))
    p2=np.reshape(P[1,0:3],(3,1))
    p3=np.reshape(P[2,0:3],(3,1))
    divider1=(np.transpose(p1).dot(points_3d)+P[0,3])
    divider2=(np.transpose(p2).dot(points_3d)+P[1,3])
    divider3=(np.transpose(p3).dot(points_3d)+P[2,3])
    inner_product=np.dot(np.transpose(np.cross(p1,p3,axis=0)),np.cross(p2,p3,axis=0))
    #print(inner_product.shape)
    #print(divider1.shape)
    p1CrossdpDivdp=np.zeros((3,3))
    p2CrossdpDivdp=np.zeros((3,3))
    p3CrossdpDivdp=np.zeros((3,3))
    id_array=np.identity(3)
    for i in range(0,3):
        p1CrossdpDivdp[i,:]=np.cross(np.transpose(p1),id_array[i,:])
        p2CrossdpDivdp[i,:]=np.cross(np.transpose(p2),id_array[i,:])
        p3CrossdpDivdp[i,:]=np.cross(np.transpose(p3),id_array[i,:])

    #compute gradient of p1
    sum_g1=2.0*np.dot(divider1/divider3-points_2d[0,:],np.transpose(points_3d/divider3))    
    g_p[0,0:3]=sum_g1+lamda2*np.transpose(np.dot(p3CrossdpDivdp,np.cross(p2,p3,axis=0)))

    #compute gradient of p14
    g_p[0,3]=2.0*np.dot(divider1/divider3-points_2d[0,:],np.transpose(1/divider3))
    
    #compute gradient of p2
    sum_g2=2.0*np.dot(divider2/divider3-points_2d[1,:],np.transpose(points_3d/divider3))
    g_p[1,0:3]=sum_g2+lamda2*np.transpose(np.dot(p3CrossdpDivdp,np.cross(p1,p3,axis=0)))
    
    #compute gradient of p24
    g_p[1,3]=2.0*np.dot(divider2/divider3-points_2d[1,:],np.transpose(1/divider3))
    
    #compute gradient of p3
    complexCrossProduct=np.dot(p1CrossdpDivdp,np.cross(p2,p3,axis=0))+\
                        np.dot(p2CrossdpDivdp,np.cross(p1,p3,axis=0))
                        
    sum_g3=-2.0*np.dot(divider1/divider3-points_2d[0,:],\
           np.transpose(points_3d*divider1/(divider3*divider3)))\
           -2.0*np.dot(divider2/divider3-points_2d[1,:],\
           np.transpose(points_3d*divider2/(divider3*divider3)))
    g_p[2,0:3]=sum_g3+lamda2*np.reshape(complexCrossProduct,(1,3))\
               +2*lamda1*np.transpose(p3)
                  
    #compute gradient of p34
    g_p[2,3]=-2.0*np.dot(divider1/divider3-points_2d[0,:],\
             np.transpose(divider1/(divider3*divider3)))\
             -2.0*np.dot(divider2/divider3-points_2d[1,:],\
             np.transpose(divider2/(divider3*divider3)))
    # compute error with subjection
    data_3d=np.ones((4,54))
    data_3d[0:3,:]=points_3d
    data_2d=np.dot(P_temp,data_3d)
    data_2d=data_2d/data_2d[2,:]
    error_mat=(data_2d[0:2,:]-points_2d)*(data_2d[0:2,:]-points_2d)
    error=np.sum(error_mat.sum(axis=0))+lamda1*np.dot(np.transpose(p3),p3)+\
            lamda2*np.dot(np.transpose(np.cross(p2,p3,axis=0)),np.cross(p1,p3,axis=0))
    return g_p,error

#%% compare the change of position on the image
def imageCompare(P1,name):  
    data_3d=np.ones((4,N))
    data_3d[0:3,:]=points_3d_new
    data_2d=P1.dot(data_3d)
    data_2d=np.around(data_2d/data_2d[2,:])
    img = cv2.imread('/Users/zhangdi/Documents/course computer vision/project2/frame1.bmp')
    #print(img.size)
    for i in range(72):
        pt1 = (int(points_2d[1,i]), int(points_2d[0,i]))
        pt2 = (int(data_2d[1,i]), int(data_2d[0,i]))
        cv2.arrowedLine(img, pt1, pt2, 255, 2)
#    cv2.imshow('Image with arrow', img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    cv2.imwrite(path+name+'.jpg',img)
# %% load data
path="/Users/zhangdi/Documents/course computer vision/project2/"
f=open(path+"Left_2Dpoints.txt")
lines=f.readlines()
N=len(lines)
points_2d=np.zeros((2, N))
points_3d=np.zeros((3, N))
for i in range(0,N):
    line=lines[i].split()
    points_2d[0,i],points_2d[1,i] = int(line[0]), int(line[1])
f.close()
points_3d_new=np.ones((3, N))
f=open(path+"3Dpointnew.txt")
lines=f.readlines()
for i in range(0,N):
    line=lines[i].split()
    points_3d_new[0,i],points_3d_new[1,i], points_3d_new[2,i]= int(line[0]), int(line[1]), int(line[2])
f.close()
f=open(path+"bad_3dpts.txt")
points_3d=np.zeros((3, N))
lines=f.readlines()
for i in range(0,N):
    line=lines[i].split()
    points_3d[0,i],points_3d[1,i], points_3d[2,i]= int(line[0]), int(line[1]), int(line[2])
f.close()

#%% find out 18 worsest point and delete them
# get error array
P_temp =np.loadtxt(path+'P_linear.txt')
data_3d=np.ones((4,N))
data_3d[0:3,:]=p_3d
data_2d=P.dot(data_3d)
data_2d=np.around(data_2d/data_2d[2,:])
error_mat=(data_2d[0:2,:]-points_2d)*(data_2d[0:2,:]-points_2d)
P_error=np.sqrt(error_mat.sum(axis=0))

#find worst and delete
find_bad=[]
right_2d=np.zeros((2,N-18))
right_3d=np.zeros((3,N-18))
for i in range(N):
    find_bad.append([i,P_error[i]])
find_bad.sort(key=lambda x: x[1],reverse=True)
print('bad=',find_bad[0:18])
j=0
flag=0

#generate new subset
for i in range(N):
    for m in range(18):
        if i==find_bad[m][0]:
            flag=1
            break
    if flag:
        flag=0
        continue
    right_2d[:,j]=points_2d[:,i]
    right_3d[:,j]=points_3d[:,i]
    j=j+1
#%% nonlinear method-gradient descent
m_list=[]
g_list=[]
error_list=[]
np.seterr(all='raise') 
#M=1000000
M=100000
min_error=100
P_temp =np.loadtxt(path+'P.txt')
P_min=np.zeros((3,4))
min_error=400
learning_rate=0.000001/N
#learning_rate=0.0000000001/N
for m in range(M): 
    g_p,error=computeGradientP(P_temp,right_2d,right_3d)
    threshold=np.max(abs(g_p))
    if m%10==0:
        #print(g_p)
        if m==0:
            min_g_p=threshold
            g_p_pre=threshold
            P_temp=P_temp-learning_rate*g_p
        else:
            if g_p_pre<threshold or threshold>10**6:
                learning_rate*=0.85
            else:
                if g_p_pre*1.1>threshold:
                    learning_rate*=1.0001
            g_p_pre=threshold
            
    # print immediate result and collect data
    if m%1000==0:
        print(m,threshold,'error=',error[0])
        m_list.append(m)
        error_list.append(error[0])
        g_list.append(threshold)
        
    # find min P
    if threshold<min_g_p:
        min_g_p=threshold
        P_min=P_temp
    P_temp=P_temp-learning_rate*g_p

#%% save data 
np.savetxt(path+'error1.txt', error_list)
np.savetxt(path+'m1.txt', m_list)
np.savetxt(path+'P_extra2.txt', P_min)

#%%  show P, M, W  
P_GD =np.loadtxt(path+'P_extra2.txt')
print(P_GD)
print('mean error=',projection_error_computation(P_GD,points_3d_new))
W_GD,M_GD=parameter_computation(P_GD)

#%% show the error iteration plot
error_list=np.loadtxt(path+'error1.txt')
m_list=np.loadtxt(path+'m1.txt')
plt.plot(m_list, error_list, 'r-')
plt.axis([0, M, min(error_list)-5, max(error_list)+5])
plt.ylabel('error with subjection')
plt.xlabel('iteration times')
plt.show()
plt.plot(m_list, g_list, 'g-')
plt.axis([0, M, 0, max(g_list)+5])
plt.ylabel('max(|gradient|))')
plt.xlabel('iteration times')
plt.show()

#%% compare position
imageCompare(P_GD,'P_GD')