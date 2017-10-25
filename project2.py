#Di Zhang Project 2
import numpy as np
import math
import cv2
import random
import cv2
#%% set up print options
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

#%% use linear method 1 to compute P using whole data set
def computePAll(data2,data3):
    #get random number
    k=72
    A=np.zeros((2*k,12))
    P=np.zeros((3,4))
    data_2d_batch=np.zeros((2,k))
    data_3d_batch=np.zeros((3,k))
    for i in range(0,k):
        data_2d_batch[:,i]=data2[:,i];
        data_3d_batch[:,i]=data3[:,i];
        
        #construct matrix A
        A[2*i,0:3]   =np.transpose(data_3d_batch[:,i])
        A[2*i,3]     =1
        A[2*i,8:11]  =-data_2d_batch[0,i]*np.transpose(data_3d_batch[:,i])
        A[2*i,11]    =-data_2d_batch[0,i]
        A[2*i+1,4:7] =np.transpose(data_3d_batch[:,i])
        A[2*i+1,7]   =1
        A[2*i+1,8:11]=-data_2d_batch[1,i]*np.transpose(data_3d_batch[:,i])
        A[2*i+1,11]  =-data_2d_batch[1,i]
        #print(i)
    #compute P
    rank_a=np.linalg.matrix_rank(A)
    U, s, V = np.linalg.svd(A, full_matrices=True)
    P_vec=V[-1,:]
    alpha=1/np.sqrt(V[-1,8]**2+V[-1,9]**2+V[-1,10]**2)
    P_vec=P_vec*alpha
    P[0,:]=P_vec[0:4]
    P[1,:]=P_vec[4:8]
    P[2,:]=P_vec[8:12]
    
    #compute projection error
    data_3d=np.ones((4,N))
    data_3d[0:3,:]=points_3d
    data_2d=P.dot(data_3d)
    data_2d=data_2d/data_2d[2,:]
    error_mat=(data_2d[0:2,:]-data2)*(data_2d[0:2,:]-data2)
    P_error=np.mean(np.sqrt(error_mat.sum(axis=0)))
    return P,np.mean(P_error)

#%% using k subset to compute P
def computeP(k):
    #get random number
    nums = [x for x in range(N)]
    random.shuffle(nums)
    A=np.zeros((2*k,12))
    P=np.zeros((3,4))
    data_2d_batch=np.zeros((2,k))
    data_3d_batch=np.zeros((3,k))
    for i in range(0,k):
        data_2d_batch[:,i]=points_2d[:,nums[i]];
        data_3d_batch[:,i]=points_3d[:,nums[i]];
        #construct matrix A
        A[2*i,0:3]   =np.transpose(data_3d_batch[:,i])
        A[2*i,3]     =1
        A[2*i,8:11]  =-data_2d_batch[0,i]*np.transpose(data_3d_batch[:,i])
        A[2*i,11]    =-data_2d_batch[0,i]
        A[2*i+1,4:7] =np.transpose(data_3d_batch[:,i])
        A[2*i+1,7]   =1
        A[2*i+1,8:11]=-data_2d_batch[1,i]*np.transpose(data_3d_batch[:,i])
        A[2*i+1,11]  =-data_2d_batch[1,i]
        #print(i)
        
    #compute P
    data_3d=np.ones((4,N))
    rank_a=np.linalg.matrix_rank(A)
    if rank_a==11:
        #linear solution 1  
        U, s, V = np.linalg.svd(A, full_matrices=True)
        P_vec=V[-1,:]
        alpha=1/np.sqrt(V[-1,8]**2+V[-1,9]**2+V[-1,10]**2)
        
        #reconstruct P
        P_vec=P_vec*alpha
        P[0,:]=P_vec[0:4]
        P[1,:]=P_vec[4:8]
        P[2,:]=P_vec[8:12]
        
        #compute projection error
        data_3d=np.ones((4,N))
        data_3d[0:3,:]=points_3d
        data_2d=P.dot(data_3d)
        data_2d=data_2d/data_2d[2,:]
        error_mat=(data_2d[0:2,:]-points_2d)*(data_2d[0:2,:]-points_2d)
        P_error=np.mean(np.sqrt(error_mat.sum(axis=0)))
        return P,np.mean(P_error)
    else:
        #linear solution 2 
        #print('linear solution 2')
        B=A[:,0:11]
        b=A[:,11]
        BTB=np.transpose(B).dot(B)
        try:
            inverse_BTB=np.linalg.inv(np.transpose(B).dot(B))
        except np.linalg.LinAlgError:
            #print(np.linalg.matrix_rank(BTB))
            print('np.linalg.LinAlgError')
            return P, 1000
        else:
            Y=(-1)*inverse_BTB.dot(np.transpose(B)).dot(b)
            p34=1./math.sqrt(Y[8]*Y[8]+Y[9]*Y[9]+Y[10]*Y[10])
            P_vec=Y*p34
            
            #reconstruct P
            P[0,:]  =P_vec[0:4]
            P[1,:]  =P_vec[4:8]
            P[2,0:3]=P_vec[8:11]
            P[2,3]  =p34
            
            #compute projection error
            data_3d=np.ones((4,N))
            data_3d[0:3,:]=points_3d
            #data_3d[0:3,:]=data_3d_batch
            data_2d=P.dot(data_3d)
            data_2d[2,data_2d[2,:]==0]=0.001
            data_2d=np.around(data_2d/data_2d[2,:])
            error_mat=(data_2d[0:2,:]-points_2d)*(data_2d[0:2,:]-points_2d)
            P_error=np.sqrt(error_mat.sum(axis=0))
#            print(np.mean(P_error)) 
            return P,np.mean(P_error)
        
#%% calculate parameters
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

#%% compute error of projection
def projection_error_computation(P,p_3d):
    data_3d=np.ones((4,N))
    data_3d[0:3,:]=p_3d
    data_2d=P.dot(data_3d)
    data_2d=np.around(data_2d/data_2d[2,:])
    error_mat=(data_2d[0:2,:]-points_2d)*(data_2d[0:2,:]-points_2d)
    P_error=np.sqrt(error_mat.sum(axis=0))
    return np.mean(P_error)

#%% load data
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

#%% use linear method 1 to compute the P with good points and bad points
P_1,error=computePAll(points_2d,points_3d_new)
print('P_1=',P_1)
W_1,M_1=parameter_computation(P_1)
P_2,error=computePAll(points_2d,points_3d)
print('P_2=',P_2)
W_2,M_2=parameter_computation(P_2)
print('the first method, mean error P_1=',projection_error_computation(P_1,points_3d_new))
print('the first method, mean error P_2=',projection_error_computation(P_2,points_3d_new))
np.savetxt(path+'P_1.txt', P_1)
np.savetxt(path+'P_2.txt', P_2)

#%% RANSAC method
k=6
M=100000
P_min=np.zeros((3,4))
min_error=10000
for m in range(M): 
    P_temp,error=computeP(k)
    if error<min_error:
        P_min[:,:]=P_temp[:,:]
        min_error=error
    if m%100==0:
        print(min_error)
np.savetxt(path+'P_linear.txt', P_min)

#%%
P =np.loadtxt(path+'P_linear.txt')
print('P_ransac=',P)
W_ransac,M_ransac=parameter_computation(P)
print('the RANSC method, mean error P=',projection_error_computation(P,points_3d_new))

#%% see the difference of postion of the image
def imageCompare(P1,name):  
    data_3d=np.ones((4,N))
    data_3d[0:3,:]=points_3d_new
    data_2d=P1.dot(data_3d)
    data_2d=np.around(data_2d/data_2d[2,:])
    img = cv2.imread('/Users/zhangdi/Documents/course computer vision/project2/frame1.bmp')
    #print(img.size)
    for i in range(72):
        pt1 = (int(points_2d[0,i]), int(points_2d[1,i]))
        pt2 = (int(data_2d[0,i]), int(data_2d[1,i]))
        cv2.arrowedLine(img, pt1, pt2, 255, 2)
#    cv2.imshow('Image with arrow', img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    cv2.imwrite(path+name+'.jpg',img)
    
#%% compare position
imageCompare(P_1,'P_1')
imageCompare(P_2,'P_2')
imageCompare(P,'P_ransac')
