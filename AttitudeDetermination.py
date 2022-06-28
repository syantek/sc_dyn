import numpy as np
from rotationUtils import DCM2ScalarFirstRightQ

def TRIAD(v1b,v1n,v2b,v2n):
    '''TRIAD algorithm.
    Take in 2 vector measurements in the body frame (v1b, v2b) that are also
    known in the reference frame n (v1n, v2n). They must have significant perpendicular
    component. Output the DCM'''
    b1, b2, b3 = getRightHandedUnitSet(v1b,v2b)
    n1, n2, n3 = getRightHandedUnitSet(v1n,v2n)
    # Matrix to transform from intermediate frame T to body frame B
    BT = np.concatenate((b1,b2,b3),axis=1)
    # Matrix to transform from external frame N to intermediate frame T
    TN = np.concatenate((n1,n2,n3),axis=1).transpose()
    # Result matric from frame N to frame B
    BN = np.matmul(BT,TN)
    return BN

def getRightHandedUnitSet(v1,v2):
    '''Take in 2 vectors represented by 3x1 numpy arrays and return the right-handed
    unit vector triad defined by the following vectors normalized:
    t1=v1, t2=cross(v1,v2), t3=cross(t1,t2)'''
    t1 = v1/np.linalg.norm(v1)
    t2 = np.cross(t1.transpose(),v2.transpose()).transpose()
    t2 = t2/np.linalg.norm(t2)
    t3 = np.cross(t1.transpose(),t2.transpose()).transpose()
    t3 = t3/np.linalg.norm(t3)
    return t1,t2,t3

def DCMAngleError(c1,c2):
    '''Take in 2 DCMS represented as 3x3 numpy arrays. Compute the rotation
    angle between the two DCMs and output it in radians'''
    # delta DCM
    dC = np.matmul(c1,c2.transpose())
    # Convert to delta quaternion
    dq = np.array([DCM2ScalarFirstRightQ(dC)]) #TODO change this function to output 4x1 numpy array
    # delta angle from dq
    dtheta = 2*np.arccos(dq[0,0])
    return dtheta

# Week 4 concept check 2
# 1
#v1b = np.array([[0.8273,0.5541,-0.0920]]).transpose()
#v2b = np.array([[-0.8285,0.5522,-0.0955]]).transpose()
#v1n = np.array([[-0.1517,-0.9669,0.2050]]).transpose()
#v2n = np.array([[-0.8393,0.4494,-0.3044]]).transpose()
#print(TRIAD(v1b,v1n,v2b,v2n))
# 2
#BNtilde = np.array([[0.969846,0.17101,0.173648],[-0.200706,0.96461,0.17101],[-0.138258,-0.200706,0.969846]])
#BN = np.array([[0.963592,0.187303,0.190809],[-0.223042,0.956645,0.187303],[-0.147454,-0.223042,0.963592]])
#print(np.degrees(DCMAngleError(BNtilde,BN)))