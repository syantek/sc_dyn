from re import X
import numpy as np
from rotationUtils import DCM2ScalarFirstRightQ, scalarFirstRightQ2DCM

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

def getK(w,n,b,K_only=True):
    '''Calculate the K matrix corresponding to Devenport's Q method and QUEST.
    Inputs: list of weights w
        list of external frame vectors n
        list of body frame vector corresponding to n element by element, v
    output: 4x4 numpy array K'''
    # Compute B
    B = np.zeros((3,3))
    for i, wi in enumerate(w):
        B = B + wi*np.matmul(b[i],n[i].transpose())
    # Compute components of K
    S = B + B.transpose()
    sigma = np.array([[B[0,0]+B[1,1]+B[2,2]]])
    Z = np.array([[B[1,2]-B[2,1],B[2,0]-B[0,2],B[0,1]-B[1,0]]]).transpose()
    # Compose K matrix
    K1 = np.concatenate([sigma,Z.transpose()],axis=1)
    K2 = np.concatenate([Z,S-sigma[0,0]*np.eye(3)],axis=1)
    K = np.concatenate([K1,K2],axis=0)
    if K_only: # return only the K matrix
        return K
    else: # also give the different pieces of K
        return K, S, sigma, Z

def KCharacteristicEqn(K):
    '''Characteristic equation to find eigenvalues (s) of K'''
    return lambda s: np.linalg.det(K-s*np.eye(4))

def DevenportQ(w,n,b):
    '''take in iterables of weights (w, scalar), measurement vectors in the external
    frame (n, 3x1 numpy arrays), and measurement vectors in the body frame
    (b, 3x1 numpy arrays). All 3 inputs are assumed to have the same number of
    elements. Compute the quaternion solution to Wahba's problem
    using Devenport's q method.'''
    K = getK(w,n,b)
    # Find eigensystem
    lambdas, vecs = np.linalg.eig(K)
    max_index = 0
    max_lambda = lambdas[max_index]
    for i, lambdai in enumerate(lambdas):
        if lambdai > max_lambda:
            max_index = i
            max_lambda = lambdai
    quat = np.array([vecs[:,max_index]]).transpose()
    # Compute quaternion with shorter rotation
    if quat[0,0] < 0:
        quat = -quat
    return quat

def NewtonRaphson1d(x0,f,dx_factor=1e-3,eps=1e-14):
    x = x0
    xp = x0-eps*1e5
    while abs(x-xp) > eps:
        fprime = (f(x*(1+dx_factor))-f(x))/dx_factor
        xp = x
        x = xp - f(x)/fprime
    return x


def QUEST(w,n,b):
    '''take in iterables of weights (w, scalar), measurement vectors in the external
    frame (n, 3x1 numpy arrays), and measurement vectors in the body frame
    (b, 3x1 numpy arrays, corresponding to the contents of n). All 3 inputs are assumed
    to have the same number of elements. Compute the solution to Wahba's problem
    using QUEST.'''
    K,S,sigma,Z = getK(w,n,b,K_only=False)
    f = KCharacteristicEqn(K)
    # Initial guess of greatest eigenvalue
    lambda0 = sum(w)
    # Solve for actual greatest eigenvalue
    lambdaf = NewtonRaphson1d(lambda0,f)
    # calculate CRP associated with this eigenvalue/eigenvector pair
    crp = np.matmul(np.linalg.inv((lambdaf+sigma[0,0])*np.eye(3)-S),Z)
    # Calculate quaternion from CRP
    qTq = np.matmul(crp.transpose(),crp)[0,0]
    quat = 1/np.sqrt(1+qTq)*np.concatenate([np.array([[1]]),crp])
    # Calcualte DCM
    C = scalarFirstRightQ2DCM(quat.transpose()[0]) # Need to change this to use 4x1 array
    print(C)

# Week 4 concept check 2
# 1
def do_4_2_1():
    v1b = np.array([[0.8273,0.5541,-0.0920]]).transpose()
    v2b = np.array([[-0.8285,0.5522,-0.0955]]).transpose()
    v1n = np.array([[-0.1517,-0.9669,0.2050]]).transpose()
    v2n = np.array([[-0.8393,0.4494,-0.3044]]).transpose()
    print(TRIAD(v1b,v1n,v2b,v2n))
#do_4_2_1()
# 2
#BNtilde = np.array([[0.969846,0.17101,0.173648],[-0.200706,0.96461,0.17101],[-0.138258,-0.200706,0.969846]])
#BN = np.array([[0.963592,0.187303,0.190809],[-0.223042,0.956645,0.187303],[-0.147454,-0.223042,0.963592]])
#print(np.degrees(DCMAngleError(BNtilde,BN)))

# Week 4 concept check 3
# 6
def do_4_3_6():
    n = []
    n.append(np.array([[-0.1517,-0.9669,0.2050]]).transpose())
    n.append(np.array([[-0.8393,0.4494,-0.3044]]).transpose())
    b = []
    b.append(np.array([[0.8273,0.5541,-0.0920]]).transpose())
    b.append(np.array([[-0.8285,0.5522,-0.0955]]).transpose())
    w = [10, 1]
    q = DevenportQ(w,b,n)
    C = scalarFirstRightQ2DCM(q.transpose()[0]) # Need to change this to use 4x1 array
    print(C)

#do_4_3_6()

def do_4_5_6():
    n = []
    n.append(np.array([[-0.1517,-0.9669,0.2050]]).transpose())
    n.append(np.array([[-0.8393,0.4494,-0.3044]]).transpose())
    b = []
    b.append(np.array([[0.8273,0.5541,-0.0920]]).transpose())
    b.append(np.array([[-0.8285,0.5522,-0.0955]]).transpose())
    w = [10, 1]
    q = QUEST(w,b,n)

do_4_5_6()