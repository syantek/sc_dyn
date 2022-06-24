import numpy as np

def euler321toDCM(ang1, ang2, ang3):
    '''Convert sequence of the 3 Euler angles in radians corresponding to
    3-2-1 sequence into a DCM as a numpy array'''
    c1 = np.cos(ang1)
    s1 = np.sin(ang1)
    c2 = np.cos(ang2)
    s2 = np.sin(ang2)
    c3 = np.cos(ang3)
    s3 = np.sin(ang3)
    dcm = np.array([[c1*c2, s1*c2, -s2],
                    [c1*s2*s3-s1*c3, s1*s2*s3+c1*c3, c2*s3],
                    [c1*s2*c3+s1*s3, s1*s2*c3-c1*s3, c2*c3]])
    return dcm

def DCMtoEuler321(dcm):
    '''take in a 3x3 numpy array DCM and extract the 3-2-1 Euler sequence
    angles from it'''
    a1 = np.arctan2(dcm[0][1],dcm[0][0])
    a2 = -np.arcsin(dcm[0][2])
    a3 = np.arctan2(dcm[1][2],dcm[2][2])
    return a1, a2, a3

def doProblem582(b1, b2, b3, r1, r2, r3):
    '''Takes in 2 sets of 3-2-1 Euler angle sequences in degrees. b_angles
    represents the transformation from N to B, and r_angles represents the
    transformation from N to R. Turn each into a DCM, and use those 2 DCMs
    to compute the DCM going from R to B. Then use the R to B DCM to find the
    3-2-1 Euler sequence from R to B, in degrees'''
    NtoB = euler321toDCM(np.radians(b1), np.radians(b2), np.radians(b3))
    NtoR = euler321toDCM(np.radians(r1), np.radians(r2), np.radians(r3))
    RtoB = np.matmul(NtoB,np.transpose(NtoR))
    angles = DCMtoEuler321(RtoB)
    print(np.degrees(angles))

def xdot_2_9_2(x,t):
    '''differential equation describing time evolution of euler angles for
    week 2 concept check 9 problem 2'''
    # Angular velocity
    w = np.radians(20) * np.array([[np.sin(0.1*t)],[0.01],[np.cos(0.1*t)]])
    # terms that are used a lot
    c2 = np.cos(x[1,0])
    s2 = np.sin(x[1,0])
    c3 = np.cos(x[2,0])
    s3 = np.sin(x[2,0])
    mat_term = (1/c2)*np.array([[0,s3,c3],[0,c3*c2,-s3*c2],[c2,s3*s2,c3*s2]])
    xdot = np.matmul(mat_term,w)
    return xdot
    

def integrateNewton(x0, f, t0, tf, dt):
    '''Do a simple newton integration on an n-dimensional state whose
    time derivative can also be a function of time'''
    t = t0
    x = x0
    while t<tf:
        xdot = f(x,t)
        x += xdot*dt
        t+=dt
    return x

def do_2_9_2(dt, f):
    '''Do problem 2 in week 2 concept check 9'''
    # Initial conditions
    t0 = 0
    theta1 = np.radians(40)
    theta2 = np.radians(30)
    theta3 = np.radians(80)
    x0 = np.array([[theta1],[theta2],[theta3]])
    # final conditions
    tf = 42
    # Do integration
    xf = integrateNewton(x0, f, t0, tf, dt)
    print(xf)
    # Find norm
    print(np.sqrt(xf[0,0]**2+xf[1,0]**2+xf[2,0]**2))
    return
    
#do_2_9_2(0.01, xdot_2_9_2)

def scalarFirstRightQ2DCM(q):
    '''Takes a (1D array-like) right quaternion with order (scalar, v1, v2, v3)
    and transforms it into a DCM. Outputs the DCM as a 2D numpy array'''
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    
    c11 = q0**2 + q1**2 - q2**2 - q3**2
    c12 = 2*(q1*q2 + q0*q3)
    c13 = 2*(q1*q3 - q0*q2)
    c21 = 2*(q1*q2 - q0*q3)
    c22 = q0**2 - q1**2 + q2**2 - q3**2
    c23 = 2*(q2*q3 + q0*q1)
    c31 = 2*(q1*q3 + q0*q2)
    c32 = 2*(q2*q3 - q0*q1)
    c33 = q0**2 - q1**2 - q2**2 + q3**2
    
    C = np.array([[c11, c12, c13],[c21, c22, c23],[c31, c32, c33]])
    return C

#print(scalarFirstRightQ2DCM([0.235702,0.471405,-0.471405,0.707107]))

def DCM2ScalarFirstRightQ(C):
    '''Take in a DCM represented as a 3x3 numpy array and transform it into
    a scalar first right quaternion with smallest angle magnitude. Return the
    resulting quaternion as a list'''
    # first compute the square of each quaternion element
    traceC = C[0,0]+C[1,1]+C[2,2]
    qsq = [0,0,0,0]
    qsq[0] = 0.25*(1+traceC)
    qsq[1] = 0.25*(1+2*C[0,0]-traceC)
    qsq[2] = 0.25*(1+2*C[1,1]-traceC)
    qsq[3] = 0.25*(1+2*C[2,2]-traceC)
    # Next find the index of the largest element
    index_max = np.argmax(qsq)
    # Now use the largest element to calculate all elements
    q = [0,0,0,0]
    match index_max:
        case 0:
            q[0] = np.sqrt(qsq[0])
            q[1] = (C[1,2]-C[2,1])/(4*q[0])
            q[2] = (C[2,0]-C[0,2])/(4*q[0])
            q[3] = (C[0,1]-C[1,0])/(4*q[0])
        case 1:
            q[1] = np.sqrt(qsq[1])
            q[0] = (C[1,2]-C[2,1])/(4*q[1])
            q[2] = (C[0,1]+C[1,0])/(4*q[1])
            q[3] = (C[2,0]+C[0,2])/(4*q[1])
        case 2:
            q[2] = np.sqrt(qsq[2])
            q[0] = (C[2,0]-C[0,2])/(4*q[2])
            q[1] = (C[0,1]+C[1,0])/(4*q[2])
            q[3] = (C[1,2]+C[2,1])/(4*q[2])
        case 3:
            q[3] = np.sqrt(qsq[3])
            q[0] = (C[0,1]-C[1,0])/(4*q[3])
            q[1] = (C[2,0]+C[0,2])/(4*q[3])
            q[2] = (C[1,2]+C[2,1])/(4*q[3])
        case _:
            raise ValueError("DCM2ScalarFirstRightQ could not find an appropriate maximum quaternion component magnitude")
            
    # Choose positive scalar term for minimum angle solution
    if q[0] < 0:
        for i in range(4):
            q[i] = -q[i]
    return q

#print(DCM2ScalarFirstRightQ(np.array([[-0.529403,-0.467056,0.708231],[-0.474115,-0.529403,-0.703525],[0.703525,-0.708231,0.0588291]])))
#print(DCM2ScalarFirstRightQ(euler321toDCM(np.deg2rad(20), np.deg2rad(10), np.deg2rad(-10))))

def addScalarFirstRightQuaternions(q2,q1):
    '''Use quaternion multiplication to combine 2 rotations represented by
    the 2 input (list-like) quaternions in the order q2*q1'''
    leftSide = np.array([[q2[0],-q2[1],-q2[2],-q2[3]],[q2[1],q2[0],q2[3],-q2[2]],[q2[2],-q2[3],q2[0],q2[1]],[q2[3],q2[2],-q2[1],q2[0]]])
    rightSide = np.array([[q1[0]],[q1[1]],[q1[2]],[q1[3]]])
    return np.matmul(leftSide,rightSide)

#print(addScalarFirstRightQuaternions([0.359211,0.898027,0.179605,0.179605],[0.774597,0.258199,0.516398,0.258199]))
#print(addScalarFirstRightQuaternions([0.359211,0.898027,0.179605,0.179605],[0.377964,0.755929,0.377964,0.377964]))

def getQDot(q, w):
    '''Take in a scalar first right quaternion (list-like) q, and list-like anguar rate in the resulting frame w.
    Output the time derivative of the quaternion.'''
    leftSide = np.array([[q2[0],-q2[1],-q2[2],-q2[3]],[q2[1],q2[0],-q2[3],q2[2]],[q2[2],q2[3],q2[0],-q2[1]],[q2[3],-q2[2],q2[1],q2[0]]])
    rightSide = np.array([[0],[w[0]],[w[1]],[w[2]]])
    return np.matmul(leftSide,rightSide)