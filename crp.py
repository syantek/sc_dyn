'''Classical Rodrigue Parameter functions'''

import numpy as np
from rotationUtils import integrateNewton

def skew(q):
    '''Return the skew matrix of a 3-vector represented by the 3x1 numpy array q'''
    return np.array([[0,-q[2,0],q[1,0]],[q[2,0],0,-q[0,0]],[-q[1,0],q[0,0],0]])

def crp2dcm(q):
    '''Take in a 3x1 numpy array representing CRP, q, and output the
    associated DCM'''
    qt = np.transpose(q)
    qtq = np.matmul(qt,q)
    qqt = np.matmul(q,qt)
    qtilde = skew(q)
    dcm = 1/(1+qtq)*((1-qtq)*np.eye(3) + 2*qqt - 2*qtilde)
    return dcm

def addcrp(q2,q1):
    '''Add 2 sets of CRPs, with the later rotation passed infirst as q2 and the earlier
    rotational passed 2nd as q1.
    Array multiplication in numpy is not designed for linear algebra applications'''
    numerator = q2 + q1 - np.cross(q2.transpose(),q1.transpose()).transpose()
    denom = 1 - np.vdot(q2,q1)
    return numerator/denom

#1
#print(crp2dcm(np.array([[0.1],[0.2],[0.3]])))
#1
#print(crp2dcm(np.array([[-0.2],[-0.4],[-0.6]])))
#print(crp2dcm(np.array([[0.2],[-0.4],[-0.6]])))
#print(crp2dcm(np.array([[0.2],[0.4],[0.6]])))
#3 - by inspection
#4
#q2 = np.array([[-.3,.3,.1]]).transpose()
#q1 = np.array([[-.1,-.2,-.3]]).transpose()
#print(addcrp(q2,q1))

def crpTimeDeriv(q,w):
    '''Given CRP q and result frame angular velocity w, return time
    derivative of q, qdot. Both inputs are 3x1 numpy arrays'''
    qt = np.transpose(q)
    qqt = np.matmul(q,qt)
    qtilde = skew(q)

    innerterm = np.eye(3) + qtilde + qqt
    res = 0.5*np.matmul(innerterm,w)
    return res

# week 3 concept check 13
def xdot_3_13_2(q,t):
    '''differential equation describing time evolution of CRPs for
    week 3 concept check 13 problem 2'''
    # Angular velocity
    w = np.radians(3) * np.array([[np.sin(0.1*t)],[0.01],[np.cos(0.1*t)]])
    return crpTimeDeriv(q,w)

def do_3_13_2(dt,f):
    '''Do week 3 concept check 13 problem 2'''
    # Initial conditions
    t0 = 0
    q0 = np.array([[0.4],[0.2],[-0.1]])
    # final conditions
    tf = 42
    # Do integration
    xf = integrateNewton(q0, f, t0, tf, dt)
    print(xf)
    # Find norm
    print(np.sqrt(xf[0,0]**2+xf[1,0]**2+xf[2,0]**2))
    return

#do_3_13_2(0.01,xdot_3_13_2)