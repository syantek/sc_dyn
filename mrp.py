'''Functions relating to Modified Rodriques Parameters (MRP) for the Coursera
Kinematics: Describing the Motions of Spacecraft course.'''
import numpy as np
from crp import skew

def computeShadowMRP(mrp):
    '''Take in a set of MRP and compute the shadow set corresponding to the input.
    Input is assumed to be a 3x1 np array'''
    sigma = np.linalg.norm(mrp)
    return -mrp / sigma**2

def mrp2dcm(mrp):
    '''Take input MRP and compute the associated DCM. Input is assumed to be a
    3x1 numpy array'''
    sigma2 = np.linalg.norm(mrp)**2
    sigmatTilde = skew(mrp)
    numerator = 8*np.matmul(sigmatTilde,sigmatTilde) - 4*(1-sigma2)*sigmatTilde
    denominator = (1+sigma2)**2
    dcm = np.eye(3) + numerator/denominator
    return dcm

def addMRP(mrp2,mrp1):
    '''Add successive rotations represented by MRPs, with the later rotation being
    the first argument (mrp2). Both inputs are assumed to be 3x1 numpy arrays'''
    normsq1 = np.linalg.norm(mrp1)**2
    normsq2 = np.linalg.norm(mrp2)**2
    denominator = 1 + normsq1*normsq2 - 2*np.vdot(mrp2,mrp1)
    if abs(denominator) < 1e-3:
        if normsq1 > normsq2:
            mrp1 = -mrp1/normsq1
            normsq1 = np.linalg.norm(mrp1)
        else:
            mrp2 = -mrp2/normsq2
            normsq2 = np.linalg.norm(mrp2)
    numerator = (1-normsq1)*mrp2 + (1-normsq2)*mrp1 - 2*np.cross(mrp2.transpose(),mrp1.transpose()).transpose()
    return numerator/denominator

def mrpTimeDeriv(mrp,w):
    '''Compute time derivative of modified rodrigues parameters given angular rate
    w expressed in the resulting (typically body) frame.'''
    normsq = np.linalg.norm(mrp)**2
    mrpTilde = skew(mrp)
    mrpT = np.transpose(mrp)
    mrpmrpT = np.matmul(mrp,mrpT)
    B = (1-normsq)*np.eye(3) + 2*mrpTilde + 2*mrpmrpT
    return (1./4.)*np.matmul(B,w)

# Week 3 concept check 17
#mrp_in = np.array([[0.1],[0.2],[0.3]])
#print(computeShadowMRP(mrp_in))
# Week 3 concept check 18
#1
#print(mrp2dcm(mrp_in))
#2
#print(mrp2dcm(np.array([[-.5],[.1],[.2]])))
#print(mrp2dcm(np.array([[.5],[-.1],[-.2]])))
#print(mrp2dcm(np.array([[1.66],[-.33],[-.66]])))
# Week 3 concept check 19
#2
#s_BN = np.array([[.1,.2,.3]]).transpose()
#s_RB = np.array([[-.1,.3,.1]]).transpose()
#print(addMRP(s_RB,s_BN))
#3
#s_RN = np.array([[.5,.3,.1]]).transpose()
#print(addMRP(s_BN,-s_RN))
# Week 3 concept check 20
#4
def xdot_3_20_4(mrp,t):
    '''differential equation describing time evolution of MRPs for
    week 3 concept check 20 problem 4'''
    # Angular velocity
    w = np.radians(20) * np.array([[np.sin(0.1*t)],[0.01],[np.cos(0.1*t)]])
    return mrpTimeDeriv(mrp,w)

def integrateMRPNewton(x0, f, t0, tf, dt):
    '''Do a simple newton integration on an MRP state whose
    time derivative can also be a function of time. Choose appropriate
    form of the MRP to keep magnitude <1'''
    t = t0
    x = x0
    while t<tf:
        xdot = f(x,t)
        x += xdot*dt
        t+=dt
        xMag = np.linalg.norm(x)
        if xMag > 1.:
            x = -x/(xMag**2)
    return x

def do_3_20_4(dt,f):
    '''Do week 3 concept check 20 problem 4'''
    # Initial conditions
    t0 = 0
    mrp0 = np.array([[0.4],[0.2],[-0.1]])
    # final conditions
    tf = 42
    # Do integration
    xf = integrateMRPNewton(mrp0, f, t0, tf, dt)
    print(xf)
    # Find norm
    print(np.sqrt(xf[0,0]**2+xf[1,0]**2+xf[2,0]**2))
    return

do_3_20_4(0.01,xdot_3_20_4)