import numpy as np
import control as ct

r = 0.25  # reaction wheel radius
L = 1     # pendulum length
l = 0.5   # pendulum length to com
g = 9.8   # gravitational acceleration

mp = 0.5          # pendulum mass
mw = 1            # wheel mass
m0 = mp*l + mw*L  # gravitational torque/potential energy coeff.

# pendulum and wheel moments of inertia
Ip = 0.33*mp*L**2 
Iw = mw*r**2 

# inertia matrix terms
m11 = mp*l**2 + mw*L**2 + Ip + Iw
m12 = m21 = m22 = Iw

detM = m11*m22 - m12*m21  # determinant of intertia matrix

E_d = m0*g  # energy of homoclinic orbit

kt = 0.5  # BLDC motor torque constant

# min and max input signal values
u_min = -12
u_max = 12

ke = 0.3  # swing-up gain 

small_angle_threshold = 30  # angle (in degrees) at which LQR control activates

# state space dynamics matrix
A = np.array(
    [[0, 1, 0, 0], 
     [m22*m0*g/detM, 0, 0, 0], 
     [0, 0, 0, 1], 
     [-m12*m0*g/detM, 0, 0, 0]]
)

# state space control matrix
B = np.array(
    [[0], 
     [-m21*kt/detM],
     [0],
     [m11*kt/detM]]
)

Q = np.diag([1, 1, 0, 0.1])   # state cost matrix
R = 1                         # control cost matrix 
K, S, E = ct.lqr(A, B, Q, R)  # K: linear feedback matrix


Q_nmpc = np.diag([1, 1, 0, 0.1])
R_nmpc = 1                       