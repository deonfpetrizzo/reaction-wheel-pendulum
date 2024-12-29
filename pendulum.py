import numpy as np
import param
import casadi as ca

# load system parameters
r, L, l = param.r, param.L, param.l
mp, mw, m0, g = param.mp, param.mw, param.m0, param.g
m11, m12, m21, m22 = param.m11, param.m12, param.m21, param.m22
detM = param.detM
A, B = param.A, param.B

class Pendulum:
	def __init__(self, state):
		self.qp = np.deg2rad(state[0,0])
		self.qp_d = state[1,0]
		self.qw = np.deg2rad(state[2,0])
		self.qw_d = state[3,0]
		self.dt = 0.01

	def compute_dynamics(self, Tm): 
	    '''computes the angular acceleration of the pendulum arm and wheel'''
	    qp_dd = (m22*m0*g*np.sin(self.qp) - m21*Tm) / detM
	    qw_dd = (-m12*m0*g*np.sin(self.qp) + m11*Tm) / detM
	    return (qp_dd, qw_dd)

	def update_state(self, qp_dd, qw_dd): 
	    '''uses the forward euler method to extract the state variables of the 
	       system from qp_dd and qw_dd'''
	    self.qp_d = self.qp_d + qp_dd*self.dt
	    self.qp = self.qp + self.qp_d*self.dt
	    self.qw_d = self.qw_d + qw_dd*self.dt
	    self.qw = self.qw + self.qw_d*self.dt
	    state = np.array([[self.qp], [self.qp_d], [self.qw], [self.qw_d]])
	    # state = ca.vertcat(self.qp, self.qp_d, self.qw, self.qw_d)
	    return state

	def compute_energy(self):
	    '''computes the energy of the decoupled pendulum rod'''
	    return 0.5*m11*self.qp_d**2 + m0*g*np.cos(self.qp)

	def compute_forward_kinematics(self): 
	    '''computes the (x,y) coordinates of the end of the pendulum
	       and the relative offset to a point on the rim of the reaction wheel
	       (for animating wheel spoke rotation)'''
	    xp, yp = L*np.sin(self.qp), L*np.cos(self.qp)
	    dx, dy = r*np.sin(self.qw), -r*np.cos(self.qw)
	    return (xp, yp, dx, dy)

