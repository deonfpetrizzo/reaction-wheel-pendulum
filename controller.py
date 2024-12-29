import numpy as np
import param
from pendulum import Pendulum
from scipy.optimize import minimize
import casadi as ca


# load control parameters
Q, R, K = param.Q, param.R, param.K 
Q_nmpc, R_nmpc = param.Q_nmpc, param.R_nmpc
u_min, u_max = param.u_min, param.u_max
ke, kt = param.ke, param.kt

class Controller: 
	def __init__(self): 
		pass

	def lqr(self, state_error): 
		'''updates linear control law'''
		return self._constrain(-(K @ state_error)[0,0], u_min, u_max)

	def swing_up(self, qp_d, qw_d, energy_error):
		'''updates swing-up controller'''
		return self._constrain(ke*qp_d*energy_error, u_min, u_max)
    
	def nmpc(self, x0, xd, u_prev):
		''' solves the nmpc optimization problem:
		    	J := discrete-time finite-horizon quadratic cost 
				u* = argmin(J)
					s.t. x[k+1] = f(x[k], u[k]), f := non-linear system dynamics;
			     		 x[0] = x0;
			     		 u_min <= u <= u_max
				apply u*[0] '''
		def nmpc_cost(u0, N):
			cost = 0.0
			pend = Pendulum(x0)
			for k in range(N):
				u = u0[k]
				qp_dd, qw_dd = pend.compute_dynamics(kt*u)
				x = pend.update_state(qp_dd, qw_dd)
				x_err = x - xd
				cost += x_err.T @ Q_nmpc @ x_err + R_nmpc * u**2
			return cost

		N = 20
		u0 = np.full(N, u_prev)  # initial guess for control input sequence
		u_bounds = [(u_min, u_max) for _ in range(N)]  # control input constraint

		result = minimize(
			nmpc_cost,
			u0, 
			args=N,
			bounds=u_bounds
		)

		if result.success:
			optimal_control = result.x[0]
			print(optimal_control)
		else:
			raise RuntimeError("mpc optimization failed")

		return optimal_control
	
	def nmpc_casadi(self, x0, xd, u_prev): 
		N = 20
	
		u = ca.MX.sym("u", N)
		x = ca.MX(x0)

		cost = 0.0
		constraints = []
		pend = Pendulum(x0)

		for k in range(N):
			uk = u[k]
			qp_dd, qw_dd = pend.compute_dynamics(kt*u)
			x_next = pend.update_state(qp_dd, qw_dd)
			x_err = x_next - xd
			cost += ca.mtimes([x_err.T, Q_nmpc, x_err]) + R_nmpc * uk**2
			x = x_next

		nlp = {"x": u, "f": cost}
		opts = {"ipopt.print_level": 0, "print_time": 0}
		solver = ca.nlpsol("solver", "ipopt", nlp, opts)

		u0 = np.full(N, u_prev)
		lbg = [u_min] * N
		ubg = [u_max] * N

		sol = solver(x0=u0, lbg=lbg, ubg=ubg)

		return sol["x"].full().flatten()[0]

	def _constrain(self, val, min_val, max_val): 
		'''constrains val between min_val and max_val'''
		return min(max_val, max(min_val, val))