import sys
import numpy as np
import control as ct
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import param
from pendulum import Pendulum
from controller import Controller
import casadi as ca


# load system and control parameters
L, r = param.L, param.r  
kt = param.kt
balance_threshold = np.deg2rad(param.small_angle_threshold)

# lists for storing animation and plotting data
qp_, qw_, qp_d_, qw_d_, xp_, yp_, dx_, dy_, t_ = ([] for _ in range(9))

dt = 0.01 


def animate():
    '''animates the pendulum system'''  
    x0, y0, dx0, dy0 = xp_[0], yp_[0], dx_[0], dy_[0]

    # initialize figure and axes
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # draw shapes
    arm, = ax.plot([0, x0], [0, y0], lw=2, c='black')
    spoke1, = ax.plot([x0-dx0, x0+dx0], [y0-dy0, y0+dy0], lw=1, c='red')
    spoke2, = ax.plot([x0-dy0, x0+dy0], [y0+dx0, y0-dx0], lw=1, c='red')
    pivot = ax.add_patch(plt.Circle((0,0), radius=0.03, fc='black'))
    wheel = ax.add_patch(plt.Circle((x0,y0), radius=r, ec='red', fc='none', lw=3))

    ax.set_xlim([-L-0.5, L+0.5])
    ax.set_ylim([-L-0.5, L+0.5])

    def update(i):
        '''update animation at frame i'''
        x, y, dx, dy = xp_[i], yp_[i], dx_[i], dy_[i]
        arm.set_data([0, x], [0, y])
        spoke1.set_data([x-dx, x+dx], [y-dy, y+dy])
        spoke2.set_data([x-dy, x+dy], [y+dx, y-dx])
        wheel.set_center((x,y))

    nframes = len(t_)   
    interval = dt * 500  # delay between each frame (ms)
    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=nframes, repeat=True, interval=interval
    )
    plt.show()
    ani.save('ani30fps.gif', writer=animation.PillowWriter(fps=30))


def plot(): 
    '''plots four sublots: the phase portrait of the system, qp vs t, qp_d vs t, 
       and qw_d vs t'''
    fig, axs = plt.subplots(2, 2)
    xlabels = [r'$\theta_p$', r'$t$', r'$t$', r'$t$']
    ylabels = [r'$\dot{\theta_p}$', r'$\theta_p$', r'$\dot{\theta_p}$', r'$\dot{\theta_w}$']

    axs[0, 0].plot(qp_, qp_d_)
    axs[0, 1].plot(t_, qp_)
    axs[1, 0].plot(t_, qp_d_)
    axs[1, 1].plot(t_, qw_d_)

    i = 0
    for ax in axs.flat: 
        ax.set(xlabel=xlabels[i], ylabel=ylabels[i])
        i = i + 1


def simulate(controller='lqr', qp_init=181, runtime=15, ani=False, plot=True): 
    '''simulates the reaction wheel pendulum system'''
    state = np.array([[qp_init], [0], [0], [0]]) 
    pend = Pendulum(state)
    ctr = Controller()

    time_elapsed = 0
    Tm = 0
    u_prev = 0.0

    while time_elapsed < runtime:    
        energy_error = pend.compute_energy() - param.E_d
        qp_dd, qw_dd = pend.compute_dynamics(Tm)
        state = pend.update_state(qp_dd, qw_dd)
        xp, yp, dx, dy = pend.compute_forward_kinematics()
        qp, qp_d, qw, qw_d = (state[i, 0] for i in range(4))

        match controller: 
            case 'lqr':
                if (abs(qp) % (2*np.pi) < balance_threshold or 
                        abs(qp) % (2*np.pi - balance_threshold) < balance_threshold):  
                    qp_d = 2*np.pi if qp > np.pi else 0
                    setpoint = np.array([[qp_d], [0], [0], [0]]) 
                    state_error = state - setpoint
                    u = ctr.lqr(state_error)
                else: 
                    u = ctr.swing_up(qp_d, qw_d, energy_error)
            case 'swing-up':
                u = ctr.swing_up(qp_d, qw_d, energy_error)
            case 'nmpc':
                # setpoint = np.array([[0], [0], [0], [0]]) 
                setpoint = ca.vertcat(0, 0, 0, 0)
                u = ctr.nmpc_casadi(state, setpoint, u_prev) 
            case 'none':  
                u = 0

        for lst, val in zip([qp_, qw_, qp_d_, qw_d_, xp_, yp_, dx_, dy_, t_], 
                  [qp, qw, qp_d, qw_d, xp, yp, dx, dy, time_elapsed]):
            lst.append(val)

        Tm = kt*u  # assume Tm ~ u (voltage)
        time_elapsed = time_elapsed + dt 
        u_prev = u

    if ani:
        animate()
    if plot: 
        plt.plot(qp_, qp_d_)
        plt.xlabel(r'$\theta_p$')
        plt.ylabel(r'$\dot{\theta_p}$')
        plt.show()


if __name__ == '__main__':
    simulate()