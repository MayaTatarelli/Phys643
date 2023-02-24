"""
This is a numerical simulation of the time evolution 
of a viscously accreting disk.
@author: Maya Tatarelli
@Collaborators: Amalia Karalis, Nicole Ford, Kevin Marimbu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#Useful functions
def velocity(x, kin_visc):
    u = (-9.0/2.0)*kin_visc/x
    return u

#In this code I use x, which really is r (radial distance from the star)

#Parameters for viscous diffusion
kin_visc = 0.01
D = 3*kin_visc

#Grid (spatial) parameters
xL = 0.0
xR = 1.0
dx = 0.01
Ngrid = (xR-xL)/dx + 1
Ngrid = int(Ngrid)

#x-axis
x = np.linspace(xL,xR,Ngrid)

#Velocity u for advection
u = np.append([0], velocity(x[1:], kin_visc)) #includes a place holder for u[0]

#Setting boundary conditions for u with gas flow off the grid at both ends
u[0] = -np.abs(u[1])
u[-1] = np.abs(u[-2])

#Initial condition for surface density:
#Sharp Gaussian centered at the midpoint of the simulation box
f = norm.pdf(x,0.5,1E-1)

#Temporal parameters
t = 0.0
t_final = 4.0
dt = np.min(np.abs(dx/u)) #dt <= dx/u
Nsteps = int(t_final/dt)

#Params for diffusion
beta = D*dt/dx**2

#Calculate diffusion first
# Setting up matrices for diffusion operator
A = np.eye(Ngrid) * (1.0 + 2.0 * beta) + np.eye(Ngrid, k=1) * -beta + np.eye(Ngrid, k=-1) * -beta

#Setting up plot
plt.ion()
fig, ax = plt.subplots(1,1)
ax.plot(x, f, 'k-') #plot the initial state for reference

plt1, = ax.plot(x, f, 'ro')
fig.canvas.draw()


#Loop through all time steps
for count in range(Nsteps):
    
    #Diffusion
    # Solving for the next timestep
    f = np.linalg.solve(A, f)
    
    #Calculate advection
    #Lax-Friedrichs Method
    f[1:Ngrid-1] = -0.5*u[1:Ngrid-1]*(dt/dx)*(f[2:] - f[:Ngrid-2]) + 0.5*(f[2:] + f[:Ngrid-2])
    
    #Set boundary conditions for gas outflow off the grid
    f[0] = f[1]
    f[-1] = f[-2]
    
    #Update plot
    plt1.set_ydata(f)
    ax.set_ylabel('$\Sigma$')
    ax.set_xlabel('Radial Distance')
    fig.canvas.draw()
    plt.pause(0.005)
