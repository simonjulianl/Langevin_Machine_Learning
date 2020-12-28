****Harmonic\_approximation***

Approximate the Hamiltonian function of the system by a local expansion of the potential up to second order.

In code,
 
***Hamiltonian/LJ\_term.py*** for calculations (potential, first derivative, second derivative of LJ)

***Integrator/analytic\_method.py*** for calculation of next state q,p using analytical solution of SHO with LJ potential (U0, V and G) that is code in ***/integrator/methods/application\_vv.py***

****VV\_LJ\_potential***

Velocity Verlet algorithm are used to integrate Hamilton's equations of motion.

In code,
 
***Hamiltonian/LJ\_term.py*** for calculations ( Potential and first derivative ) 

***pb.py*** for periodic boundary condition 
