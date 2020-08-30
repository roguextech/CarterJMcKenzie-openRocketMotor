README

Aug 8th 2020

Discrete Element Motor Burn  v1.2

Files:
    
    main.py
    motors.py
    propellants.py
    grains.py
    burn.py
    analytical.py
    FMM3d.py
    FMM2d.py
    geometry.py
    results.py

Background:

This program simulates the regression of a solid propellant motor from ignition to burn out
by performing a level-set type method on a grid map of the grain geometry

Instructions:

1. Set delta time 
    0. (recommended 0.05 < dt < 0.2)  
2. Set fidelity for mesh grids
    0. recommended 2d FMM: 150 < fidelity < 1250 
    0. recommended 3d FMM: 50 < fidelity < 150
3. Instantiate propellant
4. Instantiate motor with propellant, dt, and fidelity
5. Repeat 1-4 for additional motors
6. List and/or plot results for motors


Outputs:

Curves and data lists of thrust, chamber pressure, mass flow rate, burn rate, burning area,
exit pressure, and regression distance all with respect to time.  Additionally a regression map
for the 2d FMM is shown

Necessary Libraries:

- Numpy v1.19
- Scikit Fast Marching Method  https://github.com/scikit-fmm/scikit-fmm [2]
- Scikit Image  https://scikit-image.org/docs/dev/api/skimage.measure.html [1]
- matplotlib.pyplot
- mpl_toolkits.mplot3d.art3d

Notes:

The same concepts used in this program are implemented in an opensource motor simulator called OpenMotor.  https://github.com/reilleya/openMotor