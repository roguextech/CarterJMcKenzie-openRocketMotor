import motors
import propellants
import results

# experiment with values to change mesh fidelity and number of time steps
fidelity = 500  # suggested ranges--- (3d FMM: 50-150), (2d FMM: 150-1000), (analytical: Not used)
dt = 0.1  # suggested range 0.01 - 0.2

# set propellant
propellant = propellants.QDL

# set motor
motor1 = motors.ExampleStar(propellant, dt, fidelity)

# run burn method
motor1.fast_marching_2d()

# repeat for additional motors
motor2 = motors.ExampleStar2(propellant, dt, fidelity)
motor2.fast_marching_2d()
motor3 = motors.ExampleBates(propellant, dt)
motor3.analytical_method()

# OPTIONAL: print iterative calculations at each time step
# results.display_iteration_values(motor2)

# plot values with corresponding functions to show and save
results.plot_thrust([motor1, motor2, motor3])

