import numpy as np
import burn
import time
import geometry

########################################################################################################################

# Analytical Bates Profile Equations

########################################################################################################################


def bates_burn_perimeter(core_diameter, regression_depth):
    return np.pi * (core_diameter + 2 * abs(regression_depth))


def bates_face_area(motor_diameter, core_diameter, regression_depth, burn_top, burn_bottom):
    face_area = np.pi * motor_diameter ** 2 / 4 - np.pi * (core_diameter + 2 * abs(regression_depth)) ** 2 / 4
    return burn_top * face_area + burn_bottom * face_area


def burn_motor_height(regression_depth, burn_top, burn_bottom, motor_height):
    return motor_height - abs(regression_depth) * burn_top - abs(regression_depth) * burn_bottom


def bates_burn_area(regression_depth, core_diameter, motor_height, motor_diameter, burn_top, burn_bottom):
    burn_perimeter = bates_burn_perimeter(core_diameter, regression_depth)
    motor_height = burn_motor_height(regression_depth, burn_top, burn_bottom, motor_height)
    face_area = bates_face_area(motor_diameter, core_diameter, regression_depth, burn_top, burn_bottom)
    return burn_perimeter * motor_height + face_area


def bates_burn_motor(throat_area, a, density, c_star, gamma, n, exit_mach, exit_velocity, exit_area, dt, motor_height,
                     motor_diameter, core_diameter, burn_top, burn_bottom):
    regression_depth = 0
    elapsed_time = 0  # start the burn at zero seconds
    count = 0
    chamber_pressure_list = np.array([0])
    burn_rate_list = np.array([0])
    regression_depth_list = np.array([0])
    burning_area_list = np.array([0])
    m_dot_list = np.array([0])
    exit_pressure_list = np.array([0])
    thrust_list = np.array([0])
    elapsed_time_list = np.array([0])
    burning = True
    burning_area = bates_burn_area(regression_depth, core_diameter, motor_height, motor_diameter, burn_top, burn_bottom)
    sim_start = time.time()
    while burning:
        count = count + 1

        # calculate iterations
        chamber_pressure = burn.calculate_chamber_pressure(burning_area, throat_area, a, density, c_star, n)  # obsolete
        burn_rate = burn.calculate_burn_rate(a, chamber_pressure, n)
        regression_depth = burn.calculate_new_regression_dist(regression_depth, burn_rate, dt)
        burning_area = bates_burn_area(regression_depth, core_diameter, motor_height, motor_diameter, burn_top, burn_bottom)
        m_dot = burn.calculate_mass_flow_rate(burning_area, burn_rate, density)
        exit_pressure = burn.calculate_nozzle_exit_pressure(chamber_pressure, gamma, exit_mach)
        thrust = burn.calculate_vacuum_thrust(m_dot, exit_velocity, exit_area, exit_pressure)
        elapsed_time = elapsed_time + dt

        # record values
        chamber_pressure_list = np.append(chamber_pressure_list, chamber_pressure)
        burn_rate_list = np.append(burn_rate_list, burn_rate)
        regression_depth_list = np.append(regression_depth_list, regression_depth)
        burning_area_list = np.append(burning_area_list, burning_area)
        m_dot_list = np.append(m_dot_list, m_dot)
        exit_pressure_list = np.append(exit_pressure_list, exit_pressure)
        elapsed_time_list = np.append(elapsed_time_list, elapsed_time)
        thrust_list = np.append(thrust_list, thrust)
        if regression_depth * 2 + core_diameter > motor_diameter:
            burning = False
            # print("'successful' burn out")
        if count >= 1000:
            burning = False
            print("unsuccessful burn out, 1000 iteration criteria violated")
        if time.time()-sim_start > 60:
            print("unsuccessful burn out, 60 second criteria violated")
            burning = False
    return chamber_pressure_list, burn_rate_list, regression_depth_list, burning_area_list, m_dot_list, exit_pressure_list, elapsed_time_list, thrust_list


########################################################################################################################

# Analytical Star Profile Equations

########################################################################################################################


def star_burn_perimeter(Ro, Rp, epsilon, N, y, f, theta, phase):
    burn_perimeter = 0
    if phase == 1:
        s1 = Rp * np.sin(np.pi*epsilon/N) / np.sin(theta/2) - (y + f) / np.tan(theta / 2)
        s2 = (y + f) * (np.pi / 2 - theta / 2 + np.pi*epsilon/N) + (Rp + y + f) * (np.pi/N + np.pi*epsilon/N)
        burn_perimeter = 2 * N * (s1 + s2)
    if phase == 2:
        s1 = (Rp + f + y) * (np.pi/N - np.pi*epsilon/N)
        s2 = (y + f) * ((np.pi/2 - theta/2 + np.pi*epsilon/N) - np.arctan(
            ((y + f)**2 - (Rp * np.sin(np.pi*epsilon/N))**2)**0.5 / (Rp * np.sin(np.pi*epsilon/N))) - theta/2)
        burn_perimeter = 2 + N * (s1 + s2)
    if phase == 3:
        beta = np.pi / 2 - theta / 2 + np.pi * epsilon / N
        gamma = np.arctan((((y + f) ** 2 - Rp * np.sin(np.pi*epsilon/N)**2) ** 0.5) / (Rp * np.sin(np.pi*epsilon/N))) - theta / 2
        xi = np.pi - np.arccos(-(Ro ** 2 - Rp ** 2 - (y + f) ** 2) / (2 * Rp * (y + f)))
        burn_perimeter = 2 * N * ((y + f) * (beta - gamma - xi))
    return burn_perimeter


def star_port_area(Ro, Rp, Ri, epsilon, N, y, f, phase):
    theta = 2 * np.arctan(Rp * np.sin(np.pi*epsilon/N) * np.tan(np.pi*epsilon/N) / (Rp * np.sin(np.pi*epsilon/N) - Ri * np.tan(np.pi*epsilon/N)))
    port_area = 0
    if phase == 1:
        a1 = 1/2 * Rp * np.sin(np.pi*epsilon/N) * (
                    Rp * np.cos(np.pi*epsilon/N) + Rp * np.sin(np.pi*epsilon/N) * np.tan(theta/2))
        a2 = -1/2 * (Rp * np.sin(np.pi*epsilon/N) / np.sin(theta/2) - (y + f) / np.tan(theta/2)) ** 2 * np.tan(theta/2)
        a3 = 1/2 * (y + f)**2 * (np.pi/2 - theta/2 + np.pi*epsilon/N) + 1/2 * (Rp + y + f)**2 * (np.pi/N - np.pi*epsilon/N)
        port_area = 2 * N * (a1 + a2 + a3)

    if phase == 2:
        a1 = 1/2 * Rp * np.sin(np.pi*epsilon/N) * (
                    Rp * np.cos(np.pi*epsilon/N) + ((y + f)**2 - Rp * np.sin(np.pi*epsilon/N)**2)**0.5)
        a2 = 1/2 * (y + f)**2 * ((np.pi/2 - theta/2 + np.pi*epsilon/N) - np.arctan(
            ((y + f)**2 - (Rp * np.sin(np.pi*epsilon/N))**2)** 0.5 / (Rp * np.sin(np.pi*epsilon/N))) - theta/2)
        a3 = 1/2 * (Rp + y + f)**2 * (np.pi/2 - np.pi*epsilon/N)
        port_area = 2 * N * (a1 + a2 + a3)

    if phase == 3:
        xi = np.pi - np.arccos(-(Ro** 2 - Rp** 2 - (y + f)** 2) / (2 * Rp * (y + f)))
        mu = np.arcsin((y + f) / Ro * np.sin(np.pi - xi))
        beta = np.pi/2 - theta/2 + np.pi*epsilon/N
        gamma = np.arctan(((y+f)**2 - Rp * np.sin(np.pi*epsilon/N)**2)**0.5 / (Rp * np.sin(np.pi*epsilon/N))) - theta/2
        a1 = Ro** 2 * (np.pi/N * (1 - epsilon) + mu) + (y + f)**2 * (beta - gamma - xi)
        a2 = Rp * np.sin(np.pi * epsilon / N) * (Rp * np.cos(np.pi*epsilon/N) + ((y + f) ** 2 - Rp * np.sin(np.pi*epsilon/N) ** 2) ** 0.5)
        a3 = -Rp * np.sin(mu) * (Rp * np.cos(mu) + ((y + f) ** 2 - Rp * np.sin(mu)) ** 0.5)
        port_area = N * (a1 + a2 + a3)
    return port_area


def star_face_area(port_area, motor_diameter, burn_top, burn_bottom):
    face_area = np.pi * motor_diameter**2 / 4 - port_area
    return burn_top * face_area + burn_bottom * face_area


def star_burn_area(motor_height, motor_diameter, Ro, Rp, Ri, epsilon, N, y, f, theta, burn_top, burn_bottom):
    phase = 0
    if y <= Rp * np.sin(np.pi*epsilon/N) / np.cos(theta/2) - f:
        phase = 1
    if y >= Rp * np.sin(np.pi*epsilon/N) / np.cos(theta/2) - f:
        phase = 2
    # if Rp + f + y > Ro:
    #     phase = 3
    print(f"phase = {phase}")
    print(f"y = {y}")
    print(f"fraction = {Rp* np.sin(np.pi*epsilon/N)/np.cos(theta/2)}")
    print(f"f = {f}")
    print(f"")
    burn_perimeter = star_burn_perimeter(Ro, Rp, epsilon, N, y, f, theta, phase)
    port_area = star_port_area(Ro, Rp, Ri, epsilon, N, y, f, phase)
    face_area = star_face_area(port_area, motor_diameter, burn_top, burn_bottom)
    return burn_perimeter * motor_height + face_area


def star_burn(Ro, Rp, Ri, epsilon, N, f, a, density, c_star, gamma, n, exit_mach, exit_velocity, exit_area, dt, motor_height,
                     motor_diameter, throat_area, burn_top, burn_bottom):

    # initialize loop variables
    regression_depth = 0
    elapsed_time = 0  # start the burn at zero seconds
    count = 0

    # initialize output lists
    chamber_pressure_list = np.array([0])
    burn_rate_list = np.array([0])
    regression_depth_list = np.array([0])
    burning_area_list = np.array([0])
    m_dot_list = np.array([0])
    exit_pressure_list = np.array([0])
    thrust_list = np.array([0])
    elapsed_time_list = np.array([0])

    # determine theta parameter from inputs
    theta = 2 * np.arctan(Rp * np.sin(np.pi * epsilon / N) * np.tan(np.pi * epsilon / N) /
                          (Rp * np.sin(np.pi * epsilon/ N) - Ri * np.tan(np.pi * epsilon / N)))
    print(f"theta = {theta}")
    # calculate zero contour burn area
    burning_area = star_burn_area(motor_height, motor_diameter, Ro, Rp, Ri, epsilon, N, regression_depth, f, theta, burn_top, burn_bottom)
    sim_start = time.time()
    burning = True
    while burning:
        count = count + 1
        
        # iterate calculations
        chamber_pressure = burn.calculate_chamber_pressure(burning_area, throat_area, a, density, c_star, n)  # obsolete
        burn_rate = burn.calculate_burn_rate(a, chamber_pressure, n)
        regression_depth = burn.calculate_new_regression_dist(regression_depth, burn_rate, dt)
        burning_area = star_burn_area(motor_height, motor_diameter, Ro, Rp, Ri, epsilon, N, regression_depth, f, theta, burn_top, burn_bottom)
        m_dot = burn.calculate_mass_flow_rate(burning_area, burn_rate, density)
        exit_pressure = burn.calculate_nozzle_exit_pressure(chamber_pressure, gamma, exit_mach)
        thrust = burn.calculate_vacuum_thrust(m_dot, exit_velocity, exit_area, exit_pressure)
        elapsed_time = elapsed_time + dt

        # record values
        chamber_pressure_list = np.append(chamber_pressure_list, chamber_pressure)
        burn_rate_list = np.append(burn_rate_list, burn_rate)
        regression_depth_list = np.append(regression_depth_list, regression_depth)
        burning_area_list = np.append(burning_area_list, burning_area)
        m_dot_list = np.append(m_dot_list, m_dot)
        exit_pressure_list = np.append(exit_pressure_list, exit_pressure)
        elapsed_time_list = np.append(elapsed_time_list, elapsed_time)
        thrust_list = np.append(thrust_list, thrust)

        # print values
        print(f"\nIteration: {count}")
        print(f"Time: {round(elapsed_time, 3)} sec")
        print(f"Regression: {round(regression_depth, 3)} m")
        print(f"Burning Area: {round(burning_area, 3)} m^2")
        print(f"Pressure: {round(chamber_pressure, 3)} Pa")
        print(f"Burn Rate: {round(burn_rate, 3)} m/sec")
        print(f"Mass Flow Rate: {round(m_dot, 3) } kg/sec")
        print(f"Exit Pressure: {round(exit_pressure, 3)} Pa")
        print(f"Thrust: {round(thrust, 3)} N")

        # available exits
        if regression_depth + f + Ri > motor_diameter:
            burning = False
            # print("'successful' burn out")
        if count >= 1000:
            burning = False
            print("unsuccessful burn out, 1000 iteration criteria violated")
        if time.time() - sim_start > 60:
            print("unsuccessful burn out, 60 second criteria violated")
            burning = False
        if thrust == 0:
            burning = False

    return chamber_pressure_list, burn_rate_list, regression_depth_list, burning_area_list, m_dot_list, exit_pressure_list, elapsed_time_list, thrust_list


########################################################################################################################

# Analytical Wagon Wheel Profile Equations

########################################################################################################################


def wagon_wheel_burn_perimeter(Ro, Rp, Ri, f, N, epsilon, theta, y, phase):
    H = Rp * np.sin(np.pi * epsilon / N) - y - f
    burn_perimeter = 0
    if phase == 1:
        s1 = (Rp + f + y)*(np.pi/N - np.pi*epsilon/N) + (y + f)*(np.pi/2 + np.pi*epsilon/N)
        s2 = (Rp * np.cos(np.pi*epsilon/N) - Ri - (y + f)/np.sin(theta/2) + H/np.tan(theta/2) + H/np.sin(theta/2))
        burn_perimeter = 2 * N * (s1 + s2)
    if phase == 2:
        phi = np.pi/2 + np.pi*epsilon/N - np.arccos(Rp * np.sin(np.pi*epsilon/N) / (y + f))
        s1 = (Rp + f + y)*(np.pi/N - np.pi*epsilon/N) + (y + f)*phi
        burn_perimeter = 2 * N * s1
    if phase == 3:
        beta = np.pi/2 - theta/2 + np.pi*epsilon/N
        gamma = np.arctan(((y + f)** 2 - Rp * np.sin(np.pi*epsilon/N)** 2)** 0.5 / (Rp * np.sin(np.pi*epsilon/N))) - theta / 2
        xi = np.pi - np.arccos(-(Ro** 2 - Rp** 2 - (y + f)** 2) / (2 * Rp * (y + f)))
        burn_perimeter = 2 * N * ((y + f)*(beta - gamma - xi))
    return burn_perimeter


def wagon_wheel_port_area(Ro, Rp, Ri, f, N, epsilon, theta, y, phase):
    H = Rp * np.sin(np.pi * epsilon / N) - y - f
    port_area = 0
    if phase == 1:
        a1 = 1/2*(Rp + f + y)**2 * (np.pi/N - np.pi*epsilon/N) + 1/2*(f + y)**2 * (np.pi/2 + np.pi*epsilon/N)
        a2 = 1/2 * Rp**2 * np.sin(np.pi*epsilon/N) * np.cos(np.pi*epsilon/N)
        a3 = -(H*(Rp*np.cos(np.pi*epsilon/N) - Ri - (y + f)/np.sin(theta/2) + H/np.tan(theta/2)) + 1/2*H**2/np.tan(theta/2))
        port_area = 2 * N * (a1 + a2 + a3)
    if phase == 2:
        a1 = (Rp + f + y)**2 * (np.pi/N - np.pi*epsilon/N) + (f + y)**2 * (np.pi/2 + np.pi*epsilon/N - np.arccos(Rp * np.sin(np.pi*epsilon/N) / (y + f)))
        a2 = Rp * np.sin(np.pi*epsilon/N) * (Rp*np.cos(np.pi*epsilon/N) + (y + f) * np.sin(np.arccos(Rp * np.sin(np.pi*epsilon/N) / (y + f))))
        port_area = N * (a1 + a2)
    if phase == 3:
        xi = np.pi - np.arccos(-(Ro** 2 - Rp** 2 - (y + f)** 2) / (2 * Rp * (y + f)))
        mu = np.arcsin((y + f) / Ro * np.sin(np.pi - xi))
        beta = np.pi/2 - theta/2 + np.pi*epsilon/N
        gamma = np.arctan(((y+f)**2 - Rp * np.sin(np.pi*epsilon/N)**2)**0.5 / (Rp * np.sin(np.pi*epsilon/N))) - theta/2
        a1 = Ro** 2 * (np.pi/N * (1 - epsilon) + mu) + (y + f)**2 * (beta - gamma - xi)
        a2 = Rp * np.sin(np.pi * epsilon / N) * (
                    Rp * np.cos(np.pi*epsilon/N) + ((y + f) ** 2 - Rp * np.sin(np.pi*epsilon/N) ** 2) ** 0.5)
        a3 = -Rp * np.sin(mu) * (Rp * np.cos(mu) + ((y + f) ** 2 - Rp * np.sin(mu)) ** 0.5)
        port_area = N * (a1 + a2 + a3)
    return port_area


def wagon_wheel_face_area(port_area, motor_diameter, burn_top, burn_bottom):
    face_area = np.pi * motor_diameter ** 2 / 4 - port_area
    return burn_top * face_area + burn_bottom * face_area


def wagon_wheel_burn_area(motor_height, motor_diameter, Ro, Rp, Ri, f, N, epsilon, theta, y, burn_top, burn_bottom):
    phase = 0
    if y <= Rp * np.sin(np.pi * epsilon / N) - f:
        phase = 1
    if y >= Rp * np.sin(np.pi * epsilon / N) - f:
        phase = 2
    if Rp + f + y > Ro:
        phase = 3
    burn_perimeter = wagon_wheel_burn_perimeter(Ro, Rp, Ri, f, N, epsilon, theta, y, phase)
    port_area = wagon_wheel_port_area(Ro, Rp, Ri, f, N, epsilon, theta, y, phase)
    face_area = wagon_wheel_face_area(port_area, motor_diameter, burn_top, burn_bottom)
    return face_area + burn_perimeter * motor_height


def wagon_wheel_burn(Ro, Rp, Ri, f, N, epsilon, theta, a, density, c_star, gamma, n, exit_mach, exit_velocity,
                     exit_area, dt, motor_height, motor_diameter, throat_area, burn_top, burn_bottom):

    # initialize loop variables
    regression_depth = 0
    elapsed_time = 0  # start the burn at zero seconds
    count = 0

    # initialize output lists
    chamber_pressure_list = np.array([0])
    burn_rate_list = np.array([0])
    regression_depth_list = np.array([0])
    burning_area_list = np.array([0])
    m_dot_list = np.array([0])
    exit_pressure_list = np.array([0])
    thrust_list = np.array([0])
    elapsed_time_list = np.array([0])

    # calculate zero contour burn area
    burning_area = wagon_wheel_burn_area(motor_height, motor_diameter, Ro, Rp, Ri, f, N, epsilon, theta,
                                         regression_depth, burn_top, burn_bottom)
    sim_start = time.time()
    burning = True
    while burning:
        count = count + 1

        # iterate calculations
        chamber_pressure = burn.calculate_chamber_pressure(burning_area, throat_area, a, density, c_star, n)  # obsolete
        burn_rate = burn.calculate_burn_rate(a, chamber_pressure, n)
        regression_depth = burn.calculate_new_regression_dist(regression_depth, burn_rate, dt)
        burning_area = wagon_wheel_burn_area(motor_height, motor_diameter, Ro, Rp, Ri, f, N, epsilon, theta,
                                             regression_depth, burn_top, burn_bottom)
        m_dot = burn.calculate_mass_flow_rate(burning_area, burn_rate, density)
        exit_pressure = burn.calculate_nozzle_exit_pressure(chamber_pressure, gamma, exit_mach)
        thrust = burn.calculate_vacuum_thrust(m_dot, exit_velocity, exit_area, exit_pressure)
        elapsed_time = elapsed_time + dt

        # record values
        chamber_pressure_list = np.append(chamber_pressure_list, chamber_pressure)
        burn_rate_list = np.append(burn_rate_list, burn_rate)
        regression_depth_list = np.append(regression_depth_list, regression_depth)
        burning_area_list = np.append(burning_area_list, burning_area)
        m_dot_list = np.append(m_dot_list, m_dot)
        exit_pressure_list = np.append(exit_pressure_list, exit_pressure)
        elapsed_time_list = np.append(elapsed_time_list, elapsed_time)
        thrust_list = np.append(thrust_list, thrust)

        # available exits
        if (regression_depth + Ri + f) > motor_diameter:
            burning = False
            # print("'successful' burn out")
        if thrust <= 0:
            burning = False
        if count >= 1000:
            burning = False
            print("unsuccessful burn out, 1000 iteration criteria violated")
        if time.time() - sim_start > 60:
            print("unsuccessful burn out, 60 second criteria violated")
            burning = False

    return chamber_pressure_list, burn_rate_list, regression_depth_list, burning_area_list, m_dot_list,\
           exit_pressure_list, elapsed_time_list, thrust_list


########################################################################################################################

# Analytical End Burner

########################################################################################################################


def end_burn(a, density, c_star, gamma, n, exit_mach, exit_velocity, exit_area, throat_area, dt, motor_height,
             motor_diameter):

    # initialize loop variables
    regression_depth = 0
    elapsed_time = 0  # start the burn at zero seconds
    count = 0

    # initialize output lists
    chamber_pressure_list = np.array([0])
    burn_rate_list = np.array([0])
    regression_depth_list = np.array([0])
    burning_area_list = np.array([0])
    m_dot_list = np.array([0])
    exit_pressure_list = np.array([0])
    thrust_list = np.array([0])
    elapsed_time_list = np.array([0])

    burning_area = geometry.circle_area(motor_diameter)
    sim_start = time.time()
    burning = True
    while burning:
        count = count + 1

        # iterate calculations
        chamber_pressure = burn.calculate_chamber_pressure(burning_area, throat_area, a, density, c_star, n)
        burn_rate = burn.calculate_burn_rate(a, chamber_pressure, n)
        regression_depth = burn.calculate_new_regression_dist(regression_depth, burn_rate, dt)
        m_dot = burn.calculate_mass_flow_rate(burning_area, burn_rate, density)
        exit_pressure = burn.calculate_nozzle_exit_pressure(chamber_pressure, gamma, exit_mach)
        thrust = burn.calculate_vacuum_thrust(m_dot, exit_velocity, exit_area, exit_pressure)
        elapsed_time = elapsed_time + dt

        # record values
        chamber_pressure_list = np.append(chamber_pressure_list, chamber_pressure)
        burn_rate_list = np.append(burn_rate_list, burn_rate)
        regression_depth_list = np.append(regression_depth_list, regression_depth)
        burning_area_list = np.append(burning_area_list, burning_area)
        m_dot_list = np.append(m_dot_list, m_dot)
        exit_pressure_list = np.append(exit_pressure_list, exit_pressure)
        elapsed_time_list = np.append(elapsed_time_list, elapsed_time)
        thrust_list = np.append(thrust_list, thrust)

        # available exits
        if regression_depth >= motor_height:
            burning = False
            # print("'successful' burn out")
        if count >= 100000:
            burning = False
            print("unsuccessful burn out, 1000 iteration criteria violated")
        if time.time() - sim_start > 30:
            print("unsuccessful burn out, 30 second criteria violated")
            burning = False


    return chamber_pressure_list, burn_rate_list, regression_depth_list, burning_area_list, m_dot_list, exit_pressure_list, elapsed_time_list, thrust_list
