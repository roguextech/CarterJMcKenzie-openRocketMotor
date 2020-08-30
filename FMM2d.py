import numpy as np
import skfmm
import burn
from skimage import measure
import time
import matplotlib.pyplot as plt
import geometry


def make_coordinate_array(motor_diameter, fidelity):
    x, y = np.meshgrid(np.linspace(-motor_diameter / 2, motor_diameter / 2, fidelity),
                                 np.linspace(-motor_diameter / 2, motor_diameter / 2, fidelity))
    return x, y


def add_perimeter_mask(mesh, x, y, motor_diameter):
    mask = np.full_like(mesh, False, dtype=bool)
    mask[x ** 2 + y ** 2 > (motor_diameter/1.99) ** 2] = True
    masked_mesh = np.ma.MaskedArray(mesh, mask)
    return masked_mesh


def add_slice_mask(mesh, x, y, N):
    mask = np.full_like(mesh, False, dtype=bool)
    # slope of slice border
    a = np.tan(np.pi / N)
    mask[y > a * x] = True
    mask[y < -a * x] = True
    masked_mesh = np.ma.MaskedArray(mesh, mask)
    return masked_mesh


def rotate_matrix(x, y, RotRad=0):

    # Clockwise, 2D rotation matrix
    RotMatrix = np.array([[np.cos(RotRad),  np.sin(RotRad)],
                          [-np.sin(RotRad), np.cos(RotRad)]])
    return np.einsum('ji, mni -> jmn', RotMatrix, np.dstack([x, y]))


def calculate_regression(motor, map_ratio):
    return skfmm.distance(motor, map_ratio)


def calculate_port_area(self, masked_motor):
    port_points = np.where(masked_motor == 0)
    port_area = len(port_points[0]) * self.motor_diam ** 2 / self.fidelity ** 2
    return port_area


def calculate_burning_perimeter(regression_map, regression_depth, map_ratio, fidelity):
    contours = measure.find_contours(regression_map, regression_depth, fully_connected='low')
    perimeter = 0
    for contour in contours:
        perimeter += (geometry.length(contour, fidelity)) * map_ratio
    return perimeter


def calculate_burning_area(regression_map, regression_depth, motor_height, map_ratio, fidelity):
    perimeter = calculate_burning_perimeter(regression_map, regression_depth, map_ratio, fidelity)
    return perimeter * motor_height


def burn_motor(regression_map, throat_area, a, density, c_star, gamma, n, exit_mach, exit_velocity, exit_area, dt,
               map_ratio, motor_height, core_diameter, motor_diameter, fidelity):
    elapsed_time = 0  # start the burn at zero seconds
    regression_depth = 0.0000001  # m
    max_regression = (motor_diameter - core_diameter) / 2
    thrust_initial = True
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
    burning_area = calculate_burning_area(regression_map, regression_depth, motor_height, map_ratio, fidelity)
    sim_start = time.time()
    while burning:
        count = count + 1

        # calculate values
        chamber_pressure = burn.calculate_chamber_pressure(burning_area, throat_area, a, density, c_star, n)  # obsolete
        burn_rate = burn.calculate_burn_rate(a, chamber_pressure, n)
        regression_depth = burn.calculate_new_regression_dist(regression_depth, burn_rate, dt)
        burning_area = calculate_burning_area(regression_map, regression_depth, motor_height, map_ratio, fidelity)
        m_dot = burn.calculate_mass_flow_rate(burning_area, burn_rate, density)
        exit_pressure = burn.calculate_nozzle_exit_pressure(chamber_pressure, gamma, exit_mach)
        thrust = burn.calculate_vacuum_thrust(m_dot, exit_velocity, exit_area, exit_pressure)
        elapsed_time = elapsed_time + dt

        if count == 1:
            thrust_initial = thrust
        if thrust < thrust_initial * 0.1:
            burning = False
        if count >= 1000:
            burning = False
            print("2d sim took too many iterations")
        if time.time()-sim_start > 30:
            burning = False
            print("2d sim took too long")
        if regression_depth > max_regression:
            burning = False
            print(f"2d regression distance exceeded max of: {max_regression}")

        # record values
        chamber_pressure_list = np.append(chamber_pressure_list, chamber_pressure)
        burn_rate_list = np.append(burn_rate_list, burn_rate)
        regression_depth_list = np.append(regression_depth_list,regression_depth)
        burning_area_list = np.append(burning_area_list, burning_area)
        m_dot_list = np.append(m_dot_list, m_dot)
        exit_pressure_list = np.append(exit_pressure_list, exit_pressure)
        elapsed_time_list = np.append(elapsed_time_list, elapsed_time)
        thrust_list = np.append(thrust_list, thrust)

    return chamber_pressure_list, burn_rate_list, regression_depth_list, burning_area_list, m_dot_list, exit_pressure_list, elapsed_time_list, thrust_list


def burn_slice(regression_map, N, throat_area, a, density, c_star, gamma, n, exit_mach, exit_velocity, exit_area, dt,
               map_ratio, motor_height, core_diameter, motor_diameter, fidelity):
    elapsed_time = 0  # start the burn at zero seconds
    regression_depth = 0.0000001  # m
    max_regression = (motor_diameter - core_diameter) / 2
    thrust_initial = True
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
    burning_area = N * calculate_burning_area(regression_map, regression_depth, motor_height, map_ratio, fidelity)
    sim_start = time.time()
    while burning:
        count = count + 1

        # calculate values
        chamber_pressure = burn.calculate_chamber_pressure(burning_area, throat_area, a, density, c_star, n)  # obsolete
        burn_rate = burn.calculate_burn_rate(a, chamber_pressure, n)
        regression_depth = burn.calculate_new_regression_dist(regression_depth, burn_rate, dt)
        burning_area = N * calculate_burning_area(regression_map, regression_depth, motor_height, map_ratio, fidelity)
        m_dot = burn.calculate_mass_flow_rate(burning_area, burn_rate, density)
        exit_pressure = burn.calculate_nozzle_exit_pressure(chamber_pressure, gamma, exit_mach)
        thrust = burn.calculate_vacuum_thrust(m_dot, exit_velocity, exit_area, exit_pressure)
        elapsed_time = elapsed_time + dt

        if count == 1:
            thrust_initial = thrust
        if thrust < thrust_initial * 0.1:
            burning = False
        if count >= 1000:
            burning = False
            print("2d sim took too many iterations")
        if time.time()-sim_start > 30:
            burning = False
            print("2d sim took too long")
        if regression_depth > max_regression:
            burning = False
            print(f"2d regression distance exceeded max of: {max_regression}")

        # record values
        chamber_pressure_list = np.append(chamber_pressure_list, chamber_pressure)
        burn_rate_list = np.append(burn_rate_list, burn_rate)
        regression_depth_list = np.append(regression_depth_list,regression_depth)
        burning_area_list = np.append(burning_area_list, burning_area)
        m_dot_list = np.append(m_dot_list, m_dot)
        exit_pressure_list = np.append(exit_pressure_list, exit_pressure)
        elapsed_time_list = np.append(elapsed_time_list, elapsed_time)
        thrust_list = np.append(thrust_list, thrust)

    return chamber_pressure_list, burn_rate_list, regression_depth_list, burning_area_list, m_dot_list, exit_pressure_list, elapsed_time_list, thrust_list


def plot_regression(motor, motor_diameter, fidelity):
    """Uses the fast marching method to generate an image of how the grain regresses from the core map"""
    X, Y = make_coordinate_array(motor_diameter, fidelity)
    Z = skfmm.distance(motor, motor_diameter/fidelity)
    mask = np.full_like(motor, False, dtype=bool)
    mask[X ** 2 + Y ** 2 > (motor_diameter / 2) ** 2] = True
    Z = np.ma.array(Z, mask=mask)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('2D FMM Map')
    fig.savefig('2D_FMM_Map.png')
