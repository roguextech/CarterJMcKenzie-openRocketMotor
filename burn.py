import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# ---------------------------
# initialization calculations
# ---------------------------


def area_mach_relation(exit_mach, specific_heat, exit_area, throat_area):
    AMR = ((2 / (specific_heat + 1) * (1 + exit_mach ** 2 * (specific_heat - 1) / 2)) ** ((specific_heat + 1) /
                                             (2 * (specific_heat - 1)))) / exit_mach - exit_area / throat_area
    return AMR


def calculate_exit_mach(specific_heat, exit_area, throat_area):
    exit_mach = optimize.bisect(area_mach_relation, 1.01, 100, args=(specific_heat, exit_area, throat_area))
    return exit_mach


def calculate_nozzle_exit_temp(burn_temp, specific_heat, mach_exit_velocity):
    exit_temp = burn_temp / (1 + mach_exit_velocity ** 2 * (specific_heat - 1) / 2)
    return exit_temp


def calculate_char_velocity(specific_heat, gas_const, burn_temp):
    char_velocity = (specific_heat * gas_const * burn_temp) ** (1 / 2) /\
             specific_heat * ((2 / (specific_heat + 1)) ** ((specific_heat + 1) / (specific_heat - 1)))
    return char_velocity


def calculate_nozzle_exit_velocity(exit_mach, specific_heat, gas_const, temp_exit):
    exit_velocity = exit_mach * (specific_heat * gas_const * temp_exit) ** 0.5
    return exit_velocity


# ------------------
# loop calculations
# ------------------


def calculate_chamber_pressure(area_burn, area_throat, burn_coef, density_propellant, c_star, burn_exp):
    chamber_pressure = (area_burn / area_throat * burn_coef * density_propellant * c_star) ** (1 / (1 - burn_exp))
    return chamber_pressure


def calculate_burn_rate(burn_coef, p_stag, burn_exp):
    burn_rate = burn_coef * p_stag ** burn_exp
    return burn_rate


def calculate_new_regression_dist(old_dist, burn_rate, dt):
    regression_dist = old_dist + burn_rate * dt
    return regression_dist


def calculate_nozzle_exit_pressure(chamber_pressure, specific_heat, mach_exit):
    exit_pressure = chamber_pressure / (1 + (specific_heat - 1) / 2 * mach_exit ** 2) ** (specific_heat / (specific_heat - 1))
    return exit_pressure


def calculate_mass_flow_rate(burn_area, burn_rate, propellant_density):
    mass_flow_rate = burn_area * burn_rate * propellant_density
    return mass_flow_rate


def calculate_vacuum_thrust(mass_flow, exit_vel, area_exit, pressure_exit):
    vacuum_thrust = mass_flow * exit_vel + area_exit * pressure_exit
    return vacuum_thrust


def calculate_circle_area(diam):
    area = np.pi * diam ** 2 / 4
    return area


def plot_values(self, thrust_list, elapsed_time, dt):
    time_list = np.arange((elapsed_time + dt) / dt) * dt + dt
    plt.plot(time_list, thrust_list)
    plt.show()
