import numpy as np
import burn
import FMM3d
import FMM2d
import analytical


class Bates:
    def __init__(self, propellant, motor_height, motor_diameter, core_diameter, exit_area, throat_area, burn_top, burn_bottom):
        self.burn_top = burn_top
        self.burn_bottom = burn_bottom
        self.motor_height = motor_height
        self.motor_diameter = motor_diameter
        self.exit_area = exit_area
        self.throat_area = throat_area

        self.core_diameter = core_diameter

        self.density = propellant.density
        self.a = propellant.a
        self.n = propellant.n
        self.T = propellant.T
        self.gamma = propellant.gamma
        self.molar_mass = propellant.molar_mass
        self.c_star = propellant.c_star
        self.R = propellant.R

        self.exit_mach = burn.calculate_exit_mach(self.gamma, self.exit_area, self.throat_area)
        self.exit_temp = burn.calculate_nozzle_exit_temp(self.T, self.gamma, self.exit_mach)
        self.exit_velocity = burn.calculate_nozzle_exit_velocity(self.exit_mach, self.gamma, self.R, self.exit_temp)

        # initializing variables to be stored after burn method
        self.chamber_pressure = 0
        self.burn_rate = 0
        self.regression_depth = 0
        self.burning_area = 0
        self.m_dot = 0
        self.exit_pressure = 0
        self.elapsed_time = 0
        self.thrust = 0
        self.method = 0

    def analytical_method(self, dt):
        chamber_pressure, burn_rate, regression_depth, burning_area, m_dot, exit_pressure, elapsed_time, thrust = \
            analytical.bates_burn_motor(self.throat_area, self.a, self.density, self.c_star, self.gamma, self.n,
                                    self.exit_mach, self.exit_velocity, self.exit_area, dt, self.motor_height,
                                    self.motor_diameter, self.core_diameter, self.burn_top, self.burn_bottom)
        self.chamber_pressure = chamber_pressure
        self.burn_rate = burn_rate
        self.regression_depth = regression_depth
        self.burning_area = burning_area
        self.m_dot = m_dot
        self.exit_pressure = exit_pressure
        self.elapsed_time = elapsed_time
        self.thrust = thrust
        self.method = "Bates Analytical"

    def fast_marching_method_2d(self, fidelity, dt):
        map_ratio = self.motor_diameter / fidelity  # in/delta
        x, y = FMM2d.make_coordinate_array(self.motor_diameter, fidelity)

        motor = np.full_like(x, True, dtype=bool)
        motor[x ** 2 + y ** 2 < (self.core_diameter / 2) ** 2] = False

        motor = FMM2d.add_perimeter_mask(motor, x, y, self.motor_diameter)
        regression_map = FMM2d.calculate_regression(motor, map_ratio)
        FMM2d.plot_regression(motor, self.motor_diameter, fidelity)
        chamber_pressure, burn_rate, regression_depth, burning_area, m_dot, exit_pressure, elapsed_time, thrust = \
            FMM2d.burn_motor(regression_map, self.throat_area, self.a, self.density, self.c_star, self.gamma, self.n,
                             self.exit_mach, self.exit_velocity, self.exit_area, dt, map_ratio, self.motor_height,
                             self.core_diameter, self.motor_diameter, fidelity)
        self.chamber_pressure = chamber_pressure
        self.burn_rate = burn_rate
        self.regression_depth = regression_depth
        self.burning_area = burning_area
        self.m_dot = m_dot
        self.exit_pressure = exit_pressure
        self.elapsed_time = elapsed_time
        self.thrust = thrust
        self.method = "Bates FMM 2d"

    def fast_marching_method_3d(self, fidelity, dt):
        fidelity_z = int(fidelity * self.motor_height / self.motor_diameter)  # discrete elements delta
        volume_ratio = self.motor_diameter/fidelity  # in/delta
        x, y, z = FMM3d.make_coordinate_array(self.motor_diameter, self.motor_height, fidelity, fidelity_z)

        motor = -1 * np.ones((fidelity, fidelity, fidelity_z))
        for i in range(0, fidelity_z):
            dummy = -1 * np.ones((fidelity, fidelity))
            dummy[x[:, :, 0] ** 2 + y[:, :, 0] ** 2 < (self.core_diameter / 2) ** 2] = 0
            motor[:, :, fidelity_z - i - 1] = dummy

        motor = FMM3d.add_uninhibited_ends(motor, fidelity_z, False, False)
        motor = FMM3d.add_perimeter_mask(motor, self.motor_diameter, fidelity_z, x, y)
        mc_mask = FMM3d.make_marching_cubes_mask(motor, self.motor_diameter, fidelity_z, x, y)

        regression_map = FMM3d.calculate_regression(motor)

        chamber_pressure, burn_rate, regression_depth, burning_area, m_dot, exit_pressure, elapsed_time, thrust = \
            FMM3d.burn_motor(regression_map, self.throat_area, self.a, self.density, self.c_star, self.gamma, self.n,
                             self.exit_mach, self.exit_velocity, self.exit_area, dt, mc_mask, volume_ratio)
        self.chamber_pressure = chamber_pressure
        self.burn_rate = burn_rate
        self.regression_depth = regression_depth
        self.burning_area = burning_area
        self.m_dot = m_dot
        self.exit_pressure = exit_pressure
        self.elapsed_time = elapsed_time
        self.thrust = thrust
        self.method = "Bates FMM 3d"


class Star:
    def __init__(self, propellant, motor_height, motor_diameter, bore_radius, web_radius, fillet_radius, epsilon,
                 number_of_points, exit_area, throat_area, burn_top, burn_bottom):
        self.burn_top = burn_top
        self.burn_bottom = burn_bottom
        self.motor_height = motor_height
        self.motor_diameter = motor_diameter
        self.exit_area = exit_area
        self.throat_area = throat_area

        self.Ro = motor_diameter/2
        self.Ri = bore_radius
        self.Rp = web_radius
        self.f = fillet_radius
        self.epsilon = epsilon
        self.N = number_of_points
        self.theta = 2 * np.arctan(self.Rp * np.sin(np.pi*epsilon/self.N) * np.tan(np.pi*epsilon/self.N) /
                                   (self.Rp * np.sin(np.pi*epsilon/self.N) - self.Ri * np.tan(np.pi*epsilon/self.N)))

        self.density = propellant.density
        self.a = propellant.a
        self.n = propellant.n
        self.T = propellant.T
        self.gamma = propellant.gamma
        self.molar_mass = propellant.molar_mass
        self.c_star = propellant.c_star
        self.R = propellant.R

        self.exit_mach = burn.calculate_exit_mach(self.gamma, self.exit_area, self.throat_area)
        self.exit_temp = burn.calculate_nozzle_exit_temp(self.T, self.gamma, self.exit_mach)
        self.exit_velocity = burn.calculate_nozzle_exit_velocity(self.exit_mach, self.gamma, self.R, self.exit_temp)

        # initializing variables to be stored after burn method
        self.chamber_pressure = 0
        self.burn_rate = 0
        self.regression_depth = 0
        self.burning_area = 0
        self.m_dot = 0
        self.exit_pressure = 0
        self.elapsed_time = 0
        self.thrust = 0
        self.method = 0

    # def analytical_method(self, dt):
    #     chamber_pressure, burn_rate, regression_depth, burning_area, m_dot, exit_pressure, elapsed_time, thrust = \
    #         analytical.star_burn(self.Ro, self.Rp, self.Ri, self.epsilon, self.N, self.f, self.a, self.density,
    #                              self.c_star, self.gamma, self.n, self.exit_mach, self.exit_velocity, self.exit_area,
    #                              dt, self.motor_height, self.motor_diameter, self.throat_area, self.burn_top,
    #                              self.burn_bottom)
    #     self.chamber_pressure = chamber_pressure
    #     self.burn_rate = burn_rate
    #     self.regression_depth = regression_depth
    #     self.burning_area = burning_area
    #     self.m_dot = m_dot
    #     self.exit_pressure = exit_pressure
    #     self.elapsed_time = elapsed_time
    #     self.thrust = thrust
    #     self.method = "Star Analytical"

    def fast_marching_method_2d(self, fidelity, dt):
        map_ratio = self.motor_diameter / fidelity  # in/delta
        x, y = FMM2d.make_coordinate_array(self.motor_diameter, fidelity)

        motor = np.full_like(x, False, dtype=bool)

        for i in range(0, self.N):
            x, y = FMM2d.make_coordinate_array(self.motor_diameter, fidelity)
            x, y = FMM2d.rotate_matrix(x, y, 2 * np.pi / self.N * i)

            # top pointy part
            a = np.tan(self.theta/2)
            h = a * self.Ri
            l1 = np.full_like(x, False, dtype=bool)
            l1[y < a * x - h] = True

            # bottom pointy part
            a = -np.tan(self.theta/2)
            h = a * self.Ri
            l2 = np.full_like(x, False, dtype=bool)
            l2[y > a * x - h] = True

            # top flat part
            a = -np.tan(np.pi/2 - np.pi/self.N)
            h = a * self.Rp/np.cos(np.pi/self.N)
            l3 = np.full_like(x, False, dtype=bool)
            l3[y > a * x - h] = True

            # bottom flat part
            a = np.tan(np.pi/2 - np.pi/self.N)
            h = a * self.Rp/np.cos(np.pi/self.N)
            l4 = np.full_like(x, False, dtype=bool)
            l4[y < a * x - h] = True

            d1 = np.full_like(x, False, dtype=bool)
            d1[np.logical_and(l1, l2)] = True

            d2 = np.full_like(x, False, dtype=bool)
            d2[np.logical_or(l3, l4)] = True

            motor[np.logical_or(d1, d2)] = True

        motor = FMM2d.add_perimeter_mask(motor, x, y, self.motor_diameter)
        regression_map = FMM2d.calculate_regression(motor, map_ratio)
        FMM2d.plot_regression(motor, self.motor_diameter, fidelity)
        chamber_pressure, burn_rate, regression_depth, burning_area, m_dot, exit_pressure, elapsed_time, thrust = \
            FMM2d.burn_motor(regression_map, self.throat_area, self.a, self.density, self.c_star, self.gamma, self.n,
                             self.exit_mach, self.exit_velocity, self.exit_area, dt, map_ratio, self.motor_height,
                             self.Ri, self.motor_diameter, fidelity)
        self.chamber_pressure = chamber_pressure
        self.burn_rate = burn_rate
        self.regression_depth = regression_depth
        self.burning_area = burning_area
        self.m_dot = m_dot
        self.exit_pressure = exit_pressure
        self.elapsed_time = elapsed_time
        self.thrust = thrust
        self.method = "Star FMM 2d"

    # def fast_marching_method_3d(self, fidelity, dt):
    #     pass


class WagonWheel:
    def __init__(self, propellant, motor_height, motor_diameter, bore_radius, web_radius, fillet_radius, epsilon,
                 number_of_points, theta, exit_area, throat_area, burn_top, burn_bottom):
        self.burn_top = burn_top
        self.burn_bottom = burn_bottom
        self.motor_height = motor_height
        self.motor_diameter = motor_diameter
        self.exit_area = exit_area
        self.throat_area = throat_area

        self.Ro = motor_diameter/2
        self.Ri = bore_radius
        self.Rp = web_radius
        self.f = fillet_radius
        self.epsilon = epsilon
        self.N = number_of_points
        self.theta = theta

        self.density = propellant.density
        self.a = propellant.a
        self.n = propellant.n
        self.T = propellant.T
        self.gamma = propellant.gamma
        self.molar_mass = propellant.molar_mass
        self.c_star = propellant.c_star
        self.R = propellant.R

        self.exit_mach = burn.calculate_exit_mach(self.gamma, self.exit_area, self.throat_area)
        self.exit_temp = burn.calculate_nozzle_exit_temp(self.T, self.gamma, self.exit_mach)
        self.exit_velocity = burn.calculate_nozzle_exit_velocity(self.exit_mach, self.gamma, self.R, self.exit_temp)

        # initializing variables to be stored after burn method
        self.chamber_pressure = 0
        self.burn_rate = 0
        self.regression_depth = 0
        self.burning_area = 0
        self.m_dot = 0
        self.exit_pressure = 0
        self.elapsed_time = 0
        self.thrust = 0
        self.method = 0

    def analytical_method(self, dt):
        chamber_pressure, burn_rate, regression_depth, burning_area, m_dot, exit_pressure, elapsed_time, thrust = \
            analytical.wagon_wheel_burn(self.Ro, self.Rp, self.Ri, self.f, self.N, self.epsilon, self.theta, self.a,
                                        self.density, self.c_star, self.gamma, self.n, self.exit_mach,
                                        self.exit_velocity, self.exit_area, dt, self.motor_height, self.motor_diameter,
                                        self.throat_area, self.burn_top, self.burn_bottom)
        self.chamber_pressure = chamber_pressure
        self.burn_rate = burn_rate
        self.regression_depth = regression_depth
        self.burning_area = burning_area
        self.m_dot = m_dot
        self.exit_pressure = exit_pressure
        self.elapsed_time = elapsed_time
        self.thrust = thrust
        self.method = "Wagon Wheel Analytical"

    def fast_marching_method_2d(self, fidelity, dt):
        map_ratio = self.motor_diameter / fidelity  # in/delta
        x, y = FMM2d.make_coordinate_array(self.motor_diameter, fidelity)

        motor = np.full_like(x, True, dtype=bool)
        motor[x ** 2 + y ** 2 < self.Rp ** 2] = False

        for i in range(0, self.N):
            x, y = FMM2d.make_coordinate_array(self.motor_diameter, fidelity)
            x, y = FMM2d.rotate_matrix(x, y, 2 * np.pi / self.N * i)

            # top horizontal
            h = self.Rp * np.sin(np.pi * self.epsilon / self.N) - self.f
            l1 = np.full_like(x, False, dtype=bool)
            l1[y < h] = True

            # bottom horizontal
            h = self.Rp * np.sin(np.pi * self.epsilon / self.N) - self.f
            l2 = np.full_like(x, False, dtype=bool)
            l2[y > - h] = True

            # negative slope
            a = -np.tan(self.theta/2)
            h = -self.Ri * a
            l3 = np.full_like(x, False, dtype=bool)
            l3[y > a * x + h] = True

            # positive slope
            a = np.tan(self.theta/2)
            h = -self.Ri * a
            l4 = np.full_like(x, False, dtype=bool)
            l4[y < a * x + h] = True

            d1 = np.full_like(x, False, dtype=bool)
            d1[np.logical_and(l1, l2)] = True

            d2 = np.full_like(x, False, dtype=bool)
            d2[np.logical_and(l3, l4)] = True

            dummy = np.full_like(x, False, dtype=bool)
            dummy[np.logical_and(d1, d2)] = True

            motor[np.logical_or(motor, dummy)] = True

        motor = FMM2d.add_perimeter_mask(motor, x, y, self.motor_diameter)
        regression_map = FMM2d.calculate_regression(motor, map_ratio)
        FMM2d.plot_regression(motor, self.motor_diameter, fidelity)
        chamber_pressure, burn_rate, regression_depth, burning_area, m_dot, exit_pressure, elapsed_time, thrust = \
            FMM2d.burn_motor(regression_map, self.throat_area, self.a, self.density, self.c_star, self.gamma, self.n,
                             self.exit_mach, self.exit_velocity, self.exit_area, dt, map_ratio, self.motor_height,
                             self.Ri, self.motor_diameter, fidelity)
        self.chamber_pressure = chamber_pressure
        self.burn_rate = burn_rate
        self.regression_depth = regression_depth
        self.burning_area = burning_area
        self.m_dot = m_dot
        self.exit_pressure = exit_pressure
        self.elapsed_time = elapsed_time
        self.thrust = thrust
        self.method = "Wagon Wheel FMM 2d"

    # def fast_marching_method_3d(self, fidelity, dt):
    #     fidelity_z = int(fidelity * self.motor_height / self.motor_diameter)  # discrete elements delta
    #     volume_ratio = self.motor_diameter / fidelity  # in/delta
    #     x, y, z = FMM3d.make_coordinate_array(self.motor_diameter, self.motor_height, fidelity, fidelity_z)
    #
    #     motor = -1 * np.ones((fidelity, fidelity, fidelity_z))
    #     slice = np.full_like(x[:, :, 0], True, dtype=bool)
    #     slice[x[:, :, 0] ** 2 + y[:, :, 0] ** 2 < self.Rp ** 2] = False
    #
    #     for i in range(0, self.N):
    #         x, y = FMM2d.make_coordinate_array(self.motor_diameter, fidelity)
    #         x, y = FMM2d.rotate_matrix(x, y, 2 * np.pi / self.N * i)
    #
    #         # top horizontal
    #         h = self.Rp * np.sin(np.pi * self.epsilon / self.N) - self.f
    #         l1 = np.full_like(x, False, dtype=bool)
    #         l1[y < h] = True
    #
    #         # bottom horizontal
    #         h = self.Rp * np.sin(np.pi * self.epsilon / self.N) - self.f
    #         l2 = np.full_like(x, False, dtype=bool)
    #         l2[y > - h] = True
    #
    #         # negative slope
    #         a = -np.tan(self.theta/2)
    #         h = -self.Ri * a
    #         l3 = np.full_like(x, False, dtype=bool)
    #         l3[y > a * x + h] = True
    #
    #         # positive slope
    #         a = np.tan(self.theta/2)
    #         h = -self.Ri * a
    #         l4 = np.full_like(x, False, dtype=bool)
    #         l4[y < a * x + h] = True
    #
    #         d1 = np.full_like(x, False, dtype=bool)
    #         d1[np.logical_and(l1, l2)] = True
    #
    #         d2 = np.full_like(x, False, dtype=bool)
    #         d2[np.logical_and(l3, l4)] = True
    #
    #         dummy = np.full_like(x, False, dtype=bool)
    #         dummy[np.logical_and(d1, d2)] = True
    #
    #         slice[np.logical_or(slice, dummy)] = True
    #
    #     for i in range(0, fidelity_z):
    #         motor[:, :, fidelity_z - i - 1] = slice
    #
    #     x, y, z = FMM3d.make_coordinate_array(self.motor_diameter, self.motor_height, fidelity, fidelity_z)
    #     motor = FMM3d.add_uninhibited_ends(motor, fidelity_z, False, False)
    #     motor = FMM3d.add_perimeter_mask(motor, self.motor_diameter, fidelity_z, x, y)
    #     mc_mask = FMM3d.make_marching_cubes_mask(motor, self.motor_diameter, fidelity_z, x, y)
    #
    #     regression_map = FMM3d.calculate_regression(motor)
    #
    #     chamber_pressure, burn_rate, regression_depth, burning_area, m_dot, exit_pressure, elapsed_time, thrust = \
    #         FMM3d.burn_motor(regression_map, self.throat_area, self.a, self.density, self.c_star, self.gamma, self.n,
    #                          self.exit_mach, self.exit_velocity, self.exit_area, dt, mc_mask, volume_ratio)
    #     self.chamber_pressure = chamber_pressure
    #     self.burn_rate = burn_rate
    #     self.regression_depth = regression_depth
    #     self.burning_area = burning_area
    #     self.m_dot = m_dot
    #     self.exit_pressure = exit_pressure
    #     self.elapsed_time = elapsed_time
    #     self.thrust = thrust
    #     self.method = "Wagon Wheel FMM 3d"


class Finocyl:
    def __init__(self, propellant, motor_height, motor_diameter, bore_radius, fin_width, fin_length,
                 number_of_fins, exit_area, throat_area, burn_top, burn_bottom):
        self.burn_top = burn_top
        self.burn_bottom = burn_bottom
        self.motor_height = motor_height
        self.motor_diameter = motor_diameter
        self.exit_area = exit_area
        self.throat_area = throat_area

        self.Rp = motor_diameter/2
        self.fin_width = fin_width
        self.fin_length = fin_length
        self.Rb = bore_radius
        self.N = number_of_fins

        self.density = propellant.density
        self.a = propellant.a
        self.n = propellant.n
        self.T = propellant.T
        self.gamma = propellant.gamma
        self.molar_mass = propellant.molar_mass
        self.c_star = propellant.c_star
        self.R = propellant.R

        self.exit_mach = burn.calculate_exit_mach(self.gamma, self.exit_area, self.throat_area)
        self.exit_temp = burn.calculate_nozzle_exit_temp(self.T, self.gamma, self.exit_mach)
        self.exit_velocity = burn.calculate_nozzle_exit_velocity(self.exit_mach, self.gamma, self.R, self.exit_temp)

        # initializing variables to be stored after burn method
        self.chamber_pressure = 0
        self.burn_rate = 0
        self.regression_depth = 0
        self.burning_area = 0
        self.m_dot = 0
        self.exit_pressure = 0
        self.elapsed_time = 0
        self.thrust = 0
        self.method = 0

    def fast_marching_method_2d(self, fidelity, dt):
        map_ratio = self.motor_diameter / fidelity  # in/delta
        x, y = FMM2d.make_coordinate_array(self.motor_diameter, fidelity)

        motor = np.full_like(x, True, dtype=bool)
        motor[x ** 2 + y ** 2 < self.Rb ** 2] = False

        for i in range(0, self.N):
            x, y = FMM2d.make_coordinate_array(self.motor_diameter, fidelity)
            x, y = FMM2d.rotate_matrix(x, y, 2 * np.pi / self.N * i)

            # edge bore
            c1 = np.full_like(x, True, dtype=bool)
            c1[(x - self.fin_length) ** 2 + y ** 2 < (self.fin_width / 2) ** 2] = False

            motor[np.logical_or(~motor, ~c1)] = False

            # top horizontal
            h = self.fin_width/2
            l1 = np.full_like(x, True, dtype=bool)
            l1[y < h] = False

            # bottom horizontal
            h = self.fin_width/2
            l2 = np.full_like(x, True, dtype=bool)
            l2[y > - h] = False

            d1 = np.full_like(x, False, dtype=bool)
            d1[np.logical_or(l1, l2)] = True

            # back vertical
            h = 0
            l3 = np.full_like(x, True, dtype=bool)
            l3[x > h] = False

            # front vertical
            h = self.fin_length
            l4 = np.full_like(x, True, dtype=bool)
            l4[x < h] = False

            d2 = np.full_like(x, False, dtype=bool)
            d2[np.logical_or(l3, l4)] = True

            square = np.full_like(x, True, dtype=bool)
            square[np.logical_and(~d1, ~d2)] = False

            motor[np.logical_or(~motor, ~square)] = False

        motor = FMM2d.add_perimeter_mask(motor, x, y, self.motor_diameter)
        regression_map = FMM2d.calculate_regression(motor, map_ratio)
        FMM2d.plot_regression(motor, self.motor_diameter, fidelity)
        chamber_pressure, burn_rate, regression_depth, burning_area, m_dot, exit_pressure, elapsed_time, thrust = \
            FMM2d.burn_motor(regression_map, self.throat_area, self.a, self.density, self.c_star, self.gamma, self.n,
                             self.exit_mach, self.exit_velocity, self.exit_area, dt, map_ratio, self.motor_height,
                             self.Rb, self.motor_diameter, fidelity)
        self.chamber_pressure = chamber_pressure
        self.burn_rate = burn_rate
        self.regression_depth = regression_depth
        self.burning_area = burning_area
        self.m_dot = m_dot
        self.exit_pressure = exit_pressure
        self.elapsed_time = elapsed_time
        self.thrust = thrust
        self.method = "Finocyl FMM 2d"


class EndBurner:
    def __init__(self, propellant, motor_height, motor_diameter, exit_area, throat_area):
        self.motor_height = motor_height
        self.motor_diameter = motor_diameter
        self.exit_area = exit_area
        self.throat_area = throat_area

        self.density = propellant.density
        self.a = propellant.a
        self.n = propellant.n
        self.T = propellant.T
        self.gamma = propellant.gamma
        self.molar_mass = propellant.molar_mass
        self.c_star = propellant.c_star
        self.R = propellant.R

        self.exit_mach = burn.calculate_exit_mach(self.gamma, self.exit_area, self.throat_area)
        self.exit_temp = burn.calculate_nozzle_exit_temp(self.T, self.gamma, self.exit_mach)
        self.exit_velocity = burn.calculate_nozzle_exit_velocity(self.exit_mach, self.gamma, self.R, self.exit_temp)

        # initializing variables to be stored after burn method
        self.chamber_pressure = 0
        self.burn_rate = 0
        self.regression_depth = 0
        self.burning_area = 0
        self.m_dot = 0
        self.exit_pressure = 0
        self.elapsed_time = 0
        self.thrust = 0
        self.method = 0

    def analytical_method(self, dt):
        chamber_pressure, burn_rate, regression_depth, burning_area, m_dot, exit_pressure, elapsed_time, thrust = \
            analytical.end_burn(self.a, self.density, self.c_star, self.gamma, self.n, self.exit_mach,
                                self.exit_velocity, self.exit_area, self.throat_area, dt, self.motor_height,
                                self.motor_diameter)
        self.chamber_pressure = chamber_pressure
        self.burn_rate = burn_rate
        self.regression_depth = regression_depth
        self.burning_area = burning_area
        self.m_dot = m_dot
        self.exit_pressure = exit_pressure
        self.elapsed_time = elapsed_time
        self.thrust = thrust
        self.method = "End Burner Analytical"

    def fast_marching_method_3d(self, fidelity):
        pass


class CustomImport:
    def __init__(self, propellant):
        pass

    def fast_marching_method_2d(self, fidelity):
        pass

    def fast_marching_method_3d(self, fidelity):
        pass


class Cone:
    def __init__(self, propellant):
        pass

    def analytical_method(self):
        pass

    def fast_marching_method_3d(self, fidelity):
        pass

