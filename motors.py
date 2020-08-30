import geometry
import grains


class ExampleBates:
    def __init__(self, propellant, dt, fidelity=100, burn_top=0, burn_bottom=0):
        self.dt = dt
        self.fidelity = fidelity

        # vehicle dimensions
        motor_height = 3  # m
        motor_diameter = 0.5  # m
        exit_diameter = 0.4
        expansion_ratio = 10  # m
        exit_area = geometry.circle_area(exit_diameter)  # m^2
        throat_area = exit_area / expansion_ratio  # m^2
        core_diameter = 0.25  # m

        # instantiate grain class
        self.my_grain = grains.Bates(propellant, motor_height, motor_diameter, core_diameter, exit_area, throat_area, burn_top, burn_bottom)

        # initialize results
        self.chamber_pressure = 0
        self.burn_rate = 0
        self.regression_depth = 0
        self.burning_area = 0
        self.m_dot = 0
        self.exit_pressure = 0
        self.elapsed_time = 0
        self.thrust = 0
        self.method = 0

    def analytical_method(self):
        self.my_grain.analytical_method(self.dt)
        self.chamber_pressure = self.my_grain.chamber_pressure
        self.burn_rate = self.my_grain.burn_rate
        self.regression_depth = self.my_grain.regression_depth
        self.burning_area = self.my_grain.burning_area
        self.m_dot = self.my_grain.m_dot
        self.exit_pressure = self.my_grain.exit_pressure
        self.elapsed_time = self.my_grain.elapsed_time
        self.thrust = self.my_grain.thrust
        self.method = self.my_grain.method

    def fast_marching_2d(self):
        self.my_grain.fast_marching_method_2d(self.fidelity, self.dt)
        self.chamber_pressure = self.my_grain.chamber_pressure
        self.burn_rate = self.my_grain.burn_rate
        self.regression_depth = self.my_grain.regression_depth
        self.burning_area = self.my_grain.burning_area
        self.m_dot = self.my_grain.m_dot
        self.exit_pressure = self.my_grain.exit_pressure
        self.elapsed_time = self.my_grain.elapsed_time
        self.thrust = self.my_grain.thrust
        self.method = self.my_grain.method

    def fast_marching_3d(self):
        self.my_grain.fast_marching_method_3d(self.fidelity, self.dt)
        self.chamber_pressure = self.my_grain.chamber_pressure
        self.burn_rate = self.my_grain.burn_rate
        self.regression_depth = self.my_grain.regression_depth
        self.burning_area = self.my_grain.burning_area
        self.m_dot = self.my_grain.m_dot
        self.exit_pressure = self.my_grain.exit_pressure
        self.elapsed_time = self.my_grain.elapsed_time
        self.thrust = self.my_grain.thrust
        self.method = self.my_grain.method


class ExampleStar:
    def __init__(self, propellant, dt, fidelity=100, burn_top=0, burn_bottom=0):
        self.dt = dt
        self.fidelity = fidelity

        # vehicle dimensions
        motor_height = 3  # m
        motor_diameter = 0.5  # m
        exit_diameter = 0.4
        expansion_ratio = 10  # m
        exit_area = geometry.circle_area(exit_diameter)  # m^2
        throat_area = exit_area / expansion_ratio  # m^2

        bore_radius = 0.05
        web_radius = 0.15
        fillet_radius = 0.0001
        epsilon = 0.4
        number_of_points = 6

        # instantiate grain class
        self.my_grain = grains.Star(propellant, motor_height, motor_diameter, bore_radius, web_radius, fillet_radius,
                                    epsilon, number_of_points, exit_area, throat_area, burn_top, burn_bottom)

        # initialize results
        self.chamber_pressure = 0
        self.burn_rate = 0
        self.regression_depth = 0
        self.burning_area = 0
        self.m_dot = 0
        self.exit_pressure = 0
        self.elapsed_time = 0
        self.thrust = 0
        self.method = 0

    def fast_marching_2d(self):
        self.my_grain.fast_marching_method_2d(self.fidelity, self.dt)
        self.chamber_pressure = self.my_grain.chamber_pressure
        self.burn_rate = self.my_grain.burn_rate
        self.regression_depth = self.my_grain.regression_depth
        self.burning_area = self.my_grain.burning_area
        self.m_dot = self.my_grain.m_dot
        self.exit_pressure = self.my_grain.exit_pressure
        self.elapsed_time = self.my_grain.elapsed_time
        self.thrust = self.my_grain.thrust
        self.method = self.my_grain.method


class ExampleStar2:
    def __init__(self, propellant, dt, fidelity=100, burn_top=0, burn_bottom=0):
        self.dt = dt
        self.fidelity = fidelity

        # vehicle dimensions
        motor_height = 3  # m
        motor_diameter = 0.5  # m
        exit_diameter = 0.4
        expansion_ratio = 10  # m
        exit_area = geometry.circle_area(exit_diameter)  # m^2
        throat_area = exit_area / expansion_ratio  # m^2

        bore_radius = 0.05
        web_radius = 0.15
        fillet_radius = 0.0001
        epsilon = 0.998
        number_of_points = 6
        # instantiate grain class
        self.my_grain = grains.Star(propellant, motor_height, motor_diameter, bore_radius, web_radius, fillet_radius,
                                    epsilon, number_of_points, exit_area, throat_area, burn_top, burn_bottom)

        # initialize results
        self.chamber_pressure = 0
        self.burn_rate = 0
        self.regression_depth = 0
        self.burning_area = 0
        self.m_dot = 0
        self.exit_pressure = 0
        self.elapsed_time = 0
        self.thrust = 0
        self.method = 0

    def fast_marching_2d(self):
        self.my_grain.fast_marching_method_2d(self.fidelity, self.dt)
        self.chamber_pressure = self.my_grain.chamber_pressure
        self.burn_rate = self.my_grain.burn_rate
        self.regression_depth = self.my_grain.regression_depth
        self.burning_area = self.my_grain.burning_area
        self.m_dot = self.my_grain.m_dot
        self.exit_pressure = self.my_grain.exit_pressure
        self.elapsed_time = self.my_grain.elapsed_time
        self.thrust = self.my_grain.thrust
        self.method = self.my_grain.method


class ExampleWagon:
    def __init__(self, propellant, dt, fidelity=100, burn_top=0, burn_bottom=0):
        self.dt = dt
        self.fidelity = fidelity

        # vehicle dimensions
        motor_height = 3  # m
        motor_diameter = 0.5  # m
        exit_diameter = 0.4
        expansion_ratio = 10  # m
        exit_area = geometry.circle_area(exit_diameter)  # m^2
        throat_area = exit_area / expansion_ratio  # m^2

        bore_radius = 0.05  # Ri
        web_radius = 0.15  # Rp
        fillet_radius = 0.001     # f
        epsilon = 0.23
        number_of_points = 6   # N
        theta = 2.8

        # instantiate grain class
        self.my_grain = grains.WagonWheel(propellant, motor_height, motor_diameter, bore_radius, web_radius,
                                          fillet_radius, epsilon, number_of_points, theta, exit_area, throat_area,
                                          burn_top, burn_bottom)

        # initialize results
        self.chamber_pressure = 0
        self.burn_rate = 0
        self.regression_depth = 0
        self.burning_area = 0
        self.m_dot = 0
        self.exit_pressure = 0
        self.elapsed_time = 0
        self.thrust = 0
        self.method = 0

    def analytical_method(self):
        self.my_grain.analytical_method(self.dt)
        self.chamber_pressure = self.my_grain.chamber_pressure
        self.burn_rate = self.my_grain.burn_rate
        self.regression_depth = self.my_grain.regression_depth
        self.burning_area = self.my_grain.burning_area
        self.m_dot = self.my_grain.m_dot
        self.exit_pressure = self.my_grain.exit_pressure
        self.elapsed_time = self.my_grain.elapsed_time
        self.thrust = self.my_grain.thrust
        self.method = self.my_grain.method

    def fast_marching_2d(self):
        self.my_grain.fast_marching_method_2d(self.fidelity, self.dt)
        self.chamber_pressure = self.my_grain.chamber_pressure
        self.burn_rate = self.my_grain.burn_rate
        self.regression_depth = self.my_grain.regression_depth
        self.burning_area = self.my_grain.burning_area
        self.m_dot = self.my_grain.m_dot
        self.exit_pressure = self.my_grain.exit_pressure
        self.elapsed_time = self.my_grain.elapsed_time
        self.thrust = self.my_grain.thrust
        self.method = self.my_grain.method


class ExampleFinocyl:
    def __init__(self, propellant, dt, fidelity=100, burn_top=0, burn_bottom=0):
        self.dt = dt
        self.fidelity = fidelity

        # vehicle dimensions
        motor_height = 3  # m
        motor_diameter = 0.5  # m
        exit_diameter = 0.4
        expansion_ratio = 10  # m
        exit_area = geometry.circle_area(exit_diameter)  # m^2
        throat_area = exit_area / expansion_ratio  # m^2

        bore_radius = 0.05  # m
        fin_width = 0.03  # m
        fin_length = 0.15  # m
        number_of_fins = 4

        # instantiate grain class
        self.my_grain = grains.Finocyl(propellant, motor_height, motor_diameter, bore_radius, fin_width, fin_length,
                                       number_of_fins, exit_area, throat_area, burn_top, burn_bottom)

        # initialize results
        self.chamber_pressure = 0
        self.burn_rate = 0
        self.regression_depth = 0
        self.burning_area = 0
        self.m_dot = 0
        self.exit_pressure = 0
        self.elapsed_time = 0
        self.thrust = 0
        self.method = 0

    def fast_marching_2d(self):
        self.my_grain.fast_marching_method_2d(self.fidelity, self.dt)
        self.chamber_pressure = self.my_grain.chamber_pressure
        self.burn_rate = self.my_grain.burn_rate
        self.regression_depth = self.my_grain.regression_depth
        self.burning_area = self.my_grain.burning_area
        self.m_dot = self.my_grain.m_dot
        self.exit_pressure = self.my_grain.exit_pressure
        self.elapsed_time = self.my_grain.elapsed_time
        self.thrust = self.my_grain.thrust
        self.method = self.my_grain.method


# class ExampleEndBurner:
#     def __init__(self, propellant, dt, fidelity=100):
#         self.dt = dt
#         self.fidelity = fidelity
#
#         # vehicle dimensions
#         motor_height = 3  # m
#         motor_diameter = 2  # m
#         exit_diameter = 0.4
#         expansion_ratio = 10  # m
#         exit_area = geometry.circle_area(exit_diameter)  # m^2
#         throat_area = exit_area / expansion_ratio  # m^2
#
#         # instantiate grain class
#         self.my_grain = grains.EndBurner(propellant, motor_height, motor_diameter, exit_area, throat_area)
#
#         # initialize results
#         self.chamber_pressure = 0
#         self.burn_rate = 0
#         self.regression_depth = 0
#         self.burning_area = 0
#         self.m_dot = 0
#         self.exit_pressure = 0
#         self.elapsed_time = 0
#         self.thrust = 0
#         self.method = 0
#
#     def analytical_method(self):
#         self.my_grain.analytical_method(self.dt)
#         self.chamber_pressure = self.my_grain.chamber_pressure
#         self.burn_rate = self.my_grain.burn_rate
#         self.regression_depth = self.my_grain.regression_depth
#         self.burning_area = self.my_grain.burning_area
#         self.m_dot = self.my_grain.m_dot
#         self.exit_pressure = self.my_grain.exit_pressure
#         self.elapsed_time = self.my_grain.elapsed_time
#         self.thrust = self.my_grain.thrust
#         self.method = self.my_grain.method