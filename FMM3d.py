import numpy as np
import skfmm
import burn
from skimage import measure
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def check_symmetry(mesh, fidelity_z):
    if np.all(mesh[:, :, 1] == mesh[:, :, fidelity_z - 2]):
        check = True
    else:
        check = False
    return check


def make_coordinate_array(motor_diameter, motor_height, fidelity, fidelity_z):
    x, y, z = np.meshgrid(np.linspace(-motor_diameter / 2, motor_diameter / 2, fidelity),
                          np.linspace(-motor_diameter / 2, motor_diameter / 2, fidelity),
                          np.linspace(-motor_height / 2, motor_height / 2, fidelity_z))
    return x, y, z


def add_uninhibited_ends(mesh, fidelity_z, burn_top, burn_bottom):
    if burn_top:
        mesh[:, :, 0] = 0
    if burn_bottom:
        mesh[:, :, fidelity_z - 1] = 0
    return mesh


def add_perimeter_mask(mesh, motor_diameter, fidelity_z, x, y):
    mask = np.full_like(mesh, False, dtype=bool)
    dummy = np.full_like(mesh[:, :, 0], False, dtype=bool)
    dummy[x[:, :, 0] ** 2 + y[:, :, 0] ** 2 > (motor_diameter / 1.9) ** 2] = True
    for i in range(0, fidelity_z):
        mask[:, :, i] = dummy
    masked_mesh = np.ma.MaskedArray(mesh, mask)
    return masked_mesh


def make_marching_cubes_mask(mesh, motor_diameter, fidelity_z, x, y):
    mc_mask = np.full_like(mesh, True, dtype=bool)
    dummy = np.full_like(mesh[:, :, 0], True, dtype=bool)
    dummy[x[:, :, 0] ** 2 + y[:, :, 0] ** 2 > (motor_diameter / 2) ** 2] = False
    for i in range(0, fidelity_z):
        mc_mask[:, :, i] = dummy
    return mc_mask


def calculate_regression(mesh):
    return skfmm.distance(mesh)


def calculate_burning_area(regression_map, regression_depth, mc_mask, volume_ratio):
    regression_depth = regression_depth / volume_ratio
    verts, faces, norm, line = measure.marching_cubes(regression_map, level=regression_depth, mask=mc_mask, spacing=
                                                      (volume_ratio, volume_ratio, volume_ratio))  # (hide false values)
    burning_area = measure.mesh_surface_area(verts, faces)
    return burning_area


def burn_motor(regression_map, throat_area, a, density, c_star, gamma, n, exit_mach, exit_velocity, exit_area, dt, mc_mask, volume_ratio):
    elapsed_time = 0  # start the burn at zero seconds
    regression_depth = 0.00001  # m
    thrust_initial = 10
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
    burning_area = calculate_burning_area(regression_map, -0.001, mc_mask, volume_ratio)
    sim_start = time.time()
    while burning:
        count = count + 1
        chamber_pressure = burn.calculate_chamber_pressure(burning_area, throat_area, a, density, c_star, n)  # obsolete
        burn_rate = burn.calculate_burn_rate(a, chamber_pressure, n)
        regression_depth = burn.calculate_new_regression_dist(regression_depth, burn_rate, dt)
        burning_area = calculate_burning_area(regression_map, -regression_depth, mc_mask, volume_ratio)
        m_dot = burn.calculate_mass_flow_rate(burning_area, burn_rate, density)
        exit_pressure = burn.calculate_nozzle_exit_pressure(chamber_pressure, gamma, exit_mach)
        thrust = burn.calculate_vacuum_thrust(m_dot, exit_velocity, exit_area, exit_pressure)
        elapsed_time = elapsed_time + dt

        if count == 1:
            thrust_initial = thrust

        chamber_pressure_list = np.append(chamber_pressure_list, chamber_pressure)
        burn_rate_list = np.append(burn_rate_list, burn_rate)
        regression_depth_list = np.append(regression_depth_list,regression_depth)
        burning_area_list = np.append(burning_area_list, burning_area)
        m_dot_list = np.append(m_dot_list, m_dot)
        exit_pressure_list = np.append(exit_pressure_list, exit_pressure)
        elapsed_time_list = np.append(elapsed_time_list, elapsed_time)
        thrust_list = np.append(thrust_list, thrust)
        if thrust < thrust_initial * 0.1:
            burning = False
        if count >= 1000:
            burning = False
        if time.time()-sim_start > 60:
            burning = False
    return chamber_pressure_list, burn_rate_list, regression_depth_list, burning_area_list, m_dot_list, exit_pressure_list, elapsed_time_list, thrust_list


def make_fast_marching_mask(self, motor):
    Xarray, Yarray, Zarray = self.make_coordinate_array()
    fmm_mask = np.full_like(motor, True, dtype=bool)
    dummy = np.full_like(motor[:, :, 0], True, dtype=bool)
    dummy[Xarray[:, :, 0] ** 2 + Yarray[:, :, 0] ** 2 > (self.motor_diam / 2) ** 2] = False
    for i in range(0, self.fidelityZ):
        fmm_mask[:, :, i] = dummy
    return fmm_mask


def make_mask_copy(self, motor):
    Xarray, Yarray, Zarray = self.make_coordinate_array()
    mask_copy = np.full_like(motor, False, dtype=bool)
    dummy = np.full_like(motor[:, :, 0], False, dtype=bool)
    dummy[Xarray[:, :, 0] ** 2 + Yarray[:, :, 0] ** 2 > ((self.motor_diam - 0.3) / 2) ** 2] = True
    for i in range(0, self.fidelityZ):
        mask_copy[:, :, i] = dummy
    return ~mask_copy


def calculate_burning_volume(mesh, motor_volume, volume_ratio):
    point_cloud = np.where(mesh == 0)
    points = len(point_cloud[0])
    bore_volume = points * volume_ratio ** 3
    propellant_volume = motor_volume - bore_volume
    return propellant_volume, bore_volume


def calculate_port_area(self, masked_motor):
    port_points = np.where(masked_motor[:, :, 0] == 0)
    port_area = len(port_points[0]) * self.motor_diam ** 2 / self.fidelityXY ** 2
    return port_area


def plot_point_mesh(self, motor_mesh):
    point_cloud = np.where(motor_mesh == 0)
    sizeCheck = np.size(point_cloud)
    if sizeCheck < 1000000:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(point_cloud[0], point_cloud[1], point_cloud[2])
        ax.set_xlabel(f"X Axis [{self.motor_diam}in]")
        ax.set_ylabel(f"Y Axis [{self.motor_diam}in]")
        ax.set_zlabel(f"Z Axis [{self.motor_height} in]")

        ax.set_xlim(0, self.fidelityXY)
        ax.set_ylim(0, self.fidelityXY)
        ax.set_zlim(0, self.fidelityZ)

        plt.tight_layout()
        plt.show()
    else:
        print(f"size = {sizeCheck} Too many points to plot.  Lower fidelity. ")


def plot_surface_mesh(self, motor_mesh, mc_mask):
    verts, faces, norm, line = measure.marching_cubes(motor_mesh, spacing=(
    self.x_volume_ratio, self.y_volume_ratio, self.z_volume_ratio), mask=mc_mask)
    fig = plt.figure(figsize=(5, 5))
    mesh_plot = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    mesh_plot.add_collection3d(mesh)
    mesh_plot.set_xlabel("x-axis")
    mesh_plot.set_ylabel("y-axis")
    mesh_plot.set_zlabel("z-axis")
    mesh_plot.set_xlim(0, self.motor_diam)
    mesh_plot.set_ylim(0, self.motor_diam)
    mesh_plot.set_zlim(0, self.motor_height)
    plt.tight_layout()
    plt.show()


def plot_surface_regression(self, fmm_map, mc_mask):
    contours = 5
    levels = np.linspace(np.min(fmm_map), np.max(fmm_map), contours)
    verts = 0
    faces = 0
    for i in levels:
        v, f, n, l = measure.marching_cubes(fmm_map, level=i,
                                            spacing=(self.x_volume_ratio, self.y_volume_ratio, self.z_volume_ratio),
                                            mask=mc_mask)
        if i == 1:
            verts = v
            faces = f
        else:
            np.append(verts, v)
            np.append(faces, f)
    fig = plt.figure(figsize=(5, 5))
    mesh_plot = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    color_dict = ['k', 'r', 'o', 'y', 'g', 'b', 'i', 'v']
    mesh.set_edgecolor(color_dict[1])
    mesh_plot.add_collection3d(mesh)
    mesh_plot.set_xlabel("x-axis")
    mesh_plot.set_ylabel("y-axis")
    mesh_plot.set_zlabel("z-axis")
    mesh_plot.set_xlim(0, self.motor_diam)
    mesh_plot.set_ylim(0, self.motor_diam)
    mesh_plot.set_zlim(0, self.motor_height)
    plt.tight_layout()
    plt.show()
