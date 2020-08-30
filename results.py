import matplotlib.pyplot as plt


def plot_chamber_pressure(grain_list):
    fig, axs = plt.subplots(1, 1)
    count = 0
    for grain in grain_list:
        count = count + 1
        axs.plot(grain.elapsed_time, grain.chamber_pressure, label=f"Motor: {count}")
        axs.set_ylabel("Chamber Pressure [Pa]")
        axs.set_xlabel("Time [s]")
    plt.legend()
    fig.savefig("Chamber_Pressure.png")
    plt.show()


def plot_thrust(grain_list):
    fig, axs = plt.subplots(1, 1)
    count = 0
    for grain in grain_list:
        count = count + 1
        axs.plot(grain.elapsed_time, grain.thrust, label=f"Motor: {count}")
        axs.set_ylabel("Thrust [N]")
        axs.set_xlabel("Time [s]")
    plt.legend()
    fig.savefig("Thrust.png")
    plt.show()


def plot_burn_rate(grain_list):
    fig, axs = plt.subplots(1, 1)
    count = 0
    for grain in grain_list:
        count = count + 1
        axs.plot(grain.elapsed_time, grain.burn_rate, label=f"Motor: {count}")
        axs.set_ylabel("Burn Rate [m/s]")
        axs.set_xlabel("Time [s]")
    plt.legend()
    fig.savefig("Burn_Rate.png")
    plt.show()


def plot_exit_pressure(grain_list):
    fig, axs = plt.subplots(1, 1)
    count = 0
    for grain in grain_list:
        count = count + 1
        axs.plot(grain.elapsed_time, grain.exit_pressure, label=f"Motor: {count}")
        axs.set_ylabel("Exit Pressure [Pa]")
        axs.set_xlabel("Time [s]")
    plt.legend()
    fig.savefig("Exit_Pressure.png")
    plt.show()


def plot_mass_flow_rate(grain_list):
    fig, axs = plt.subplots(1, 1)
    count = 0
    for grain in grain_list:
        count = count + 1
        axs.plot(grain.elapsed_time, grain.m_dot, label=f"Motor: {count}")
        axs.set_ylabel("Mass Flow Rate [kg/s]")
        axs.set_xlabel("Time [s]")
    plt.legend()
    fig.savefig("Mass_Flow.png")
    plt.show()


def plot_burning_area(grain_list):
    fig, axs = plt.subplots(1, 1)
    count = 0
    for grain in grain_list:
        count = count + 1
        axs.plot(grain.elapsed_time, grain.burning_area, label=f"Motor: {count}")
        axs.set_ylabel("Burning Area [m^2]")
        axs.set_xlabel("Time [s]")
    plt.legend()
    fig.savefig("Burning_Area.png")
    plt.show()


def plot_regression_depth(grain_list):
    fig, axs = plt.subplots(1, 1)
    count = 0
    for grain in grain_list:
        count = count + 1
        axs.plot(grain.elapsed_time, grain.regression_depth, label=f"Motor: {count}")
        axs.set_ylabel("Regression Depth [m]")
        axs.set_xlabel("Time [s]")
    plt.legend()
    fig.savefig("Regression_Depth.png")
    plt.show()


def display_iteration_values(grain):
    for i in range(0, len(grain.elapsed_time)):
        print(f"\nIteration: {i}")
        print(f"Time: {round(grain.elapsed_time[i], 3)} sec")
        print(f"Regression: {round(grain.regression_depth[i], 3)} m")
        print(f"Burning Area: {round(grain.burning_area[i], 3)} m^2")
        print(f"Chamber Pressure: {round(grain.chamber_pressure[i], 3)} Pa")
        print(f"Burn Rate: {round(grain.burn_rate[i], 3)} m/sec")
        print(f"Mass Flow Rate: {round(grain.m_dot[i], 3) } kg/sec")
        print(f"Exit Pressure: {round(grain.exit_pressure[i], 3)} Pa")
        print(f"Thrust: {round(grain.thrust[i], 3)} N")
