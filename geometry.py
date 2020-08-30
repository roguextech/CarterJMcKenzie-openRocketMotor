import numpy as np

def circle_area(diam):
    area = np.pi * diam ** 2 / 4
    return area


def circle_perimeter(d):
    area = np.pi * d
    return area


def cylinder_volume(d_out, d_in, h):
    return d_out ** 2 * np.pi / 4 * h - d_in ** 2 * np.pi / 4 * h


def burning_area(d, h):
    return circle_perimeter(d) * h


def subtract_sphere(d):
    return 4 / 3 * d ** 3 - d ** 2 * np.pi / 4 * d


def cone_area(d, h):
    """surface area of a cone without bottom circle"""
    r = d / 2
    surface_area = np.pi * r * (r + (h ** 2 + r ** 2) ** (1 / 2))  # - np.pi * r**2
    return surface_area


def cone_volume(d, h):
    """volume of a cone without bottom circle"""
    r = d / 2
    volume = np.pi * r ** 2 * h / 3
    return volume


def length(contour, fidelity, tolerance=3):
    """Returns the total length of all segments in a contour that aren't within 'tolerance' of the edge of a
    circle with diameter 'mapSize'"""
    offset = np.roll(contour.T, 1, axis=1)
    lengths = np.linalg.norm(contour.T - offset, axis=0)
    centerOffset = np.array([[fidelity / 2, fidelity / 2]])
    radius = np.linalg.norm(contour - centerOffset, axis=1)
    valid = radius < (fidelity / 2) - tolerance
    return np.sum(lengths[valid])


def mass(volume, density):
    return volume * density

