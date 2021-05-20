import numpy as np

# Transform Cartesian to polar coordinates
def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return rho, theta

# Converts rad to theta in range 0 - 360
def positive_deg_theta(theta):
    theta = np.rad2deg(theta)
    return np.mod(theta + 360, 360)

# Transform polar to Cartesian coordinates
def pol2cart(rho, theta):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y