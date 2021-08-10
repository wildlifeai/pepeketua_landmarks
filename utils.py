import math
import os
import cv2
from PIL import Image
import pandas as pd
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

def show_labels(img, labels, labels_real = None, radius = 5, thickness = 1, radius_real = 10, 
                color = (0, 0, 255), color_real = (0, 255, 0), save_image = False,
                show_img = False):
    for i in range(0, len(labels), 2):
        point = np.round([labels[i], labels[i+1]]).astype(int)
        point = tuple(point)
        img = cv2.circle(img, point, radius, color, thickness)
        if labels_real is not None:
            point = np.round([labels_real[i], labels_real[i + 1]]).astype(int)
            point = tuple(point)
            img = cv2.circle(img, point, radius, color_real, thickness)
            img = cv2.circle(img, point, radius_real, color_real, thickness)

    # Save the image 
    if save_image:
        nn = np.random.randint(0,1000)
        image_path = os.path.join('landmark_images','new_image_{0}.jpg').format(nn)
        cv2.imwrite(image_path, img)
    if show_img:
        show_image(img)
    return img

def show_image(img):
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()