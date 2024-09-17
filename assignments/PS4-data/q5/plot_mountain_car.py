import numpy as np
import matplotlib.pyplot as plt

def plot_mountain_car(x):
    plt.clf()
    
    # Plot the mountain
    x_vals = np.arange(-1.2, 0.5, 0.1)
    y_vals = 0.3 * np.sin(3 * x_vals)
    plt.plot(x_vals, y_vals, 'k-')

    # Compute the car's angle and position
    theta = np.arctan2(3 * 0.3 * np.cos(3 * x[0]), 1.0)
    y = 0.3 * np.sin(3 * x[0])

    # Define the car body
    car = np.array([
        [-0.05, 0.05],
        [0.05, 0.05],
        [0.05, 0.01],
        [-0.05, 0.01],
        [-0.05, 0.05]
    ]).T

    # Define the wheels
    angles = np.arange(0, 2 * np.pi + 0.5, 0.5)
    fwheel = np.array([
        0.035 + 0.01 * np.cos(angles),
        0.01 + 0.01 * np.sin(angles)
    ])
    rwheel = np.array([
        -0.035 + 0.01 * np.cos(angles),
        0.01 + 0.01 * np.sin(angles)
    ])

    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Rotate and translate car and wheels
    car = R @ car + np.array([[x[0]], [y]])
    fwheel = R @ fwheel + np.array([[x[0]], [y]])
    rwheel = R @ rwheel + np.array([[x[0]], [y]])

    # Plot car and wheels
    plt.plot(car[0, :], car[1, :], 'b-')
    plt.plot(fwheel[0, :], fwheel[1, :], 'r-')
    plt.plot(rwheel[0, :], rwheel[1, :], 'r-')

    # Set plot limits and aspect ratio
    plt.axis([-1.3, 0.6, -0.4, 0.4])
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()  # Display the plot
