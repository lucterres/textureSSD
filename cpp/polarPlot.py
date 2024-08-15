import matplotlib.pyplot as plt
import numpy as np
import math as m

# Create a figure and axis
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
e = m.e
pi = np.pi



# Initialize the plot
theta = np.linspace(0, 2*pi, 1000)
r = np.sin(theta)
line, = ax.plot(theta, r)

# Update the plot
for i in range(100):
    r =  e**(theta*i)+e**(pi*theta*i)   #np.sin(i * theta)
    line.set_ydata(r)
    plt.draw()
    plt.pause(0.1)

# Show the plot
plt.show()