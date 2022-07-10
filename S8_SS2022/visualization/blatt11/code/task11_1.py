import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

MAX_SIZE = 2500

# read data
life_expectancy = pd.read_csv("life_expectancy_filtered.csv")
population = pd.read_csv("population_filtered.csv")
gdp = pd.read_csv("gdp_filtered.csv")
child_mortality = pd.read_csv("child_mortality_filtered.csv")


def adapt_shape(df: pd.DataFrame):
    """Clip name, turn into numpy array, set time to the primary key."""
    return df.iloc[:, 1:].to_numpy().T


def circle(arr):
    """Append the first entry to the end of the array."""
    return np.concatenate((arr, [arr[0]]))


years = gdp.keys()[1:]
years = circle(years)

# position
# [time][country]
life_expectancy_np = adapt_shape(life_expectancy)
gdp_np = adapt_shape(gdp)
# show in 1000 $ (i think numbers without that many 0s are more readable)
gdp_np /= 1000
# [time][country][x|y]
p = np.dstack((gdp_np, life_expectancy_np))
# circle around
p = circle(p)

# size
s = circle(adapt_shape(population))
s *= MAX_SIZE / np.max(s)

# color
c = circle(adapt_shape(child_mortality))

# we only need one subplot
fig, ax = plt.subplots(figsize=(13, 8))

# labels and stuff
ax.set_ylabel("Life expectancy (years)")
ax.set_xlabel("GDP per capita (1000 inflation adjusted $)")

# fix axis limits to prevent jumpy animations
ax.set(xlim=(0, np.max(p[:, :, 0])), ylim=(0, 100))

# create an initial scatterplot of first time point
scatterplot = ax.scatter(p[0, :, 0], p[0, :, 1],
                         s=s[0], c=c[0], vmin=0.0, vmax=200.0)
cbar = fig.colorbar(scatterplot)
cbar.set_label("Child mortality per 1000 born")
ax.grid(color="lightgray")

# animation parameters
time_res = 5  # interpolation steps between keyframes
time_speed = 0.1  # seconds between each keyframe
time_steps = time_res * (len(p) - 1)  # number of global timesteps


def animate(i):
    # linear interpolation parameters
    t = i / time_res  # current point in time (e.g. 2.15)
    t_low = int(t)    # lower discrete time (e.g. 2)
    f = t - t_low     # interpolation factor (e.g. 0.15)

    # set the new positions
    p_interp = (1 - f) * p[t_low] + f * p[t_low + 1]
    scatterplot.set_offsets(p_interp)

    # set the new sizes
    s_interp = (1 - f) * s[t_low] + f * s[t_low + 1]
    scatterplot.set_sizes(s_interp)

    # set the new colors
    c_interp = (1 - f) * c[t_low] + f * c[t_low + 1]
    scatterplot.set_array(c_interp)

    # title with year
    ax.set_title(years[math.ceil(t)])


# show the animation with a call to FuncAnimation
# this needs to be stored in a variable (here: 'anim') to prevent garbage collection
anim = FuncAnimation(fig, animate, interval=(
    1000 * time_speed) / time_res, frames=time_steps)
plt.show()
