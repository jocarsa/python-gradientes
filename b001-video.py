import numpy as np
from PIL import Image
import random
import os
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the width and height of the image
width = 2048
height = 1024

num_points = 3
num_frames = 100
fps = 30

# Create the render folder if it does not exist
render_folder = "render"
os.makedirs(render_folder, exist_ok=True)

# Generate random points, their colors, directions, and speeds
points = np.array([(random.randint(0, width - 1), random.randint(0, height - 1)) for _ in range(num_points)], dtype=float)
colors = np.array([(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_points)])
directions = np.array([(random.choice([-1, 1]), random.choice([-1, 1])) for _ in range(num_points)])
speeds = np.array([random.uniform(1, 5) for _ in range(num_points)])

def update_points(points, directions, speeds):
    for i in range(num_points):
        # Update the position
        points[i] += directions[i] * speeds[i]
        
        # Bounce off the walls
        if points[i, 0] < 0 or points[i, 0] >= width:
            directions[i, 0] *= -1
        if points[i, 1] < 0 or points[i, 1] >= height:
            directions[i, 1] *= -1

    return points

def generate_frame(frame_num):
    # Update points positions
    update_points(points, directions, speeds)
    
    # Create mesh grid for all pixel coordinates
    xv, yv = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')

    # Calculate distances from each pixel to each point
    distances = np.sqrt((xv[..., None] - points[:, 0]) ** 2 + (yv[..., None] - points[:, 1]) ** 2)

    # Calculate weights using a sharper inverse distance function
    weights = 1 / (distances ** 3 + 1e-10)  # Cubic inverse distance for sharper transitions

    # Normalize weights
    weights /= weights.sum(axis=2)[..., None]

    # Calculate interpolated colors
    interpolated_colors = np.dot(weights, colors)

    # Convert to integers
    interpolated_colors = interpolated_colors.astype(np.uint8)

    # Create the image from the array
    image = Image.fromarray(interpolated_colors, 'RGB')
    return np.array(image)

# Set up the figure and axis for the animation
fig, ax = plt.subplots()
ax.set_axis_off()

# Initialize the image for the animation
im = ax.imshow(generate_frame(0), animated=True)

def update_frame(frame_num):
    im.set_array(generate_frame(frame_num))
    return [im]

# Create the animation
ani = FuncAnimation(fig, update_frame, frames=num_frames, blit=True)

# Get the current epoch time for unique filenames
epoch_time = int(time.time())

# Save the animation as an MP4 file
mp4_path = os.path.join(render_folder, f"gradient_points_animation_{epoch_time}.mp4")
ani.save(mp4_path, fps=fps, extra_args=['-vcodec', 'libx264'])

print(f"Animation saved successfully as {mp4_path}.")
