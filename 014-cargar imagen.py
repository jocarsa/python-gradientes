from PIL import Image
import random
import os
import time
import numpy as np

# Define the width and height of the image
width = 2048
height = 1024

# Create the render folder if it does not exist in the same directory as the script
render_folder = "render"
os.makedirs(render_folder, exist_ok=True)

# Load the JPG file located in the same folder as the script
input_image_path = os.path.join(render_folder, "entrada.jpg")
input_image = Image.open(input_image_path).resize((width, height))
input_image_array = np.array(input_image)

# Sample one pixel every 200 pixels along both x and y axes
step = 100
sampled_points = []
sampled_colors = []

for y in range(0, height, step):
    for x in range(0, width, step):
        color = input_image_array[y, x]
        sampled_points.append((x, y))
        sampled_colors.append(tuple(color))

# Convert sampled points and colors to numpy arrays
points = np.array(sampled_points)
colors = np.array(sampled_colors)

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

# Get the current epoch time for unique filenames
epoch_time = int(time.time())

# Save the image to the render folder with epoch time and iteration number in the filename
output_image_path = os.path.join(render_folder, f"gradient_points_image_{epoch_time}.jpg")
image.save(output_image_path, "JPEG")
image.show()
print("Images saved successfully.")
