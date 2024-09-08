from PIL import Image, ImageDraw
import random
import os
import time
import numpy as np

# Create a new 1080x1080 image with a white background
image_size = 1080
image = Image.new("RGB", (image_size, image_size), "white")
draw = ImageDraw.Draw(image)

# Define the number of points
num_points = 50

# Generate random points and their colors
points = np.array([(random.randint(0, image_size - 1), random.randint(0, image_size - 1)) for _ in range(num_points)])
colors = np.array([(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_points)])

# Create mesh grid for all pixel coordinates
xv, yv = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing='ij')

# Calculate distances from each pixel to each point
distances = np.sqrt((xv[..., None] - points[:, 0]) ** 2 + (yv[..., None] - points[:, 1]) ** 2)

# Calculate weights using squared inverse distance
weights = 1 / (distances ** 2 + 1e-10)

# Normalize weights
weights /= weights.sum(axis=2)[..., None]

# Calculate interpolated colors
interpolated_colors = np.dot(weights, colors)

# Convert to integers
interpolated_colors = interpolated_colors.astype(np.uint8)

# Create the image from the array
image = Image.fromarray(interpolated_colors, 'RGB')

# Create the render folder if it does not exist in the same directory as the script
render_folder = "render"
os.makedirs(render_folder, exist_ok=True)

# Get the current epoch time
epoch_time = int(time.time())

# Save the image to the render folder with epoch time in the filename
image_path = os.path.join(render_folder, f"gradient_points_image_{epoch_time}.png")
image.save(image_path)

image_path
