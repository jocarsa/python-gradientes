from PIL import Image
import os
import time
import numpy as np
from scipy.spatial import cKDTree

# Define the width and height of the image
width = 400
height = 400

# Create the render folder if it does not exist in the same directory as the script
render_folder = "render"
os.makedirs(render_folder, exist_ok=True)

# Load the JPG file located in the same folder as the script
input_image_path = os.path.join(render_folder, "entrada.jpg")
input_image = Image.open(input_image_path).resize((width, height))
input_image_array = np.array(input_image)

# Sample one pixel every 200 pixels along both x and y axes
step = 20
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

# Create KD-tree for sampled points
tree = cKDTree(points)

# Create an empty array for the interpolated colors
interpolated_colors = np.zeros((height, width, 3), dtype=np.uint8)

# Function to compute the IDW interpolation
def idw_interpolation(x, y, points, colors, tree, k=4, p=2):
    dists, idxs = tree.query([x, y], k=k)
    weights = 1 / (dists ** p + 1e-10)
    weights /= weights.sum()
    color = np.dot(weights, colors[idxs])
    return color

# Interpolate colors using IDW for each pixel
for y in range(height):
    for x in range(width):
        interpolated_colors[y, x] = idw_interpolation(x, y, points, colors, tree)

# Create the image from the array
image = Image.fromarray(interpolated_colors, 'RGB')

# Get the current epoch time for unique filenames
epoch_time = int(time.time())

# Save the image to the render folder with epoch time and iteration number in the filename
output_image_path = os.path.join(render_folder, f"gradient_points_image_{epoch_time}.jpg")
image.save(output_image_path, "JPEG")
image.show()
print("Images saved successfully.")
