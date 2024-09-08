from PIL import Image, ImageDraw
import random
import os
import time
import numpy as np

# Create a new 1080x1080 image with a white background
image = Image.new("RGB", (1080, 1080), "white")
draw = ImageDraw.Draw(image)

# Define the number of points
num_points = 10

# Generate random points and their colors
points = [(random.randint(0, 1079), random.randint(0, 1079)) for _ in range(num_points)]
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_points)]

# Create an array to hold the colors for each pixel
color_array = np.zeros((1080, 1080, 3), dtype=int)

# Fill the entire space with gradients between points using IDW
for i in range(1080):
    for j in range(1080):
        distances = [((i - x) ** 2 + (j - y) ** 2) ** 0.5 for x, y in points]
        weights = [1 / (d ** 2 + 1e-10) for d in distances]  # Use squared inverse distance for smoother gradient
        total_weight = sum(weights)
        
        interpolated_color = [0, 0, 0]
        for k in range(3):
            interpolated_color[k] = int(sum(colors[m][k] * weights[m] for m in range(num_points)) / total_weight)
        
        color_array[i, j] = interpolated_color

# Apply the color array to the image
for i in range(1080):
    for j in range(1080):
        draw.point((i, j), fill=tuple(color_array[i, j]))

# Create the render folder if it does not exist in the same directory as the script
render_folder = "render"
os.makedirs(render_folder, exist_ok=True)

# Get the current epoch time
epoch_time = int(time.time())

# Save the image to the render folder with epoch time in the filename
image_path = os.path.join(render_folder, f"gradient_points_image_{epoch_time}.png")
image.save(image_path)

image_path
