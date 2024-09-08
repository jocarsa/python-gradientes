from PIL import Image, ImageDraw
import random
import os
import time
import numpy as np
from multiprocessing import Pool, cpu_count

# Function to interpolate color using IDW for a specific chunk of pixels
def interpolate_chunk(args):
    chunk_start, chunk_end, points, colors, num_points = args
    chunk_result = []
    
    for idx in range(chunk_start, chunk_end):
        i, j = divmod(idx, 1080)
        distances = [((i - x) ** 2 + (j - y) ** 2) ** 0.5 for x, y in points]
        weights = [1 / (d ** 2 + 1e-10) for d in distances]  # Use squared inverse distance for smoother gradient
        total_weight = sum(weights)
        
        interpolated_color = [0, 0, 0]
        for k in range(3):
            interpolated_color[k] = int(sum(colors[m][k] * weights[m] for m in range(num_points)) / total_weight)
        
        chunk_result.append((i, j, tuple(interpolated_color)))
    
    return chunk_result

# Create a new 1080x1080 image with a white background
image = Image.new("RGB", (1080, 1080), "white")
draw = ImageDraw.Draw(image)

# Define the number of points
num_points = 10

# Generate random points and their colors
points = [(random.randint(0, 1079), random.randint(0, 1079)) for _ in range(num_points)]
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_points)]

# Prepare a list of chunks for parallel processing
num_chunks = cpu_count()
chunk_size = (1080 * 1080) // num_chunks
tasks = [(i * chunk_size, (i + 1) * chunk_size, points, colors, num_points) for i in range(num_chunks)]

# Process the tasks in parallel using multiprocessing
with Pool(processes=cpu_count()) as pool:
    results = pool.map(interpolate_chunk, tasks)

# Apply the color array to the image
for chunk in results:
    for result in chunk:
        i, j, color = result
        draw.point((i, j), fill=color)

# Create the render folder if it does not exist in the same directory as the script
render_folder = "render"
os.makedirs(render_folder, exist_ok=True)

# Get the current epoch time
epoch_time = int(time.time())

# Save the image to the render folder with epoch time in the filename
image_path = os.path.join(render_folder, f"gradient_points_image_{epoch_time}.png")
image.save(image_path)

