from PIL import Image, ImageDraw
import random
import os
import time

# Function to interpolate between two colors
def interpolate_color(color1, color2, factor):
    return tuple(int(color1[i] + (color2[i] - color1[i]) * factor) for i in range(3))

# Create a new 1080x1080 image with a white background
image = Image.new("RGB", (1080, 1080), "white")
draw = ImageDraw.Draw(image)

# Define the number of points
num_points = 10

# Generate random points and their colors
points = [(random.randint(0, 1079), random.randint(0, 1079)) for _ in range(num_points)]
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_points)]

# Draw gradients between points
for i in range(num_points - 1):
    x1, y1 = points[i]
    x2, y2 = points[i + 1]
    color1 = colors[i]
    color2 = colors[i + 1]
    
    for j in range(100):
        factor = j / 100.0
        x = int(x1 + (x2 - x1) * factor)
        y = int(y1 + (y2 - y1) * factor)
        color = interpolate_color(color1, color2, factor)
        draw.point((x, y), fill=color)

# Create the render folder if it does not exist in the same directory as the script
render_folder = "render"
os.makedirs(render_folder, exist_ok=True)

# Get the current epoch time
epoch_time = int(time.time())

# Save the image to the render folder with epoch time in the filename
image_path = os.path.join(render_folder, f"gradient_points_image_{epoch_time}.png")
image.save(image_path)
