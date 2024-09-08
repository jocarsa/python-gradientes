from PIL import Image, ImageDraw
import random
import os
import time

# Create a new 1080x1080 image with a white background
image = Image.new("RGB", (1080, 1080), "white")
draw = ImageDraw.Draw(image)

# Define the number of points
num_points = 100

# Draw random colored points on the image
for _ in range(num_points):
    x = random.randint(0, 1079)
    y = random.randint(0, 1079)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    draw.point((x, y), fill=color)

# Create the render folder if it does not exist
render_folder = "/mnt/data/render"
os.makedirs(render_folder, exist_ok=True)

# Get the current epoch time
epoch_time = int(time.time())

# Save the image to the render folder with epoch time in the filename
image_path = os.path.join(render_folder, f"colored_points_image_{epoch_time}.png")
image.save(image_path)

image.show()
