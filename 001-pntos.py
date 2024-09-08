from PIL import Image, ImageDraw
import random

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

# Save the image to a file
image_path = "/mnt/data/colored_points_image.png"
image.save(image_path)
image.show()
