from PIL import Image
import matplotlib.pyplot as plt

# Specify the path to the image file
image_path = r'D:/_phd/code/textures/textureSSD/findBox/z1Sample.png'

# Load the image
image = Image.open(image_path)

# Show the image using matplotlib
plt.figure()
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Specify the coordinates of the inside box
left = 100
top = 100
right = 300
bottom = 300

# Crop the image to get the inside box
inside_box = image.crop((left, top, right, bottom))

# Show the inside box image using matplotlib
plt.figure()
plt.imshow(inside_box)
plt.title('Inside Box Image')
plt.axis('off')
plt.show()

# Find the biggest crop coordinates
max_crop_size = max(image.size[0] - left, right - left, image.size[1] - top, bottom - top)

# Calculate the center coordinates of the biggest crop
center_x = (left + right) // 2
center_y = (top + bottom) // 2

# Calculate the new crop coordinates
new_left = center_x - max_crop_size // 2
new_top = center_y - max_crop_size // 2
new_right = center_x + max_crop_size // 2
new_bottom = center_y + max_crop_size // 2

# Crop the image to get the biggest crop
biggest_crop = image.crop((new_left, new_top, new_right, new_bottom))

# Show the biggest crop image using matplotlib
plt.figure()
plt.imshow(biggest_crop)
plt.title('Biggest Crop Image')
plt.axis('off')
plt.show()