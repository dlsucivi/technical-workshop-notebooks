import os
from PIL import Image

img_path = "/home/jupyter-aaron_gabrielle_di-865b9/ACOD-12K/Train/Imgs/peach_left_00350.jpg"

if os.path.exists(img_path):
    print("✅ File exists!")
else:
    print("❌ File does NOT exist!")

if os.path.isfile(img_path):
    print("✅ File is a regular file!")
else:
    print("❌ File is NOT a regular file!")

try:
    img = Image.open(img_path)
    img.show()
    print("✅ Image loaded successfully!")
except Exception as e:
    print("❌ Error opening image:", e)

with open(img_path, "rb") as f:
    print("✅ File read successfully!")

