import pydicom as pdcm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
img_path=r"D:\Project\LIDC-IDRI-Preprocessing\LIDC-IDRI\LIDC-IDRI-0001\01-01-2000-NA-NA-30178\3000566.000000-NA-03192\1-040.dcm"
# plt.imshow(img)
# plt.show()

file = pdcm.dcmread(img_path)
img = file.pixel_array
# img_pil = Image.fromarray(img.astype(np.uint8))/

# cv2.imshow('show',img)
plt.imshow(img)
plt.show()

