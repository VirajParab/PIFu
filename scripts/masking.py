import cv2
import numpy as np
import matplotlib.pyplot as plt

# load the image
image = cv2.imread('./sample_images/ryota.png')
image_copy = image.copy()


black_pixels_mask = np.all(image == [0, 0, 0], axis=-1)

non_black_pixels_mask = np.any(image != [0, 0, 0], axis=-1)
# or non_black_pixels_mask = ~black_pixels_mask

image_copy[black_pixels_mask] = [0, 0, 0]
image_copy[non_black_pixels_mask] = [255, 255, 255]


height = image.shape[0]
width = image.shape[1]
stretch_mask_img = cv2.resize(image_copy, (512, 512),
               interpolation = cv2.INTER_NEAREST)

stretch_img = cv2.resize(image, (512, 512),
               interpolation = cv2.INTER_NEAREST)
#
# stretch_near = cv2.resize(stretch_img, ( 512, 512),
#                interpolation = cv2.INTER_NEAREST)


# gray = cv2.cvtColor(stretch_near, cv2.COLOR_BGR2GRAY)
# img2 = np.zeros_like(stretch_near)
# img2[:,:,0] = gray
# img2[:,:,1] = gray
# img2[:,:,2] = gray
# plt.imshow(img2)
# plt.show()


# image1 = cv2.imread('./sample_images_old/ryota_mask.png')
# height = stretch_near.shape[0]
# width = stretch_near.shape[1]
# channels = stretch_near.shape[2]
#
print("height : ", height)
print("width : ", width)
# print("channels : ", channels)

cv2.imwrite('./sample_images/ryota_mask.png', stretch_mask_img)
cv2.imwrite('./sample_images/ryota.png', stretch_img)
