import os
import random

from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

from utils.augment import distort


# Transform = transforms.ColorJitter(brightness=0.25, contrast=0.25)
# test_image = '7.png'

# for i in range(10):
#     image = Image.open(test_image)
#     image = Transform(image.copy())
#     cv2.imshow('image', np.array(image))
#     cv2.waitKey(0)


# image = Image.open(test_image)
# dst = distort(image, 8, img_type='PIL_Image')
# dst.show()


# image = Image.open(test_image)
#
#
# def random_erasing(PIL_Image, h, w, value=255):
#     width, height = PIL_Image.size
#     # print(height, width)
#     np_image = np.array(PIL_Image)
#     top = random.randint(0, height - h)
#     left = random.randint(0, width - w)
#     value = np_image[-1, -1]
#     np_image[top: top + h, left: left + w] = value
#
#     dst_Image = Image.fromarray(np_image)
#
#     return dst_Image
#
# dst = random_erasing(image, 20, 10, 0)
# dst.show()

# image = Image.open(test_image)
from PIL import ImageFilter, ImageOps

# blured = image.filter(ImageFilter.BoxBlur(1))
# blured.show()

# reverse = ImageOps.invert(image)
# reverse.show()


from utils.augment import random_erasing, blur, reverse

transformer = transforms.RandomApply(
    [transforms.ColorJitter(brightness=0.25, contrast=0.25),
     transforms.RandomChoice(
         [transforms.Lambda(lambda img: distort(img, 8, img_type='PIL_Image')),
          transforms.Lambda(lambda img: random_erasing(img, 20, 10)),
          transforms.Lambda(lambda img: blur(img, radius=1)),
          # transforms.Lambda(lambda img: reverse(img))
          ]
     )
     ],
    p=1
)

test_dir = 'test'
for image_name in os.listdir(test_dir):
    image_path = os.path.join(test_dir, image_name)
    image = Image.open(image_path)
    dst = transformer(image)
    dst.show()
