import cv2
import random


def crop(image, crop_size):
    height, width, _ = image.shape
    x_offset = random.randint(0, width - crop_size[0])
    y_offset = random.randint(0, height - crop_size[1])

    return image[y_offset: y_offset+crop_size[1], x_offset: x_offset+crop_size[0]]


def mirror(image):
    _, width, _ = image.shape
    image_m = image[:, ::-1]

    return image_m


def normalize_(image, mean, std):
    image -= mean
    image /= std
