import numpy as np
import imageio
import glob

image_filter = 'Bilder/*.jpg'
images = [imageio.imread(image) for image in glob.glob(image_filter)]


def ex1_min_max(image):
    # initiate min and max values and coords
    print(image.shape)
    image_min = image[0]
    image_min_x = 0
    image_min_y = 0
    image_max = image[0]
    image_max_x = 0
    image_max_y = 0
    # iterate through all the pixels in the image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # adjust min values and coords
            pixel = image.item(x, y)
            if pixel < image_min:
                image_min = pixel
                image_min_x = x
                image_min_y = y
            # adjust max values and coords
            if pixel > image_max:
                image_max = pixel
                image_max_x = x
                image_max_y = y

    return image_min, image_min_x, image_min_y, image_max, image_max_x, image_max_y

image = np.random.choice(images)
image_min, image_min_x, image_min_y, image_max, image_max_x, image_max_y = ex1_min_max(image)
print(f"Minimum at\tx={image_min_x},\t{image_min_y}:\t{image_min}\nMaximum at\tx={image_max_x},\t{image_max_y}:\t{image_max}")