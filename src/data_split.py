from PIL import Image
import image_slicer
import random
import os

path = '/workspaces/ps_project/data/raw/'
train_path = '/workspaces/ps_project/data/train/'
validation_path = '/workspaces/ps_project/data/validation/'
test_path = '/workspaces/ps_project/data/test/'

def imgcrop(input_path, input_image, patch_size, destination):
    # filename, file_extension = os.path.splitext(input)
    # print(input_image)
    filename = input_image[:-4]
    file_extension = input_image[-3:]
    input_image = input_path + input_image
    # print(input_image)
    # print(filename)
    # print(file_extension)
    im = Image.open(input_image)
    imgwidth, imgheight = im.size
    height = width = patch_size
    pieces = imgheight//patch_size
    for i in range(0, pieces):
        for j in range(0, pieces):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            a = im.crop(box)
            try:
                # pass
                print(destination + filename + "_" + str(i) + "_" + str(j) + '.' + file_extension)
                a.save(destination + filename + "_" + str(i) + "_" + str(j) + '.' + file_extension)
            except:
                pass

image_range = list(range(0,33))
random.shuffle(image_range)


# generate training set
for i in image_range[:-3]:  
    image_path = path + str(i) + '/'
    for r,d,f in os.walk(image_path):
        for file in f:
            if ('.png' in file) | ('.bmp' in file):
                imgcrop(image_path, file, patch_size=32, destination=train_path)
            else:
                pass

# generate validation set
for i in image_range[-3:-1]:
    image_path = path + str(i) + '/'
    for r,d,f in os.walk(image_path):
        for file in f:
            if ('.png' in file) | ('.bmp' in file):
                imgcrop(image_path, file, patch_size=32, destination=validation_path)
            else:
                pass

# generate a test set
i = image_range[-1]
image_path = path + str(i) + '/'
for r,d,f in os.walk(image_path):
    for file in f:
        if ('.png' in file) | ('.bmp' in file):
            imgcrop(image_path, file, patch_size=32, destination=test_path)
        else:
            pass
