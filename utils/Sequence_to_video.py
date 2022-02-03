import cv2
import glob
import re

img_array = []
numbers = re.compile(r'(\d+)')
# ROOT = 'C:/Users/USER/Documents/Programming/ImageRegression/out/base/sample_*.png'
# path = 'C:/Users/USER/Documents/Programming/ImageRegression/out/base/error_*.png'
# path = 'C:/Users/USER/Documents/Programming/ImageRegression/out/base/sample_*.png'
# path = 'C:/Users/USER/Documents/Programming/ImageRegression/out/grownet/pred/_ensemble_*.png'
path = 'C:/Users/USER/Documents/Programming/ImageRegression/out/grownet/error/_grad_*.png'


def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


size = None
for filename in sorted(glob.glob(path), key=numerical_sort):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter("out/error.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 15, size)
# out = cv2.VideoWriter("out/sample.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
