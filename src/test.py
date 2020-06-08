import numpy as np 
import colour
from colour import SpectralDistribution
from PIL import Image

cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1964 10 Degree Standard Observer']
illuminant = colour.ILLUMINANTS_SDS['D65']

w = 32
h = 32
n = 31
# wavelengths = [400 + (10 * j) for j in range(n)]
# data_rgb = np.empty((3,w*h))

image = Image.open(f"/workspaces/ps_project/data/balloons_ms/balloons_ms/balloons_RGB.bmp")
image = image.crop((0,0,w,h))
data = np.asarray(image)
# print(data[:,:,0].reshape((1,32*32)))
print(data[0,0])
image.save('/workspaces/ps_project/src/original.png')

