import numpy as np 
from PIL import Image
import pandas as pd 
import colour

# Load illimunation and cie64 spectrum
cie64 = pd.read_csv('/workspaces/ps_project/data/ciexyz.csv', header = None, names =['wv', 'x', 'y', 'z'] )
d65 = pd.read_csv('/workspaces/ps_project/data/d65.csv', header = None, names = ['wv', 'ill'])

# Keep only required wavelengths
cie64 = cie64[((cie64['wv'] % 10) == 0) & (cie64['wv'] >= 400) & (cie64['wv'] <= 700)]
d65 = d65[((d65['wv'] % 10) == 0) & (d65['wv'] >= 400) & (d65['wv'] <= 700)]

# Drop 'wavelength' column
cie64 = cie64.drop(['wv'], axis = 1)
d65 = d65.drop(['wv'], axis = 1)

# Dimension of patch, and n = number of spectra
w = 32
h = 32
n = 31

# Initialise empty arrays for hyperspectral and RGB data
data_hyp = np.empty((n, w*h))
data_rgb = np.empty((3,w*h))

# Open all 'n' images and store the patch as a single (n, w*h) array
# wavelengths = [400 + (10 * j) for j in range(n)]
for k in range(n):
    image = Image.open(f"/workspaces/ps_project/data/balloons_ms/balloons_ms/balloons_ms_{k + 1:02}.png")
    image = image.crop((0,0,w,h))
    pixels = np.asarray(image)
    pixels = np.reshape(pixels, (w*h,))
    pixels = pixels / np.amax(pixels)
    data_hyp[k] = pixels

# Calculate normalising factor
N = np.dot(np.transpose(cie64['y'].to_numpy()), d65['ill'].to_numpy())

# Calculating XYZ co-ordinates
data_hyp = d65['ill'].to_numpy().reshape((31,1)) * data_hyp
XYZ = np.dot(np.transpose(cie64.to_numpy()), data_hyp)
XYZ /= N            # Normalise by N

# Converting XYZ to sRGB
RGB = colour.XYZ_to_sRGB(np.transpose(XYZ))

# reshaping array for display
RGB = np.reshape(np.transpose(RGB), (w, h, 3))

# Convert [0, 1] to [0, 255]
RGB *= 255

# Save image
image = Image.fromarray(RGB, mode = 'RGB')
image.save('/workspaces/ps_project/src/hyptoRGB.png')


