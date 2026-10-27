from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

import matplotlib.pyplot as plt

with fits.open("data/fits/jw01232-o003_t001_nirspec_g395m-f290lp_s3d.fits") as file:
    hdu = file[1]
    
    wcs = WCS(hdu.header).spectral

    data = np.array(hdu.data, dtype=np.float32)
    data = np.flip(data, axis=1)

    spectral_axis = wcs.pixel_to_world_values(np.arange(data.shape[0])) * 1e6 # micrometer


class Background:
    def __init__(self, spec):
        self.k, self.m = np.polyfit(spectral_axis, spec, 1)
        self.data = self.k * spectral_axis + self.m

    def eval(self, x):
        return self.k * x + self.m

def get_image(layer: int):
    return data[layer+1, :, :]

def get_spectrum(x, y):
    spec = data[:, x+1, y+1]
    bg = Background(spec)
    return spec, bg

def load_ascii_spectrum(path):
    with open(path, "r") as file:
        spec = np.array(file.read().split(), dtype=np.float32)
        spec = np.reshape(spec, (len(spec)//2, 2))

    return spec[:, 0], spec[:, 1]

def spectrum_index(value):
    return np.abs(spectral_axis - value).argmin()

def spectrum_slice(from_, to):
    return slice(spectrum_index(from_), spectrum_index(to)+1)

def fit_normal(spec, background: Background, spec_slice):
    X = spectral_axis[spec_slice]
    Y = spec[spec_slice]

    correct_Y = np.clip(Y - background.data[spec_slice], 0, None)

    mu = np.average(X, weights=correct_Y)
    std = np.sqrt(np.average((X - mu)**2, weights=correct_Y))
    a = correct_Y.max()

    def fn(x):
        return a * np.exp(-(x - mu)**2 / (2 * std**2)) + background.eval(x)

    return mu, std, a, fn

def eval_normal(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

spec_slice = spectrum_slice(4.04, 4.08)
spec, background = get_spectrum(28, 23)

mu, std, a, fn = fit_normal(spec, background, spec_slice)

X = spectral_axis[spec_slice]
Y = spec[spec_slice]
X_lin = np.linspace(X[0], X[-1], 100)


plt.title(f"Image 1001 - {spectral_axis[1000]:.3f} $\\mu m$")
plt.imshow(get_image(1001), cmap="afmhot")
plt.show()

plt.title("Spectrum of pixel (28, 23)")
plt.xlabel("$\\lambda$ [$\\mu$m]")
plt.ylabel("[MJy/sr]")
plt.plot(X, Y)
plt.plot(X_lin, fn(X_lin))
plt.legend(["Raw data", "Normal fit"])
plt.show()




