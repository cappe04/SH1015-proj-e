from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

data: np.ndarray[np.float32]
spectral_axis: np.ndarray[np.float32]


def load_data(path):
    global data
    global spectral_axis

    with fits.open(path) as file:
        hdu = file[1]
        wcs = WCS(hdu.header).spectral

        data = np.array(hdu.data, dtype=np.float32)

        spectral_axis = wcs.pixel_to_world_values(np.arange(data.shape[0])) * 1e6 # micrometer
        # spectral_axis = np.nan_to_num(spectral_axis)


class Background:
    def __init__(self, spec):
        """Linear fit of all wavelengths. Pixel specific."""
        idx = np.isfinite(spectral_axis) & np.isfinite(spec)

        self.k, self.m = np.polyfit(spectral_axis[idx], spec[idx], 1)
        self.data = self.k * spectral_axis + self.m

    def eval(self, x):
        """Returns the background level at a sepcific wavelength x."""
        return self.k * x + self.m

def get_image(layer: int):
    """One 2D image corresponding to a specific layer. One wavelength."""
    return data[layer-1, :, :]

def get_spectrum(x, y):
    """Gets spectrum from pixel (x, y). Same for all layers!
    Returns spectrum and background"""
    spec = data[:, y-1, x-1]
    # spec = np.nan_to_num(spec)

    if np.isfinite(spec).sum() == 0:
        return spec, None

    bg = Background(spec)
    return spec, bg

def load_ascii_spectrum(path):
    """"""
    with open(path, "r") as file:
        spec = np.array(file.read().split(), dtype=np.float32)
        spec = np.reshape(spec, (len(spec)//2, 2))

    return spec[:, 0], spec[:, 1]

def spectrum_index(value):
    """Returns the layer closest to a desired wavelength?"""
    return np.abs(spectral_axis - value).argmin()

def spectrum_slice(from_, to):
    """Slices a spectrum. Takes wavelength interval, returns layer interval."""
    return slice(spectrum_index(from_), spectrum_index(to)+1)

def fit_normal(spec, background: Background, spec_slice):
    """Fits a normal curve to a spectrum. The spectrum should be sliced to only contain one peak."""
    X = spectral_axis[spec_slice]
    Y = spec[spec_slice]

    correct_Y = np.clip(Y - background.eval(X), 0, None)

    mu = np.average(X, weights=correct_Y)
    std = np.sqrt(np.average((X - mu)**2, weights=correct_Y))
    a = correct_Y.max()

    def fn(x):
        return a * np.exp(-(x - mu)**2 / (2 * std**2)) + background.eval(x)

    return mu, std, a, fn

def speed_from_doppler(wl_ref, wl_measured, mps = True):
    """Returns the speed of a pixel based on the shift of wavelength. Stadard v [m/s], option v/c.
    Rounds to lowest sig fig"""
    c = 299792458
    sigfig = min(len(str(wl_ref))-1, len(str(wl_measured))-1)
    v = c*(wl_measured**2 - wl_ref**2) / (wl_ref**2 + wl_measured**2)
    integers = len(str(round(v)))
    if mps == True: 
        print(mps)
        return round(v, sigfig - integers) 
    else: 
        n = 0
        while int(v/c) == 0:
            print(v/c)
            v = v * 10
            n += 1
        return round(v/c / 10**n , n + sigfig)