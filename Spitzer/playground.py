from astropy.io import fits
import os
import glob

BASE_PATH = r"C:\Users\robob\Documents\astro research; stellar pulsations\Spitzer datasets"
AOR = 43962624

# Find one BCD file
bcd_dir = os.path.join(BASE_PATH, str(AOR), f"r{AOR}", "ch2", "bcd")
files = sorted(glob.glob(os.path.join(bcd_dir, "*_bcd.fits")))
print("N files:", len(files))
fname = files[0]
print("Using file:", fname)

with fits.open(fname) as hdul:
    h = hdul[0].header
    data = hdul[0].data

print("DATA SHAPE:", data.shape)
for key in ["MBJD_OBS", "BMJD_OBS", "BJD_OBS", "MJD_OBS", "DATE_OBS",
            "AINTBEG", "ATIMEEND", "EXPTIME", "FRAMTIME"]:
    print(f"{key} =", h.get(key, None))

# print a few pixel stats so we know values are reasonable
import numpy as np
print("FRAME 0 min/max/median:", np.nanmin(data[0]), np.nanmax(data[0]), np.nanmedian(data[0]))
