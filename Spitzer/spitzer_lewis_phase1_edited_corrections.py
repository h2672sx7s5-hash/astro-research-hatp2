import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import os
from scipy import stats



# ============================================================
# CONFIGURATION (Lewis+2013 Table & Section 2.1)
# ============================================================
# This section defines all of the fixed choices used later in the pipeline.
# Think of these as the "settings" for the reduction.
# If the reduction output looks weird, these are among the first things to inspect.
#
# In general:
# - BASE_PATH tells Python where the raw Spitzer files live.
# - P_ORB is not used much in Phase 1 itself, but is useful context for the system.
# - The remaining constants define how aggressively we estimate backgrounds,
#   reject bad pixels, compute centroids, measure stellar flux, and reject outliers.
#
# These numbers were chosen to mimic the 4.5 μm procedure described by Lewis+2013:
# background estimated outside 10 px, hot pixels flagged at 4.5σ, centroid aperture
# 3.5 px, photometric aperture 2.25 px, and final outlier rejection using a 16-point
# moving median with a 4.5σ threshold.

# Path to Spitzer data (UPDATE THIS to your actual path)
BASE_PATH = r"C:\Users\robob\Documents\astro research; stellar pulsations\Spitzer datasets"


# Known orbital period (fixed, from Lewis Table 2)
# This is the orbital period of HAT-P-2b in days.
# It is not directly needed to extract raw photometry from images, but it helps keep
# the planetary context visible and becomes important in later modeling.
P_ORB = 5.6334729  # days


# Photometry parameters from Lewis Section 2.1 for 4.5 μm
# These parameters control the image-processing steps:
#
# BACKGROUND_RADIUS:
#   Ignore pixels within 10 px of the image center when estimating background,
#   so the stellar PSF wings do not bias the sky estimate.
#
# BACKGROUND_SIGMA_CLIP:
#   Repeatedly reject unusual background pixels (cosmic rays, bad values, etc.)
#   before estimating the final sky level.
#
# HOT_PIXEL_SIGMA:
#   A pixel is treated as a transient hot pixel if it deviates too far from the
#   typical value for that pixel position across a 64-frame cube.
#
# CENTROID_APERTURE:
#   Radius used for center-of-light centroiding. Large enough to include the PSF core,
#   but not so large that noisy outer pixels dominate.
#
# PHOTOMETRY_APERTURE:
#   Radius used to sum the stellar flux once the centroid is known.
#
# OUTLIER_SIGMA and OUTLIER_WINDOW:
#   Used after photometry is extracted to remove obvious bad time-series points.
BACKGROUND_RADIUS = 10  # pixels; exclude region within this from background calc
BACKGROUND_SIGMA_CLIP = 3.0  # sigma for iterative background clipping
HOT_PIXEL_SIGMA = 4.5  # flag pixels this many sigma from median
CENTROID_APERTURE = 3.5  # pixels; flux-weighted centroid aperture (Lewis: 3.5 for 4.5 μm)
PHOTOMETRY_APERTURE = 2.25  # pixels; stellar flux aperture (Lewis: 2.25 for 4.5 μm) TODO: Photometry vs centroid apertures
OUTLIER_SIGMA = 4.5  # flag outliers in final photometry
OUTLIER_WINDOW = 16  # moving median window for outlier detection


# ============================================================
# 1. LOAD BCD FILES
# ============================================================
# This is the "ingest" stage of the pipeline.
#
# Goal:
#   Walk through all AOR folders, open every Spitzer BCD FITS file, extract:
#   1) the 64 individual 32x32 images in the file
#   2) a timestamp for each image
#
# Important concept:
#   In Spitzer subarray mode, one FITS file does NOT correspond to one image.
#   Instead, each FITS file contains a small time-series cube of 64 images.
#   So the code must expand each cube into 64 timestamps and 64 frames.
#
# Output of this stage:
#   all_times : one timestamp per image
#   all_data  : one 32x32 image per timestamp

def load_spitzer_bcd_files(base_path, channel="ch2"):
    """
    Load all Spitzer BCD files for 4.5 μm (channel 2).
    
    Lewis+2013 used subarray mode BCDs. Each FITS file contains a 
    (64, 32, 32) cube: 64 images of 32×32 pixels.
    
    Parameters
    ----------
    base_path : str
        Path to Spitzer AOR directories
    channel : str
        'ch2' for 4.5 μm
        
    Returns
    -------
    all_times : array
        BJD_UTC mid-exposure times
    all_data : array
        Flux cubes (N_total_images, 32, 32)
    """
    
    print("=" * 60)
    print("LOADING SPITZER 4.5 μm BCD FILES")
    print("=" * 60)
    
    # Find all AOR directories
    # Each AOR is essentially one observing request / observing block.
    # The dataset may contain multiple AOR folders, and each one can contain many FITS cubes.
    aor_dirs = sorted([d for d in os.listdir(base_path) 
                       if os.path.isdir(os.path.join(base_path, d))])
    
    # We store everything in lists first because we do not yet know the total number of images.
    # After all files are read, these lists will be concatenated into large NumPy arrays.
    all_times = []
    all_data = []
    
    for aor in aor_dirs:
        # Build the expected path to the BCD files for this AOR and channel.
        # Spitzer directory structure nests the data fairly deeply.
        aor_path = os.path.join(base_path, aor, f"r{aor}", channel, "bcd")
        
        # Find only the calibrated BCD images.
        # We intentionally do not grab other FITS products like uncertainty maps unless needed.
        bcd_pattern = os.path.join(aor_path, "SPITZER_I2_*_bcd.fits")
        bcd_files = sorted(glob.glob(bcd_pattern))
        
        if len(bcd_files) == 0:
            print(f"  ⚠ Skipping AOR {aor}: no BCD files found")
            continue
            
        print(f"  Processing AOR {aor}: {len(bcd_files)} BCD cubes")
        
        for i, filepath in enumerate(bcd_files):
            # Open one FITS cube.
            # hdul[0].data contains the 64-frame subarray cube.
            # hdul[0].header contains the metadata needed to reconstruct times.
            with fits.open(filepath) as hdul:
                header = hdul[0].header
                data = hdul[0].data  # shape: (64, 32, 32)
                
                # Compute BJD_UTC times (Lewis Section 2.1)
                # BMJD_OBS = modified JD at start of first frame
                # AINTBEG/ATIMEEND = time in seconds since IRAC turn-on
                #
                # Key idea:
                # We know the start time of the cube and the span covered by the 64 frames.
                # Assuming the frames are uniformly spaced, we can assign a time to each frame
                # by linearly spacing times between the start and end of the cube.
                bmjd = header.get("BMJD_OBS", None)
                aintbeg = header.get("AINTBEG", 0.0)
                atimeend = header.get("ATIMEEND", 0.0)


                if bmjd is None:
                    print(f"    WARNING: {os.path.basename(filepath)} missing BMJD_OBS, skipping")
                    continue


                # BMJD_OBS = MJD at START of first frame in this cube
                # AINTBEG = total seconds since IRAC warmup
                # We only want time WITHIN this 64-frame cube (~25s total)
                #
                # Why use modulo here?
                # AINTBEG and ATIMEEND are large counters measured since instrument turn-on.
                # We only care about the timing within the current 64-frame block, not the
                # absolute elapsed time since warmup. Taking modulo (64 * exposure time)
                # isolates the offset inside the current cube.


                # Time offset within cube only
                exp_time_per_frame = 0.4  # s (standard subarray)
                t_offset_start = (aintbeg % (64 * exp_time_per_frame)) / 86400.0
                t_offset_end = (atimeend % (64 * exp_time_per_frame)) / 86400.0


                # Convert BMJD to full Julian Date and add within-cube offsets.
                # 2400000.5 converts MJD-like values to JD.
                #
                # Note:
                # The comment mentions UTC→TDB, but this code does NOT actually add a TDB correction.
                # It keeps times effectively in BJD_UTC-style form, consistent with the later output.
                #
                # np.linspace(jd_start, jd_end, 64) assigns one time to each of the 64 images.
                jd_start = bmjd + 2400000.5 + t_offset_start
                jd_end = bmjd + 2400000.5 + t_offset_end
                times = np.linspace(jd_start, jd_end, 64)


                # Append this cube's 64 timestamps and 64 images to the master lists.
                all_times.append(times)
                all_data.append(data)
            
            # Progress indicator every 100 cubes
            # Helpful because a full Spitzer phase-curve dataset can contain a huge number of files.
            if (i + 1) % 100 == 0:
                print(f"    Processed {i+1}/{len(bcd_files)} cubes...")
        
        print(f"  ✓ Finished AOR {aor}")
    
    # Concatenate all
    # After this step:
    # - all_times becomes a single 1D array with one entry per image
    # - all_data becomes a single 3D array with shape (N_images, 32, 32)
    all_times = np.concatenate(all_times)
    all_data = np.concatenate(all_data, axis=0)
    
    print(f"\nTotal images (all AORs): {len(all_times)}")
    print(f"BJD range (all): {all_times.min():.5f} to {all_times.max():.5f}")
    
    # ============================================================
    # FILTER TO JULY 2011 CAMPAIGN ONLY (Lewis Figure 2)
    # ============================================================
    # Your disk may contain more Spitzer observations than just the exact 4.5 μm
    # phase-curve campaign you want. This block keeps only the time range corresponding
    # to the July 2011 full-orbit campaign used for the Lewis 4.5 μm analysis.
    #
    # After this step, all later processing is done only on the science images for that run.
    print("\n" + "=" * 60)
    print("FILTERING TO JULY 2011 CAMPAIGN (Lewis Figure 2)")
    print("=" * 60)


    bjd_start = 2455751.0
    bjd_end = 2455757.5



    # Build a boolean mask that is True only for times inside the desired campaign window.
    mask = (all_times >= bjd_start) & (all_times <= bjd_end)
    all_times = all_times[mask]
    all_data = all_data[mask]


    print(f"Filtered to BJD {bjd_start:.0f} - {bjd_end:.0f}")
    print(f"  Images before filter: {len(all_times)+len(all_data)*0}")
    print(f"  Images after filter: {len(all_times)}")
    print(f"  BJD range: {all_times.min():.1f} to {all_times.max():.1f}")
    print(f"  Duration: {(all_times.max()-all_times.min())*24:.1f} hours")
    print(f"  Expected: ~144 hours (Lewis Figure 2)")


    if len(all_times) == 0:
        print("No data in 24555751-24555757 range!")
        print("Available BJD ranges from diagnostic:")
        print(f"  {all_times.min():.0f} to {all_times.max():.0f}")
        return np.array([]), np.array([])



    # At this point, the function returns a synchronized pair:
    # each time in all_times corresponds to one 32x32 image in all_data.
    return all_times, all_data



# ============================================================
# 2. BACKGROUND ESTIMATION (Lewis Section 2.1)
# ============================================================
# Every image contains not only starlight but also a low-level background.
# If you do not subtract that background, your measured stellar flux will be biased.
#
# The basic idea:
#   1) ignore the star itself by masking out the central region
#   2) look only at "background" pixels farther from the center
#   3) reject weird outlier pixels
#   4) estimate one representative sky value for the image
#
# This code does that separately for every single 32x32 frame.

def estimate_background(image, exclude_radius=10, sigma_clip=3.0):
    """
    Estimate sky background using iterative 3σ clipping.
    
    Lewis+2013: "We determine the background level in each image from 
    the region outside of a 10 pixel radius from the central pixel. 
    We iteratively trim 3σ outliers from the background pixels then 
    fit a Gaussian to a histogram of the remaining pixel values."
    
    Parameters
    ----------
    image : array (32, 32)
        Single BCD image
    exclude_radius : float
        Exclude pixels within this radius of center (pixels)
    sigma_clip : float
        Sigma threshold for iterative clipping
        
    Returns
    -------
    bg : float
        Background flux level
    """
    
    # Create mask: exclude central region
    # The star is expected to sit near the center of the subarray.
    # We therefore reject pixels near the center to avoid contaminating the
    # background estimate with the stellar PSF.
    ny, nx = image.shape
    cy, cx = ny / 2.0, nx / 2.0
    yy, xx = np.ogrid[:ny, :nx]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    
    mask_bg = dist > exclude_radius
    bg_pixels = image[mask_bg]
    
    # Iterative 3σ clipping
    # This removes unusually high or low pixels from the background sample.
    # Those outliers could come from hot pixels, cosmic rays, residual source flux,
    # or other detector artifacts.
    for _ in range(3):
        med = np.median(bg_pixels)
        std = np.std(bg_pixels)
        mask_good = np.abs(bg_pixels - med) < sigma_clip * std
        bg_pixels = bg_pixels[mask_good]
    
    # Fit Gaussian to histogram to get final background estimate
    # (Lewis: "fit a Gaussian to a histogram of the remaining pixel values")
    #
    # Strictly speaking, this code does not perform a parametric Gaussian curve fit.
    # Instead, it builds a histogram and computes the weighted center of that histogram.
    # That weighted center acts as the final scalar background estimate.
    hist, bin_edges = np.histogram(bg_pixels, bins=50)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Simple centroid of histogram
    # This is the histogram-weighted mean pixel value after clipping.
    bg = np.sum(bin_centers * hist) / np.sum(hist)
    
    return bg



def subtract_backgrounds(data):
    """
    Subtract background from all images.
    
    Parameters
    ----------
    data : array (N, 32, 32)
        All BCD images
        
    Returns
    -------
    data_bgsub : array (N, 32, 32)
        Background-subtracted images
    bg_levels : array (N,)
        Background flux for each image
    """
    
    print("\n" + "=" * 60)
    print("SUBTRACTING BACKGROUNDS")
    print("=" * 60)
    
    N = data.shape[0]
    data_bgsub = np.zeros_like(data)
    bg_levels = np.zeros(N)
    
    # Process each image independently because the background can vary from frame to frame.
    for i in range(N):
        bg = estimate_background(data[i], 
                                 exclude_radius=BACKGROUND_RADIUS,
                                 sigma_clip=BACKGROUND_SIGMA_CLIP)
        # Subtract one scalar background level from the whole frame.
        # This assumes the background is approximately flat across the 32x32 image.
        data_bgsub[i] = data[i] - bg
        bg_levels[i] = bg
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{N} images...")
    
    print(f"✓ Background subtracted from {N} images")
    print(f"  Median background: {np.median(bg_levels):.2f} MJy/sr")
    
    # Return both the cleaned images and the measured background time series.
    # The background levels can be useful later for diagnosing instrumental behavior.
    return data_bgsub, bg_levels



# ============================================================
# 3. HOT PIXEL CORRECTION (Lewis Section 2.1)
# ============================================================
# Even after background subtraction, some individual detector pixels can temporarily
# spike high or low. These are "transient hot pixels."
#
# Lewis' idea:
#   within each 64-frame cube, compare each pixel position against its typical value
#   across the cube. If a pixel value is wildly inconsistent, replace it with the
#   median value for that same detector position.
#
# This is a smart choice because each 64-frame block is short in time, so the star and
# background should not change dramatically from one frame to the next.

def correct_hot_pixels(data, sigma=4.5):
    """
    Flag and replace hot pixels.
    
    Lewis+2013: "We correct for transient hot pixels by flagging pixels 
    more than 4.5σ away from the median flux at a given pixel position 
    across each set of 64 images then replacing flagged pixels by their 
    corresponding position median value."
    
    Parameters
    ----------
    data : array (N, 32, 32)
        Background-subtracted images
    sigma : float
        Sigma threshold for hot pixel detection
        
    Returns
    -------
    data_clean : array (N, 32, 32)
        Hot-pixel corrected images
    """
    
    print("\n" + "=" * 60)
    print("CORRECTING HOT PIXELS")
    print("=" * 60)
    
    N = data.shape[0]
    data_clean = data.copy()
    
    # Process in chunks of 64 (matching Lewis: "across each set of 64 images")
    # This mirrors the native subarray cube size in the BCD files.
    n_hot_total = 0
    
    for start in range(0, N, 64):
        end = min(start + 64, N)
        chunk = data[start:end]
        
        # Median flux at each pixel position across this chunk
        # med_chunk[y, x] = typical value of detector pixel (x, y) over these 64 frames
        # std_chunk[y, x] = natural variation of that detector pixel over the same frames
        med_chunk = np.median(chunk, axis=0)  # shape: (32, 32)
        std_chunk = np.std(chunk, axis=0)
        
        # Flag hot pixels
        # For each image in the chunk:
        #   compare every pixel to the usual value for that detector location.
        # If it is too far away, replace it with the local median.
        for i in range(chunk.shape[0]):
            residual = np.abs(chunk[i] - med_chunk)
            hot_mask = residual > sigma * std_chunk
            
            # Replace with median
            data_clean[start + i][hot_mask] = med_chunk[hot_mask]
            n_hot_total += np.sum(hot_mask)
    
    print(f"✓ Corrected {n_hot_total} hot pixels ({n_hot_total/(N*32*32)*100:.3f}% of all pixels)")
    
    return data_clean



# ============================================================
# 4. FLUX-WEIGHTED CENTROIDS (Lewis Section 2.1)
# ============================================================
# Before measuring stellar flux, we need to know where the star actually is in each frame.
# Spitzer pointing drifts slightly over time, so the star does not stay at exactly the same
# detector coordinates.
#
# This section computes a center-of-light (flux-weighted) centroid for each image.
#
# Why this matters:
# - The exact center determines which pixels fall inside the photometric aperture.
# - Small centroid shifts are strongly tied to intrapixel sensitivity systematics.
# - Tracking x and y is therefore essential for both Phase 1 diagnostics and Phase 2 corrections.

def compute_flux_weighted_centroid(image, aperture_radius=3.5):
    """
    Compute flux-weighted centroid.
    
    Lewis+2013: "For these data, we calculate the flux-weighted 
    centroid for each background subtracted image using a range 
    of aperture sizes... 3.5 pixels for the 4.5 μm observations."
    
    Parameters
    ----------
    image : array (32, 32)
        Background-subtracted, hot-pixel corrected image
    aperture_radius : float
        Aperture size for centroid calculation (pixels)
        
    Returns
    -------
    cx, cy : float
        Centroid position (pixels)
    """
    
    ny, nx = image.shape
    cy_guess, cx_guess = ny / 2.0, nx / 2.0
    
    # Create coordinate grids (use meshgrid for 2D)
    # xx[y, x] stores the x-coordinate of each pixel
    # yy[y, x] stores the y-coordinate of each pixel
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    
    # Create aperture mask
    # We only use pixels near the expected star location so distant noisy pixels
    # do not distort the centroid.
    dist = np.sqrt((xx - cx_guess)**2 + (yy - cy_guess)**2)
    mask = dist <= aperture_radius
    
    # Flux-weighted centroid
    # Formula:
    #   cx = sum(x * flux) / sum(flux)
    #   cy = sum(y * flux) / sum(flux)
    #
    # This is the "center of light" inside the chosen centroid aperture.
    flux_sum = np.sum(image[mask])
    
    if flux_sum <= 0:
        # Fallback to geometric center
        # This protects against pathological cases where the selected pixels sum to
        # zero or negative after background subtraction.
        return cx_guess, cy_guess
    
    cx = np.sum(xx[mask] * image[mask]) / flux_sum
    cy = np.sum(yy[mask] * image[mask]) / flux_sum
    
    return cx, cy



def compute_all_centroids(data, aperture_radius=3.5):
    """
    Compute centroids for all images.
    
    Parameters
    ----------
    data : array (N, 32, 32)
        Background-subtracted, hot-pixel corrected images
    aperture_radius : float
        Aperture size for centroid (pixels)
        
    Returns
    -------
    x_cent, y_cent : arrays (N,)
        Centroid positions
    """
    
    print("\n" + "=" * 60)
    print("COMPUTING FLUX-WEIGHTED CENTROIDS")
    print("=" * 60)
    
    N = data.shape[0]
    x_cent = np.zeros(N)
    y_cent = np.zeros(N)
    
    # Compute one centroid per image.
    # The output arrays line up 1-to-1 with the time array from the loading step.
    for i in range(N):
        x_cent[i], y_cent[i] = compute_flux_weighted_centroid(data[i], aperture_radius)
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{N} images...")
    
    print(f"✓ Centroids computed for {N} images")
    print(f"  X range: {x_cent.min():.3f} to {x_cent.max():.3f} pixels")
    print(f"  Y range: {y_cent.min():.3f} to {y_cent.max():.3f} pixels")
    
    return x_cent, y_cent



# ============================================================
# 5. APERTURE PHOTOMETRY (Lewis Section 2.1)
# ============================================================
# Now that we know where the star is, we measure its brightness.
#
# Basic aperture photometry idea:
#   draw a circle centered on the star
#   sum the pixel values inside that circle
#
# Because the background was already subtracted, this sum is our estimate of the
# stellar signal in that image.

def aperture_photometry(image, cx, cy, aperture_radius=2.25):
    """
    Circular aperture photometry.
    
    Lewis+2013: "We estimate the stellar flux from each background 
    subtracted image using circular aperture photometry... a fixed 
    2.25 pixel aperture gives the lowest scatter in the final solution."
    
    Parameters
    ----------
    image : array (32, 32)
        Background-subtracted, hot-pixel corrected image
    cx, cy : float
        Centroid position (pixels)
    aperture_radius : float
        Aperture radius (pixels)
        
    Returns
    -------
    flux : float
        Total flux in aperture
    """
    
    ny, nx = image.shape
    yy, xx = np.ogrid[:ny, :nx]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    mask = dist <= aperture_radius
    
    # Sum all pixel values inside the photometric aperture.
    # This is a simple "hard-edged" circular aperture with no fractional-pixel weighting.
    flux = np.sum(image[mask])
    
    return flux



def extract_photometry(data, x_cent, y_cent, aperture_radius=2.25):
    """
    Extract aperture photometry for all images.
    
    Parameters
    ----------
    data : array (N, 32, 32)
        Background-subtracted, hot-pixel corrected images
    x_cent, y_cent : arrays (N,)
        Centroid positions
    aperture_radius : float
        Aperture radius (pixels)
        
    Returns
    -------
    flux : array (N,)
        Stellar flux
    """
    
    print("\n" + "=" * 60)
    print("EXTRACTING APERTURE PHOTOMETRY")
    print("=" * 60)
    
    N = data.shape[0]
    flux = np.zeros(N)
    
    # For each image:
    #   center the aperture on that image's measured centroid
    #   sum the flux in the aperture
    #
    # The result is a raw light curve before later decorrelation steps.
    for i in range(N):
        flux[i] = aperture_photometry(data[i], x_cent[i], y_cent[i], aperture_radius)
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{N} images...")
    
    print(f"✓ Photometry extracted for {N} images")
    
    return flux



# ============================================================
# 6. NOISE PIXEL CALCULATION (Lewis Figure 1)
# ============================================================
# The "noise pixel" quantity is a diagnostic related to the effective width/shape
# of the PSF. In Spitzer work, it is often used as a proxy for image quality and,
# in some analyses, to define variable aperture sizes.
#
# Important note:
#   The exact canonical Spitzer noise-pixel formula is usually based on image moments.
#   This code instead uses a simpler proxy: the number of pixels above half the peak.
#
# So this is a diagnostic quantity, not a strict reproduction of the standard formal
# definition. It is still useful for tracking PSF changes versus time.
# TODO: THIS IS USELESS FOR CHANNEL 2. NO NEED TO USE NOISE PIXELS.
 
# def compute_noise_pixels(data):
#     """
#     Compute "noise pixels" as shown in Lewis Figure 1.
    
#     This is a proxy for PSF width / image quality.
#     Lewis+2013 Appendix A mentions this but doesn't define exactly;
#     we estimate as the number of pixels above half-max in each image.
    
#     Parameters
#     ----------
#     data : array (N, 32, 32)
#         Background-subtracted images
        
#     Returns
#     -------
#     noise_pix : array (N,)
#         Noise pixel proxy
#     """
    
#     print("\n" + "=" * 60)
#     print("COMPUTING NOISE PIXEL PROXY")
#     print("=" * 60)
    
#     N = data.shape[0]
#     noise_pix = np.zeros(N)
    
#     # For each image:
#     #   find the peak pixel
#     #   count how many pixels exceed half that peak
#     #
#     # A broader PSF tends to spread light over more pixels, so this count tends to grow.
#     for i in range(N):
#         peak = np.max(data[i])
#         half_max = peak / 2.0
#         noise_pix[i] = np.sum(data[i] > half_max)
    
#     print(f"✓ Noise pixels computed for {N} images")
#     print(f"  Median noise pixels: {np.median(noise_pix):.1f}")
    
#     return noise_pix



# ============================================================
# 7. OUTLIER REMOVAL (Lewis Section 2.1)
# ============================================================
# Once the raw photometry is extracted, some points will still be bad:
# - images hit by residual detector issues
# - centroid failures
# - frames with unusual flux excursions
#
# To remove them, the code compares each flux value to a local moving median.
# If a point lies too far away from the nearby trend, it is discarded.
#
# This is done after photometry extraction because we care about outliers in the
# final time series, not just in individual image pixels.

def remove_outliers(times, flux, x, y, sigma=4.5, window=16):
    """
    Remove outliers using moving median filter.
    
    Lewis+2013: "We remove outliers from our final photometric data 
    sets by discarding points more than 4.5σ away from a moving 
    boxcar median 16 points wide."
    
    Parameters
    ----------
    times, flux, x, y, noise_pix : arrays (N,)
        Photometry and auxiliary data
    sigma : float
        Outlier threshold
    window : int
        Moving median window width
        
    Returns
    -------
    All inputs with outliers removed
    """
    
    print("\n" + "=" * 60)
    print("REMOVING OUTLIERS")
    print("=" * 60)
    
    N = len(flux)
    
    # Compute moving median
    # median_filter creates a local baseline using nearby points in time order.
    # This helps preserve long-term astrophysical/instrumental trends while still
    # flagging isolated spikes.
    from scipy.ndimage import median_filter
    flux_median = median_filter(flux, size=window, mode='nearest')
    
    # Flag outliers
    # residual = distance between each point and its local median baseline
    # std      = overall scatter of those residuals
    # good     = points not too far from the local baseline
    residual = np.abs(flux - flux_median)
    std = np.std(residual)
    good = residual < sigma * std
    
    n_outliers = N - np.sum(good)
    print(f"  Flagged {n_outliers} outliers ({n_outliers/N*100:.2f}%)")
    
    # Remove the same bad points from every synchronized array.
    # This is crucial: time, flux, x, y, and noise_pix must always stay aligned.
    times = times[good]
    flux = flux[good]
    x = x[good]
    y = y[good]
    # noise_pix = noise_pix[good]
    
    print(f"✓ {len(flux)} points remaining after outlier removal")
    
    return times, flux, x, y



# ============================================================
# 8. NORMALIZE FLUX
# ============================================================
# Raw aperture sums are in detector flux units, not convenient relative units.
# For plotting and later correction work, it is much more useful to divide by a
# representative scale so the baseline is near 1.
#
# Here the code uses the median flux of the whole dataset.

def normalize_flux(flux):
    """
    Normalize flux to median = 1.0.
    
    Parameters
    ----------
    flux : array (N,)
        Raw flux
        
    Returns
    -------
    flux_norm : array (N,)
        Normalized flux
    median_flux : float
        Median used for normalization
    """
    
    print("\n" + "=" * 60)
    print("NORMALIZING FLUX")
    print("=" * 60)
    
    median_flux = np.median(flux)
    flux_norm = flux / median_flux
    
    print(f"  Median flux: {median_flux:.2f}")
    print(f"  Normalized range: {flux_norm.min():.4f} to {flux_norm.max():.4f}")
    
    return flux_norm, median_flux



# ============================================================
# 9. PLOT RAW DATA (Lewis Figure 2 style)
# ============================================================
# This function makes the main diagnostic figure for Phase 1.
#
# It plots:
#   (a) x centroid versus time
#   (b) y centroid versus time
#   (c) noise pixel proxy versus time
#   (d) raw normalized photometry versus time
#
# Why this figure matters:
# - It lets you visually inspect pointing drift and reacquisition jumps.
# - It shows whether flux variations correlate with x/y motion.
# - It gives a sanity check that your extracted raw light curve resembles the
#   published Lewis-style diagnostics.
def plot_raw_diagnostics(times, x, y, flux_norm, save_path="spitzer_raw_diagnostics.png"):
    """
    Generate Figure 2-style diagnostic plots.
    
    Lewis Figure 2: Shows x position, y position, noise pixels, 
    and raw photometry vs observation time.
    
    Parameters
    ----------
    times : array (N,)
        BJD_UTC times
    x, y : arrays (N,)
        Centroid positions
    noise_pix : array (N,)
        Noise pixel proxy
    flux_norm : array (N,)
        Normalized flux
    save_path : str
        Output filename
    """
    
    print("\n" + "=" * 60)
    print("GENERATING RAW DIAGNOSTIC PLOTS (Lewis Figure 2 style)")
    print("=" * 60)
    
    # Convert to observation time (hours from start)
    # This makes the x-axis easier to interpret visually than absolute BJD.
    t_hours = (times - times.min()) * 24.0
    
    # FAST binning using pandas (much faster than loop)
    # Lewis bins the diagnostic plots in time to reduce scatter and make long-term
    # trends visible. Here the code bins to 3-minute intervals.
    bin_size = 3.0 / 60.0  # 3 min in hours
    
    import pandas as pd
    df = pd.DataFrame({
        't': t_hours,
        'x': x,
        'y': y,
        'flux': flux_norm
    })
    
    # Bin by rounding to nearest bin
    # This assigns each point to a 3-minute time bin, then takes the median within each bin.
    # Median binning is robust against occasional remaining outliers.
    df['bin'] = (df['t'] / bin_size).round() * bin_size
    df_binned = df.groupby('bin').median().reset_index()
    
    t_bin = df_binned['bin'].values
    x_bin = df_binned['x'].values
    y_bin = df_binned['y'].values
    # noise_bin = df_binned['noise'].values
    flux_bin = df_binned['flux'].values
    
    print(f"  Binned {len(df)} points into {len(df_binned)} bins")
    
    # Create 4-panel plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Panel (a): X position
    # This shows how the stellar centroid moves in x across the detector over time.
    axes[0].plot(t_bin, x_bin, 'k.', ms=2, alpha=0.6)
    axes[0].axhline(16, color='gray', ls='-', lw=1, label='pixel center')
    axes[0].axhline(15, color='gray', ls='--', lw=0.8, alpha=0.7)
    axes[0].axhline(17, color='gray', ls='--', lw=0.8, alpha=0.7)
    axes[0].set_ylabel("X Position (pixels)")
    axes[0].set_ylim(x_bin.min() - 0.1, x_bin.max() + 0.1)
    axes[0].text(0.02, 0.95, '(a)', transform=axes[0].transAxes, 
                 fontsize=12, va='top', fontweight='bold')
    
    # Panel (b): Y position
    # Same idea as panel (a), but for y. Spitzer often shows clear long-term drift/oscillation here.
    axes[1].plot(t_bin, y_bin, 'k.', ms=2, alpha=0.6)
    axes[1].axhline(16, color='gray', ls='-', lw=1, label='pixel center')
    axes[1].axhline(15, color='gray', ls='--', lw=0.8, alpha=0.7)
    axes[1].axhline(17, color='gray', ls='--', lw=0.8, alpha=0.7)
    axes[1].set_ylabel("Y Position (pixels)")
    axes[1].set_ylim(y_bin.min() - 0.1, y_bin.max() + 0.1)
    axes[1].text(0.02, 0.95, '(b)', transform=axes[1].transAxes, 
                 fontsize=12, va='top', fontweight='bold')
    
    # Panel (c): Noise pixels
    # This tracks changes in the effective PSF width/shape over time.
    # axes[2].plot(t_bin, noise_bin, 'k.', ms=2, alpha=0.6)
    axes[2].set_ylabel("Noise Pixels")
    # axes[2].set_ylim(noise_bin.min() - 0.5, noise_bin.max() + 0.5)
    axes[2].text(0.02, 0.95, '(c)', transform=axes[2].transAxes, 
                 fontsize=12, va='top', fontweight='bold')
    
    # Panel (d): Raw photometry
    # This is the extracted raw light curve before Phase 2 decorrelation/modeling.
    axes[2].plot(t_bin, flux_bin, 'k.', ms=2, alpha=0.6)
    axes[2].set_ylabel("Relative Flux")
    axes[2].set_xlabel("Observation Time (hours)")
    axes[2].set_ylim(flux_bin.min() - 0.005, flux_bin.max() + 0.005)
    axes[2].text(0.02, 0.95, '(d)', transform=axes[2].transAxes, 
                 fontsize=12, va='top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved {save_path}")
    plt.close()




# ============================================================
# MAIN PIPELINE
# ============================================================
# This function stitches together every processing stage in the correct order.
#
# High-level flow:
#   raw FITS cubes
#   -> background-subtracted images
#   -> hot-pixel-corrected images
#   -> centroids
#   -> raw aperture photometry
#   -> noise-pixel diagnostic
#   -> outlier-cleaned time series
#   -> normalized raw light curve
#   -> diagnostic figure + CSV output
#
# If you ever want to explain Phase 1 out loud to your mentor, this order is the
# cleanest summary of the whole script.
def main(apply_background=True, apply_hot_pixels=True, label="full"):
    """
    Run Phase 1: Raw photometry extraction following Lewis+2013 Section 2.1.
    """
    print("\n" + "=" * 60)
    print("SPITZER 4.5 μm RAW PHOTOMETRY PIPELINE")
    print("Following Lewis+2013 (ApJ 766:95) Section 2.1")
    print("=" * 60 + "\n")
    
    # Step 1: Load BCD files
    # Read raw images and reconstruct one timestamp per frame.
    times, data = load_spitzer_bcd_files(BASE_PATH, channel="ch2")
    
    # Step 2: Background subtraction
    if apply_background:
        data_bgsub, bg_levels = subtract_backgrounds(data)
    else:
        print("\nSkipping background subtraction")
        data_bgsub = data.copy()
        bg_levels = np.zeros(len(data))

    # Step 3: Hot pixel correction
    if apply_hot_pixels:
        data_clean = correct_hot_pixels(data_bgsub, sigma=HOT_PIXEL_SIGMA)
    else:
        print("\nSkipping hot pixel correction")
        data_clean = data_bgsub.copy()
    
    # Step 4: Flux-weighted centroids
    # Measure the star's x and y position in every cleaned image.
    x_cent, y_cent = compute_all_centroids(data_clean, aperture_radius=CENTROID_APERTURE)
    
    # Step 5: Aperture photometry
    # Measure raw stellar flux using a fixed circular aperture centered on the centroid.
    flux = extract_photometry(data_clean, x_cent, y_cent, aperture_radius=PHOTOMETRY_APERTURE)
    
    # Step 6: Noise pixel calculation
    # Compute a PSF-width/quality diagnostic for each image.
    # noise_pix = compute_noise_pixels(data_clean)
    
    # Step 7: Remove outliers
    # Reject obviously bad photometric points while keeping all output arrays aligned.
    times, flux, x_cent, y_cent = remove_outliers(
        times, flux, x_cent, y_cent,
        sigma=OUTLIER_SIGMA, window=OUTLIER_WINDOW
    )
    
    # Step 8: Normalize
    # Divide the raw fluxes by their median so the light curve baseline is near 1.
    flux_norm, median_flux = normalize_flux(flux)
    
    plot_raw_diagnostics(
        times, x_cent, y_cent, flux_norm,
        save_path=f"spitzer_raw_diagnostics_{label}.png"
    )

    df = pd.DataFrame({
        'BJD_UTC': times,
        'flux_norm': flux_norm,
        'x_cent': x_cent,
        'y_cent': y_cent
    })
    df.to_csv(f"spitzer_raw_photometry_{label}.csv", index=False)
        
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"  Total points: {len(flux_norm)}")
    print(f"  Time span: {times.max() - times.min():.2f} days")
    print(f"  Next: Phase 2 will apply ramp correction, intrapixel sensitivity, and model")


if __name__ == "__main__":
    # Full pipeline (all corrections applied)
    main(
        apply_background=True,
        apply_hot_pixels=True,
        label="full"
    )

    # Test 1: No background subtraction
    main(
        apply_background=False,
        apply_hot_pixels=True,
        label="no_background"
    )

    # Test 2: No hot pixel correction
    main(
        apply_background=True,
        apply_hot_pixels=False,
        label="no_hot_pixels"
    )