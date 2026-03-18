import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import os
import corner
import scipy.stats as stats
import batman
import emcee

# ============================================================
# CONFIG / CONSTANTS
# ============================================================

BASE_PATH = r"C:\Users\robob\Documents\astro research; stellar pulsations\Spitzer datasets"

# HAT-P-2b orbital parameters from Lewis+2013 (Table 4)
P_ORB = 5.6334729          # days
ECC  = 0.50910
OMEGA_DEG = 188.09

# Ephemerides (BJD_UTC) from Lewis+2013 (Table 4)
E_TRANSIT = 2455288.84923  # mid-transit
E_SEC     = 2455289.93211  # mid-secondary
E_PERI    = 2455289.4721   # periapse passage

PHI_SEC   = 0.19253        # orbital phase of secondary eclipse

# Approximate secondary eclipse duration at 4.5 um (Table 4)
T14_SEC   = 0.1650         # days

# Geometry / radius from Lewis+2013 (weighted average)
A_R_STAR_MEAN  = 8.70
A_R_STAR_SIG   = 0.20
INC_MEAN_DEG   = 85.97
INC_SIG_DEG    = 0.30
RP_RSTAR_MEAN  = 0.06933
RP_RSTAR_SIG   = 0.0020

# Limb darkening for IRAC 4.5 µm:
# (these are placeholders; update with your preferred values)
U1_4P5 = 0.05
U2_4P5 = 0.05

# Transit/eclipse windows (phase around transit and secondary)
PHASE_WINDOW_TRANSIT   = 0.03   # half-width around phase 0
PHASE_WINDOW_ECLIPSE   = 0.03   # half-width around phi_sec

# Subarray cube properties
N_FRAMES_PER_CUBE = 64
SUBARRAY_SIZE = 32

# ============================================================
# BASIC UTILS
# ============================================================

def sigma_clip(data, sigma=3.0, max_iters=5):
    mask = np.isfinite(data)
    for _ in range(max_iters):
        if not np.any(mask):
            break
        d = data[mask]
        med = np.median(d)
        std = np.std(d)
        new_mask = (data > med - sigma*std) & (data < med + sigma*std) & mask
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask
    return mask

def compute_bjd_times_from_header(header):
    """
    Compute BJD_UTC mid-exposure times for each of the 64 images.
    Use BMJD_OBS (not MBJD_OBS) + AINTBEG/ATIMEEND as in IRAC BCD headers. [web:22]
    """
    bmjd = header.get("BMJD_OBS", None)
    aintbeg = header.get("AINTBEG", 0.0)
    atimeend = header.get("ATIMEEND", 0.0)

    if bmjd is None:
        print(f"DEBUG: No BMJD_OBS in header!")
        return None

    # AINTBEG/ATIMEEND are in seconds since IRAC turn-on; convert to days
    t_start = bmjd + aintbeg / 86400.0
    t_end   = bmjd + atimeend / 86400.0

    dt = (t_end - t_start) / N_FRAMES_PER_CUBE

    # center of each frame
    frames = np.arange(N_FRAMES_PER_CUBE, dtype=float)
    times = t_start + (frames + 0.5) * dt
    return times


def background_and_photometry(cube):
    """
    Given a 64 x 32 x 32 cube of images, return per-frame:
    - flux (aperture photometry)
    - centroid x, y (flux-weighted)
    We do:
      * background from pixels outside radius 10
      * crude hot-pixel replacement (per-pixel median across frames)
      * centroid and aperture photometry on background-subtracted images
    """

    if cube.shape != (N_FRAMES_PER_CUBE, SUBARRAY_SIZE, SUBARRAY_SIZE):
        # Fallback: treat as full-frame but still loop
        n_frames = cube.shape[0]
        h, w = cube.shape[1:]
        y_grid, x_grid = np.indices((h, w))
        cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0
        r = np.sqrt((x_grid - cx0)**2 + (y_grid - cy0)**2)
        bg_mask = r >= 10.0

        # crude hot pixel median replacement
        median_image = np.nanmedian(cube, axis=0)
        std_image = np.nanstd(cube, axis=0)
        std_image[std_image == 0] = np.nanmedian(std_image[std_image > 0])

        fluxes = []
        xs = []
        ys = []
        for k in range(n_frames):
            img = cube[k].astype(float)
            # hot pixels
            bad = np.abs(img - median_image) > 4.5 * std_image
            img[bad] = median_image[bad]

            bg_vals = img[bg_mask]
            clip = sigma_clip(bg_vals, sigma=3.0)
            bg = np.median(bg_vals[clip]) if np.any(clip) else np.median(bg_vals)

            img_sub = img - bg

            total_flux = np.sum(img_sub)
            if total_flux <= 0:
                fluxes.append(np.nan)
                xs.append(np.nan)
                ys.append(np.nan)
                continue

            x_cen = np.sum(x_grid * img_sub) / total_flux
            y_cen = np.sum(y_grid * img_sub) / total_flux

            # aperture radius 2.25 px
            r_ap = np.sqrt((x_grid - x_cen)**2 + (y_grid - y_cen)**2)
            ap_mask = r_ap <= 2.25
            flux = np.sum(img_sub[ap_mask])

            fluxes.append(flux)
            xs.append(x_cen)
            ys.append(y_cen)

        return np.array(fluxes), np.array(xs), np.array(ys)

    # subarray case (32x32)
    n_frames, h, w = cube.shape
    y_grid, x_grid = np.indices((h, w))
    cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.sqrt((x_grid - cx0)**2 + (y_grid - cy0)**2)
    bg_mask = r >= 10.0

    # hot pixel median replacement
    median_image = np.nanmedian(cube, axis=0)
    std_image = np.nanstd(cube, axis=0)
    std_image[std_image == 0] = np.nanmedian(std_image[std_image > 0])

    fluxes = []
    xs = []
    ys = []
    for k in range(n_frames):
        img = cube[k].astype(float)
        bad = np.abs(img - median_image) > 4.5 * std_image
        img[bad] = median_image[bad]

        bg_vals = img[bg_mask]
        clip = sigma_clip(bg_vals, sigma=3.0)
        bg = np.median(bg_vals[clip]) if np.any(clip) else np.median(bg_vals)

        img_sub = img - bg

        total_flux = np.sum(img_sub)
        if total_flux <= 0:
            fluxes.append(np.nan)
            xs.append(np.nan)
            ys.append(np.nan)
            continue

        x_cen = np.sum(x_grid * img_sub) / total_flux
        y_cen = np.sum(y_grid * img_sub) / total_flux

        r_ap = np.sqrt((x_grid - x_cen)**2 + (y_grid - y_cen)**2)
        ap_mask = r_ap <= 2.25
        flux = np.sum(img_sub[ap_mask])

        fluxes.append(flux)
        xs.append(x_cen)
        ys.append(y_cen)

    return np.array(fluxes), np.array(xs), np.array(ys)


def load_spitzer_4p5_lightcurve(base_path):
    """
    Walk all AOR directories, read subarray BCD cubes, and build
    a combined light curve (time BJD_UTC, flux, x, y, aor_id).
    We assume the directory tree is like base_path / <AOR> / r<AOR> / ... *_bcd.fits
    and that these are the 4.5 µm IRAC channel 2 BCDs. [file:9]
    """

    aor_dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    all_t = []
    all_f = []
    all_x = []
    all_y = []
    all_aor = []

    print(f"Found {len(aor_dirs)} AOR directories\n")

    total_cubes = 0
    for aor in aor_dirs:
        aor_path = os.path.join(base_path, aor, f"r{aor}", "ch2", "bcd")
        if not os.path.exists(aor_path):
            continue

        # IRAC ch2 BCD naming from your tree: SPITZER_I2_<AORID>_*.fits
        bcd_files = glob.glob(os.path.join(aor_path, "SPITZER_I2_*_bcd.fits"))
        if not bcd_files:
            print(f"AOR {aor}: no SPITZER_I2_*.fits found in {aor_path}")
            continue

        print(f"AOR {aor}: {len(bcd_files)} BCD files in {aor_path}")


        for i, fits_file in enumerate(sorted(bcd_files)):
            try:
                with fits.open(fits_file, memmap=True) as hdul:
                    header = hdul[0].header
                    data = hdul[0].data

                    times = compute_bjd_times_from_header(header)
                    if times is None:
                        continue

                    # ensure we have full cube
                    if data.ndim == 3 and data.shape[0] >= N_FRAMES_PER_CUBE:
                        cube = data[:N_FRAMES_PER_CUBE, :, :]
                    else:
                        # treat as stack of 2D frames
                        if data.ndim == 2:
                            cube = data[None, :, :]
                        else:
                            cube = data

                    fluxes, xs, ys = background_and_photometry(cube)

                    n = len(fluxes)
                    all_t.append(times[:n])
                    all_f.append(fluxes)
                    all_x.append(xs)
                    all_y.append(ys)
                    all_aor.append(np.full(n, int(aor)))

                    total_cubes += 1
                    if (i + 1) % 100 == 0:
                        print(f"  Processed {i+1}/{len(bcd_files)} cubes...")

            except Exception as e:
                # skip problematic file
                continue

        print(f"  ✓ Finished AOR {aor}\n")

    if not all_t:
        print("No data found.")
        return None, None, None, None, None, None


    t = np.concatenate(all_t)
    f = np.concatenate(all_f)
    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    aor_id = np.concatenate(all_aor)

    # sort by time
    order = np.argsort(t)
    t = t[order]
    f = f[order]
    x = x[order]
    y = y[order]
    aor_id = aor_id[order]

    # remove NaNs
    good = np.isfinite(t) & np.isfinite(f) & np.isfinite(x) & np.isfinite(y)
    t = t[good]
    f = f[good]
    x = x[good]
    y = y[good]
    aor_id = aor_id[good]

    # normalize flux
    med_flux = np.median(f)
    f_norm = f / med_flux

    # simple uniform error estimate from global scatter
    ferr = np.ones_like(f_norm) * np.std(f_norm)

    print(f"Total images: {len(t)}")
    print(f"BJD range: {t[0]:.5f} to {t[-1]:.5f} (span {t[-1] - t[0]:.2f} days)")

    return t, f_norm, ferr, x, y, aor_id


# ============================================================
# ORBITAL PHASE + WINDOW SELECTION
# ============================================================

def orbital_phase(t_bjd):
    """Orbital phase relative to E_TRANSIT, in [0,1)."""
    return ((t_bjd - E_TRANSIT) / P_ORB) % 1.0


def select_transit_and_eclipse_windows(t_bjd, f, ferr, x, y):
    """
    Keep only data near transit (phase ~0) and secondary eclipse (phase ~phi_sec).
    """
    phase = orbital_phase(t_bjd)

    # shift so transit is centered at 0 in [-0.5, 0.5]
    phase_c = phase.copy()
    phase_c[phase_c > 0.5] -= 1.0

    mask_transit = np.abs(phase_c) < PHASE_WINDOW_TRANSIT
    mask_ecl = np.abs(phase - PHI_SEC) < PHASE_WINDOW_ECLIPSE

    mask = mask_transit | mask_ecl

    return t_bjd[mask], f[mask], ferr[mask], x[mask], y[mask]


# ============================================================
# BATMAN SETUP + PLANET PHASE MODEL
# ============================================================

# base batman params; we will overwrite t0, rp, a, inc each call
base_params = batman.TransitParams()
base_params.per = P_ORB
base_params.ecc = ECC
base_params.w = OMEGA_DEG
base_params.rp = RP_RSTAR_MEAN
base_params.a = A_R_STAR_MEAN
base_params.inc = INC_MEAN_DEG
base_params.limb_dark = "quadratic"
base_params.u = [U1_4P5, U2_4P5]


def planet_flux_model(t, Fp_Fsmin, c1, c2, c3, c4, c5, c6):
    """
    Asymmetric Lorentzian phase curve in time (days from periapse),
    with eclipse window where planet flux = 0.
    This is your HST-style model, interpreted in units of days. [file:9]
    """
    u = np.where(t < c2, (t - c2) / c3, (t - c2) / c4)
    eclipse = (t > (c5 - 0.5 * c6)) & (t < (c5 + 0.5 * c6))
    model = np.where(eclipse, 0.0, Fp_Fsmin + c1 / (u**2 + 1.0))
    return model


def full_model(theta, t_bjd, x, y):
    """
    Combined model: transit + eclipse (batman) + planet phase curve + baseline.
    Times are in BJD_UTC; we convert to days from periapse for batman and planet. [file:9]
    """
    # unpack
    (
        t0_rel,    # days offset of transit from periapse
        rp, a, inc,
        Fp_Fsmin, c1, c2, c3, c4, c5, c6,
        b0, b1, b2,
        bx, by, bxx, byy, bxy,
        log_sigma_jit
    ) = theta

    # batman params
    base_params.t0 = t0_rel       # this is days from periapse; we pass t_rel to batman
    base_params.rp = rp
    base_params.a = a
    base_params.inc = inc
    base_params.u = [U1_4P5, U2_4P5]

    # time relative to periapse in days
    t_rel = t_bjd - E_PERI

    m = batman.TransitModel(base_params, t_rel, supersample_factor=3, exp_time=0.002)
    transit = m.light_curve(base_params)

    # planet phase curve
    planet = planet_flux_model(t_rel, Fp_Fsmin, c1, c2, c3, c4, c5, c6)
    # planet = 0

    # baseline: polynomial in time, x, y
    t0 = np.median(t_rel)
    dt = t_rel - t0

    x0 = np.median(x)
    y0 = np.median(y)
    dx = x - x0
    dy = y - y0

    baseline = (
        b0
        + b1 * dt
        + b2 * dt**2
        + bx * dx
        + by * dy
        + bxx * dx**2
        + byy * dy**2
        + bxy * dx * dy
    )

    # total flux (star normalized to 1.0)
    model_flux = transit + planet - Fp_Fsmin + baseline

    sigma_jit = np.exp(log_sigma_jit)

    return model_flux, sigma_jit


# ============================================================
# PRIORS + LOG-LIKELIHOOD
# ============================================================

def log_prior(theta):
    (
        t0_rel,
        rp, a, inc,
        Fp_Fsmin, c1, c2, c3, c4, c5, c6,
        b0, b1, b2,
        bx, by, bxx, byy, bxy,
        log_sigma_jit
    ) = theta

    # geometry priors (Gaussians around Lewis+2013 values) [file:9]
    lp = 0.0

    # t0_rel near expected transit time relative to periapse
    # from ephemeris: E_TRANSIT - E_PERI
    t0_mean = E_TRANSIT - E_PERI
    t0_sig = 0.01  # days
    lp += -0.5 * ((t0_rel - t0_mean) / t0_sig)**2

    # Rp/R*
    lp += -0.5 * ((rp - RP_RSTAR_MEAN) / RP_RSTAR_SIG)**2

    # a/R*
    lp += -0.5 * ((a - A_R_STAR_MEAN) / A_R_STAR_SIG)**2

    # inc
    lp += -0.5 * ((inc - INC_MEAN_DEG) / INC_SIG_DEG)**2

    # basic sanity bounds
    if not (0.04 < rp < 0.12):
        return -np.inf
    if not (5.0 < a < 15.0):
        return -np.inf
    if not (80.0 < inc < 90.0):
        return -np.inf

    # phase-curve priors (very loose, just enforce positivity and reasonable ranges) [file:9]
    if not (0.0 < Fp_Fsmin < 0.005):
        return -np.inf
    if not (0.0 < c1 < 0.01):
        return -np.inf

    # time offsets and widths in days
    # peak offset c2: around ~0.24 days after periapse (Lewis+2013 4.5 µm) [file:9]
    c2_mean = 5.84 / 24.0
    c2_sig = 0.05
    lp += -0.5 * ((c2 - c2_mean) / c2_sig)**2

    if not (0.0 < c3 < 1.0):
        return -np.inf
    if not (0.0 < c4 < 1.0):
        return -np.inf

    # eclipse center and duration (c5, c6) around Lewis+ eclipse time and duration [file:9]
    c5_mean = E_SEC - E_PERI
    c5_sig = 0.01
    lp += -0.5 * ((c5 - c5_mean) / c5_sig)**2

    c6_mean = T14_SEC
    c6_sig = 0.02
    lp += -0.5 * ((c6 - c6_mean) / c6_sig)**2

    if not (0.0 < c6 < 0.4):
        return -np.inf

    # weak priors on baseline coefficients and jitter
    if not (-1.0 < b0 < 1.0):
        return -np.inf
    if not (-10.0 < b1 < 10.0):
        return -np.inf
    if not (-10.0 < b2 < 10.0):
        return -np.inf

    for coeff in (bx, by, bxx, byy, bxy):
        if not (-10.0 < coeff < 10.0):
            return -np.inf

    if not (-15.0 < log_sigma_jit < 0.0):
        return -np.inf

    return lp


def log_likelihood(theta, t_bjd, f, ferr, x, y):
    model_flux, sigma_jit = full_model(theta, t_bjd, x, y)

    var = ferr**2 + sigma_jit**2
    resid = f - model_flux

    return -0.5 * np.sum(resid**2 / var + np.log(2.0 * np.pi * var))


def log_prob(theta, t_bjd, f, ferr, x, y):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t_bjd, f, ferr, x, y)


# ============================================================
# MAIN: LOAD DATA, RUN EMCEE
# ============================================================

def run_emcee_on_spitzer():
    # 1. load 4.5 µm light curve
    t, f, ferr, x, y, aor_id = load_spitzer_4p5_lightcurve(BASE_PATH)
    if t is None:
        return

    # 2. select windows around transit and eclipse
    t_sel, f_sel, ferr_sel, x_sel, y_sel = select_transit_and_eclipse_windows(t, f, ferr, x, y)
    print(f"Selected {len(t_sel)} points in transit+eclipse windows")

    # 3. initial guess for theta
    t0_init = E_TRANSIT - E_PERI

    # rough priors for phase curve:
    Fp_Fsmin_init = 0.0005
    c1_init = 0.0010
    c2_init = 5.84 / 24.0
    c3_init = 0.1
    c4_init = 0.1
    c5_init = E_SEC - E_PERI
    c6_init = T14_SEC

    b0_init = 0.0
    b1_init = 0.0
    b2_init = 0.0
    bx_init = 0.0
    by_init = 0.0
    bxx_init = 0.0
    byy_init = 0.0
    bxy_init = 0.0

    log_sigma_jit_init = np.log(0.001 * np.std(f_sel))

    theta_init = np.array([
        t0_init,
        RP_RSTAR_MEAN, A_R_STAR_MEAN, INC_MEAN_DEG,
        Fp_Fsmin_init, c1_init, c2_init, c3_init, c4_init, c5_init, c6_init,
        b0_init, b1_init, b2_init,
        bx_init, by_init, bxx_init, byy_init, bxy_init,
        log_sigma_jit_init
    ])

    ndim = len(theta_init)
    nwalkers = 40
    nsteps = 400

    # small random scatter around initial guess
    pos = theta_init + 1e-4 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob,
        args=(t_sel, f_sel, ferr_sel, x_sel, y_sel)
    )

    print(f"Running emcee: nwalkers={nwalkers}, nsteps={nsteps}, ndim={ndim}")
    sampler.run_mcmc(pos, nsteps, progress=True)

    flat = sampler.get_chain(discard=int(0.3 * nsteps), thin=10, flat=True)
    best = np.median(flat, axis=0)

    print("\n" + "=" * 60)
    print("BEST-FIT PARAMETERS (Spitzer 4.5 um)")
    print("=" * 60)
    labels = [
        "t0_rel", "rp", "a", "inc",
        "Fp_Fsmin", "c1", "c2", "c3", "c4", "c5", "c6",
        "b0", "b1", "b2",
        "bx", "by", "bxx", "byy", "bxy",
        "log_sigma_jit"
    ]
    for name, val in zip(labels, best):
        print(f"{name:12s} = {val:.6g}")
    print("=" * 60)

    # Make a quick model plot
    model_flux, sigma_jit = full_model(best, t_sel, x_sel, y_sel)
    resid = f_sel - model_flux
    var   = ferr_sel**2 + sigma_jit**2

    chi2 = np.sum(resid**2 / var)
    dof  = len(resid) - len(best)      # N_data - N_params
    chi2_red = chi2 / dof

    print(f"chi2       = {chi2:.2f}")
    print(f"chi2_red   = {chi2_red:.3f}")
    print(f"N_data     = {len(resid)}")
    print(f"N_params   = {len(best)}")

    z = resid / np.sqrt(var)

    # Optionally clip extreme outliers so they don't dominate
    z_clip = z[np.abs(z) < 5]

    skew = stats.skew(z_clip)
    kurt = stats.kurtosis(z_clip, fisher=True)  # this is k-3 (excess)

    print(f"Residual skewness      = {skew:.3f}")
    print(f"Residual excess kurtosis (k-3) = {kurt:.3f}")
    
    plt.figure(figsize=(9,4))
    plt.plot(t_sel, resid*1e3, "k.", ms=2, alpha=0.5)
    plt.axhline(0, color="r", ls="--")
    plt.xlabel("BJD_UTC")
    plt.ylabel("Residuals (mmag)")
    plt.tight_layout()
    plt.show()

    labels = [
        "t0_rel", "rp", "a", "inc",
        "Fp_Fsmin", "c1", "c2", "c3", "c4", "c5", "c6",
        "b0", "b1", "b2",
        "bx", "by", "bxx", "byy", "bxy",
        "log_sigma_jit",
    ]

    fig = corner.corner(flat, labels=labels, show_titles=True,
                        title_fmt=".4g", max_n_ticks=4)
    fig.savefig("spitzer_corner1.png", dpi=200)
    print("Saved spitzer_corner1.png")



if __name__ == "__main__":
    run_emcee_on_spitzer()
