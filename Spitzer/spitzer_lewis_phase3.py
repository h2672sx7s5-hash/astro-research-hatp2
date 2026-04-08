import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.ndimage import median_filter, gaussian_filter
import batman
import emcee

try:
    import corner
    HAVE_CORNER = True
except Exception:
    HAVE_CORNER = False


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = "spitzer_raw_photometry_p1.csv"

OUTPUT_MODEL_CSV = "phase3_corrected_model.csv"
OUTPUT_SUMMARY_CSV = "phase3_summary.csv"
OUTPUT_PHASE_FIG = "figure5phasecurve.png"
OUTPUT_ZOOM_FIG = "figure7zoomevents.png"
OUTPUT_CHAIN_NPY = "phase3_chain.npy"
OUTPUT_CORNER_FIG = "phase3_corner.png"

RANDOM_SEED = 42

SEGMENT_GAP_DAYS = 1.0 / 24.0   # 1 hour
TRIM_HOURS = 1.0

NBINS_MAP = 25
MIN_PTS_PER_BIN = 8
MAP_SMOOTH_SIGMA = 1.0

SUPERSAMPLE = 7
EXP_TIME_DAYS = 0.4 / 86400.0

SIGMA_CLIP_NSIG = 10.0
SIGMA_CLIP_FILTER = 31
EVENT_MASK_HALF_WIDTH = 0.20
PHASE_BINSIZE_DAYS = 5.0 / 24.0 / 60.0  # 5 min

# Lewis 2013 4.5 um fixed orbital solution
LEWIS = {
    "per": 5.6334729,
    "t0": 2455756.42520,
    "rp": 0.07041,
    "a_rs": 8.28,
    "inc_deg": 84.91,
    "ecc": 0.50829,
    "w_deg": 188.25,
    "u": [0.12, 0.34, 0.20, 0.10],
    "fp": 0.001031,
    "c2": 0.000293,
    "c4": 0.000163,
}

# Fit theta = [dt0, log_fp, c2, c4, baseline]
THETA0 = np.array([
    0.0,
    np.log(LEWIS["fp"]),
    LEWIS["c2"],
    LEWIS["c4"],
    1.0,
])

LOWER = np.array([
    -0.01,          # dt0 (days)
    np.log(5e-4),   # log_fp
    -1.0e-3,        # c2
    -1.0e-3,        # c4
    0.997,          # baseline
])

UPPER = np.array([
    0.01,           # dt0
    np.log(2.5e-3), # log_fp
    1.0e-3,         # c2
    1.0e-3,         # c4
    1.003,          # baseline
])

# Fast MCMC settings (on binned data)
RUN_MCMC = True
N_WALKERS = 20
N_STEPS = 500
BURNIN = 150
THIN = 5
NDIM = 5  # dt0, log_fp, c2, c4, baseline


# ============================================================
# UTILS
# ============================================================

def robust_sigma(y):
    y = np.asarray(y, dtype=float)
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    if not np.isfinite(mad) or mad == 0:
        return np.nanstd(y)
    return 1.4826 * mad

def split_segments(t, gap_days=SEGMENT_GAP_DAYS):
    dt = np.diff(t)
    breaks = np.where(dt > gap_days)[0] + 1
    return np.split(np.arange(len(t)), breaks)

def trim_first_hour_each_segment(t, *arrays, trim_hours=TRIM_HOURS):
    trim_days = trim_hours / 24.0
    segments = split_segments(t)
    keep = np.zeros(len(t), dtype=bool)
    for seg in segments:
        tseg = t[seg]
        keep[seg] = (tseg - tseg[0]) >= trim_days
    out = [t[keep]]
    for arr in arrays:
        out.append(arr[keep])
    return (keep, *out)

def sigma_clip_flux(t, flux, *arrays, nsig=SIGMA_CLIP_NSIG, filt_size=SIGMA_CLIP_FILTER):
    med = median_filter(flux, size=filt_size, mode="nearest")
    resid = flux - med
    sig = robust_sigma(resid)
    if not np.isfinite(sig) or sig == 0:
        keep = np.ones_like(flux, dtype=bool)
    else:
        keep = np.abs(resid) < nsig * sig
    out = [t[keep], flux[keep]]
    for arr in arrays:
        out.append(arr[keep])
    return (keep, *out)

def fill_nan_grid(grid, fill_value=1.0, niter=10):
    g = grid.copy()
    for _ in range(niter):
        if np.isfinite(g).all():
            break
        newg = g.copy()
        for j in range(g.shape[0]):
            for i in range(g.shape[1]):
                if not np.isfinite(g[j, i]):
                    j0 = max(0, j - 1)
                    j1 = min(g.shape[0], j + 2)
                    i0 = max(0, i - 1)
                    i1 = min(g.shape[1], i + 2)
                    patch = g[j0:j1, i0:i1]
                    val = np.nanmedian(patch)
                    if np.isfinite(val):
                        newg[j, i] = val
        g = newg
    g[~np.isfinite(g)] = fill_value
    return g

def build_sensitivity_map(x, y, ratio, nbins=NBINS_MAP, min_pts=MIN_PTS_PER_BIN):
    xbins = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
    ybins = np.linspace(np.nanmin(y), np.nanmax(y), nbins + 1)
    grid = np.full((nbins, nbins), np.nan)
    for i in range(nbins):
        xm = (x >= xbins[i]) & (x < xbins[i + 1])
        for j in range(nbins):
            m = xm & (y >= ybins[j]) & (y < ybins[j + 1])
            if np.sum(m) >= min_pts:
                grid[j, i] = np.nanmedian(ratio[m])
    global_med = np.nanmedian(ratio[np.isfinite(ratio)])
    if not np.isfinite(global_med):
        global_med = 1.0
    grid = fill_nan_grid(grid, fill_value=global_med)
    grid = gaussian_filter(grid, sigma=MAP_SMOOTH_SIGMA)
    return xbins, ybins, grid

def apply_sensitivity_map(x, y, xbins, ybins, grid):
    ix = np.clip(np.searchsorted(xbins, x, side="right") - 1, 0, grid.shape[1] - 1)
    iy = np.clip(np.searchsorted(ybins, y, side="right") - 1, 0, grid.shape[0] - 1)
    return grid[iy, ix]

def bin_data(t, y, binsize_days):
    bins = np.arange(np.nanmin(t), np.nanmax(t) + binsize_days, binsize_days)
    bc, by, be = [], [], []
    for i in range(len(bins) - 1):
        m = (t >= bins[i]) & (t < bins[i + 1])
        if np.sum(m) >= 3:
            yy = y[m]
            bc.append(0.5 * (bins[i] + bins[i + 1]))
            by.append(np.nanmedian(yy))
            be.append(robust_sigma(yy) / np.sqrt(np.sum(m)))
    return np.asarray(bc), np.asarray(by), np.asarray(be)


# ============================================================
# ORBIT HELPERS
# ============================================================

def true_to_eccentric_anomaly(f, e):
    return 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(f / 2.0),
                            np.sqrt(1.0 + e) * np.cos(f / 2.0))

def eccentric_to_true_anomaly(E, e):
    return 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0),
                            np.sqrt(1.0 - e) * np.cos(E / 2.0))

def solve_kepler(M, e, tol=1e-12, maxiter=100):
    M = np.asarray(M)
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(maxiter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E = E + dE
        if np.nanmax(np.abs(dE)) < tol:
            break
    return E

def periapse_time_from_t0(t0, per, e, w_deg):
    w = np.deg2rad(w_deg)
    f_tr = (np.pi / 2.0 - w) % (2.0 * np.pi)
    E_tr = true_to_eccentric_anomaly(f_tr, e)
    M_tr = E_tr - e * np.sin(E_tr)
    return t0 - (M_tr / (2.0 * np.pi)) * per

def secondary_time_from_t0(t0, per, e, w_deg):
    tp = periapse_time_from_t0(t0, per, e, w_deg)
    w = np.deg2rad(w_deg)
    f_sec = (-np.pi / 2.0 - w) % (2.0 * np.pi)
    E_sec = true_to_eccentric_anomaly(f_sec, e)
    M_sec = E_sec - e * np.sin(E_sec)
    return tp + (M_sec / (2.0 * np.pi)) * per

def true_anomaly_from_time(t, t0, per, e, w_deg):
    tp = periapse_time_from_t0(t0, per, e, w_deg)
    M = 2.0 * np.pi * (t - tp) / per
    E = solve_kepler(M, e)
    f = eccentric_to_true_anomaly(E, e)
    return f, tp

def nearest_epoch(base, t_ref, per):
    return base + np.round((t_ref - base) / per) * per


# ============================================================
# MODEL
# ============================================================

def build_primary_model(t, t0):
    p = batman.TransitParams()
    p.t0 = t0
    p.per = LEWIS["per"]
    p.rp = LEWIS["rp"]
    p.a = LEWIS["a_rs"]
    p.inc = LEWIS["inc_deg"]
    p.ecc = LEWIS["ecc"]
    p.w = LEWIS["w_deg"]
    p.limb_dark = "nonlinear"
    p.u = LEWIS["u"]

    m = batman.TransitModel(
        p, t,
        supersample_factor=SUPERSAMPLE,
        exp_time=EXP_TIME_DAYS
    )
    return p, m

def build_secondary_model(t, t0, fp):
    tsec = secondary_time_from_t0(t0, LEWIS["per"], LEWIS["ecc"], LEWIS["w_deg"])

    p = batman.TransitParams()
    p.t0 = t0
    p.t_secondary = tsec
    p.per = LEWIS["per"]
    p.rp = LEWIS["rp"]
    p.a = LEWIS["a_rs"]
    p.inc = LEWIS["inc_deg"]
    p.ecc = LEWIS["ecc"]
    p.w = LEWIS["w_deg"]
    p.limb_dark = "uniform"
    p.u = []
    p.fp = fp

    m = batman.TransitModel(
        p, t,
        transittype="secondary",
        supersample_factor=SUPERSAMPLE,
        exp_time=EXP_TIME_DAYS
    )
    return p, m

def phase_curve_eq12(theta_orbit, fp, c2, c4):
    raw = fp + c2 * (np.sin(theta_orbit) - 1.0) + c4 * np.sin(2.0 * theta_orbit)
    return np.clip(raw, 0.0, None)

def astrophysical_model(t, theta):
    dt0, log_fp, c2, c4, baseline = theta
    t0 = LEWIS["t0"] + dt0
    fp = np.exp(log_fp)

    f_true, tp = true_anomaly_from_time(t, t0, LEWIS["per"], LEWIS["ecc"], LEWIS["w_deg"])
    theta_orbit = f_true + np.deg2rad(LEWIS["w_deg"]) + np.pi

    phase_flux = phase_curve_eq12(theta_orbit, fp, c2, c4)

    p1, m1 = build_primary_model(t, t0)
    primary_flux = m1.light_curve(p1)

    p2, m2 = build_secondary_model(t, t0, fp=1.0)
    sec_full = m2.light_curve(p2)
    vis = np.clip(sec_full - 1.0, 0.0, 1.0)

    return baseline * (primary_flux + phase_flux * vis)

def transit_secondary_ref_model(t, t0):
    p1, m1 = build_primary_model(t, t0)
    primary_flux = m1.light_curve(p1)

    p2, m2 = build_secondary_model(t, t0, fp=1.0)
    sec_full = m2.light_curve(p2)
    vis = np.clip(sec_full - 1.0, 0.0, 1.0)

    fp_ref = LEWIS["fp"]
    planet_flux = fp_ref * vis

    return primary_flux + planet_flux


# ============================================================
# FITTING + MCMC
# ============================================================

def estimate_flux_err(flux):
    sig = robust_sigma(np.diff(flux))
    if not np.isfinite(sig) or sig <= 0:
        sig = robust_sigma(flux - np.nanmedian(flux))
    if not np.isfinite(sig) or sig <= 0:
        sig = 3e-4
    return np.full_like(flux, sig)

def residuals(theta, t, flux, flux_err):
    model = astrophysical_model(t, theta)
    return (flux - model) / flux_err

def fit_five_params(t, flux, flux_err, theta0):
    res = least_squares(
        residuals,
        theta0,
        bounds=(LOWER, UPPER),
        args=(t, flux, flux_err),
        method="trf",
        max_nfev=400
    )
    return res.x, res

def log_prior(theta):
    if np.any(theta < LOWER) or np.any(theta > UPPER):
        return -np.inf
    dt0, log_fp, c2, c4, baseline = theta
    lp = 0.0
    lp += -0.5 * ((baseline - 1.0) / 0.005) ** 2
    return lp

def log_likelihood(theta, t, flux, flux_err):
    model = astrophysical_model(t, theta)
    resid = flux - model
    var = flux_err**2
    return -0.5 * np.sum(resid**2 / var + np.log(2.0 * np.pi * var))

def log_probability(theta, t, flux, flux_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, flux, flux_err)


# ============================================================
# NORMALIZATION / DECORRELATION
# ============================================================

def make_event_mask(t, t0):
    tsec0 = secondary_time_from_t0(t0, LEWIS["per"], LEWIS["ecc"], LEWIS["w_deg"])
    tmid = np.median(t)

    ttr = nearest_epoch(t0, tmid, LEWIS["per"])
    tsec_near = nearest_epoch(tsec0, tmid, LEWIS["per"])
    tsec_prev = tsec_near - LEWIS["per"]
    tsec_next = tsec_near + LEWIS["per"]

    mask = (
        (np.abs(t - ttr) > EVENT_MASK_HALF_WIDTH) &
        (np.abs(t - tsec_prev) > EVENT_MASK_HALF_WIDTH) &
        (np.abs(t - tsec_near) > EVENT_MASK_HALF_WIDTH) &
        (np.abs(t - tsec_next) > EVENT_MASK_HALF_WIDTH)
    )
    return mask, ttr, tsec_prev, tsec_near, tsec_next

def one_pass_correct_and_fit(t, flux_raw, x, y):
    flux_err = estimate_flux_err(flux_raw)

    t0_ref = LEWIS["t0"]
    model_ref = transit_secondary_ref_model(t, t0_ref)
    safe_model_ref = np.where(model_ref <= 0, 1.0, model_ref)

    ratio = flux_raw / safe_model_ref
    xbins, ybins, sens_grid = build_sensitivity_map(x, y, ratio)
    sens = apply_sensitivity_map(x, y, xbins, ybins, sens_grid)

    flux_corr = flux_raw / sens

    oot_mask, *_ = make_event_mask(t, t0_ref)
    if np.sum(oot_mask) > 100:
        flux_corr = flux_corr / np.nanmedian(flux_corr[oot_mask])

    theta_best, fitres = fit_five_params(t, flux_corr, flux_err, THETA0)

    return theta_best, fitres, flux_corr, flux_err, sens, xbins, ybins, sens_grid


# ============================================================
# PLOTTING
# ============================================================

def plot_phase_curve(t, flux_corr, model, theta_best):
    dt0, _, _, _, _ = theta_best
    t0 = LEWIS["t0"] + dt0
    _, tp = true_anomaly_from_time(t, t0, LEWIS["per"], LEWIS["ecc"], LEWIS["w_deg"])
    trel = t - tp

    tb, fb, _ = bin_data(trel, flux_corr, PHASE_BINSIZE_DAYS)
    _, mb, _ = bin_data(trel, model, PHASE_BINSIZE_DAYS)

    n = min(len(tb), len(fb), len(mb))
    tb, fb, mb = tb[:n], fb[:n], mb[:n]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tb, fb, "ko", ms=2.5, label="4.5 μm binned 5 min")
    ax.plot(tb, mb, "r-", lw=1.8, label="Best-fit Eq. 12 model")
    ax.axhline(1.0, color="r", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("Time from periapse (days)")
    ax.set_ylabel("Relative Flux")
    ax.set_title("HAT-P-2b 4.5 μm phase curve")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_PHASE_FIG, dpi=150)
    plt.close(fig)

def plot_zoom_events(t, flux_corr, model, theta_best):
    dt0, _, _, _, _ = theta_best
    t0 = LEWIS["t0"] + dt0
    tsec0 = secondary_time_from_t0(t0, LEWIS["per"], LEWIS["ecc"], LEWIS["w_deg"])

    tmid = 0.5 * (np.min(t) + np.max(t))
    ttr = nearest_epoch(t0, tmid, LEWIS["per"])
    tsec1 = nearest_epoch(tsec0, np.min(t) + 0.2 * (np.max(t) - np.min(t)), LEWIS["per"])
    tsec2 = tsec1 + LEWIS["per"]

    events = [
        (tsec1, "First Secondary Eclipse", 0.15, (0.9986, 1.0022)),
        (ttr,   "Transit",                 0.15, (0.9948, 1.0006)),
        (tsec2, "Second Secondary Eclipse",0.15, (0.9986, 1.0022)),
    ]

    fig = plt.figure(figsize=(10, 11))
    gs = fig.add_gridspec(6, 1, height_ratios=[3, 1, 3, 1, 3, 1], hspace=0.45)

    for i, (tc, label, win, ylim) in enumerate(events):
        ax = fig.add_subplot(gs[2*i, 0])
        axr = fig.add_subplot(gs[2*i+1, 0])

        trel = t - tc
        m = np.abs(trel) <= win
        tw = trel[m]
        fw = flux_corr[m]
        mw = model[m]

        if len(tw) == 0:
            continue

        bw, fwb, _ = bin_data(tw, fw, PHASE_BINSIZE_DAYS)
        _, mwb, _ = bin_data(tw, mw, PHASE_BINSIZE_DAYS)

        n = min(len(bw), len(fwb), len(mwb))
        bw, fwb, mwb = bw[:n], fwb[:n], mwb[:n]

        ax.plot(bw, fwb, "ko", ms=3.0, label="Data 5 min bins")
        ax.plot(bw, mwb, "r-", lw=1.8, label="Best-fit model")
        ax.set_xlim(-win, win)
        ax.set_ylim(*ylim)
        ax.set_ylabel("Rel. Flux")
        ax.set_title(label, fontsize=10, pad=4)
        ax.legend(fontsize=8, loc="upper right")
        if i < len(events) - 1:
            ax.tick_params(labelbottom=False)

        resid = (fwb - mwb) * 1000.0
        axr.axhline(0.0, color="r", lw=0.8, ls="--")
        axr.plot(bw, resid, "k.", ms=3.0)
        axr.set_xlim(-win, win)
        axr.set_ylim(-8, 8)
        axr.set_ylabel("Res.\nmmag", fontsize=8)
        if i == len(events) - 1:
            axr.set_xlabel("Time from event center (days)")
        else:
            axr.tick_params(labelbottom=False)

    fig.suptitle("HAT-P-2b 4.5 μm zoomed events")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUTPUT_ZOOM_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================

def build_summary_df(theta_best, fitres, flux_corr, model):
    dt0, log_fp, c2, c4, baseline = theta_best
    fp = np.exp(log_fp)
    resid = flux_corr - model

    rows = [
        {"parameter": "dt0", "value": dt0},
        {"parameter": "log_fp", "value": log_fp},
        {"parameter": "c2", "value": c2},
        {"parameter": "c4", "value": c4},
        {"parameter": "baseline", "value": baseline},
        {"parameter": "t0", "value": LEWIS["t0"] + dt0},
        {"parameter": "fp", "value": fp},
        {"parameter": "rp", "value": LEWIS["rp"]},
        {"parameter": "a_rs", "value": LEWIS["a_rs"]},
        {"parameter": "inc_deg", "value": LEWIS["inc_deg"]},
        {"parameter": "ecc", "value": LEWIS["ecc"]},
        {"parameter": "w_deg", "value": LEWIS["w_deg"]},
        {"parameter": "ecosw", "value": LEWIS["ecc"] * np.cos(np.deg2rad(LEWIS["w_deg"]))},
        {"parameter": "esinw", "value": LEWIS["ecc"] * np.sin(np.deg2rad(LEWIS["w_deg"]))},
        {"parameter": "least_squares_cost", "value": fitres.cost},
        {"parameter": "fit_success", "value": float(bool(fitres.success))},
        {"parameter": "rms_residual", "value": np.std(resid)},
    ]
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    np.random.seed(RANDOM_SEED)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Could not find input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    t = np.asarray(df["BJD_UTC"], dtype=float)
    flux = np.asarray(df["flux_norm"], dtype=float)
    x = np.asarray(df["x_cent"], dtype=float)
    y = np.asarray(df["y_cent"], dtype=float)
    noisepix = np.asarray(df["noise_pix"], dtype=float)

    s = np.argsort(t)
    t = t[s]
    flux = flux[s]
    x = x[s]
    y = y[s]
    noisepix = noisepix[s]

    print("=" * 70)
    print("LOADING PHOTOMETRY")
    print("=" * 70)
    print(f"Loaded {len(t)} points spanning {(t.max() - t.min()):.4f} days")

    segments = split_segments(t)
    print(f"Found {len(segments)} segments from >1 hr gaps")

    print("=" * 70)
    print("TRIMMING FIRST HOUR OF EACH SEGMENT")
    print("=" * 70)
    _, t, flux, x, y, noisepix = trim_first_hour_each_segment(
        t, flux, x, y, noisepix, trim_hours=TRIM_HOURS
    )
    print(f"Kept {len(t)} points after trimming")

    print("=" * 70)
    print("SIGMA CLIP")
    print("=" * 70)
    _, t, flux, x, y, noisepix = sigma_clip_flux(
        t, flux, x, y, noisepix,
        nsig=SIGMA_CLIP_NSIG,
        filt_size=SIGMA_CLIP_FILTER
    )
    print(f"Kept {len(t)} points after clipping")

    print("=" * 70)
    print("INTRAPIXEL MAP + 5-PARAMETER LSQ FIT")
    print("=" * 70)
    theta_best, fitres, flux_corr, flux_err, sens, xbins, ybins, sens_grid = one_pass_correct_and_fit(
        t, flux, x, y
    )

    model = astrophysical_model(t, theta_best)
    resid = flux_corr - model

    # ----------- prepare binned data for fast MCMC -----------
    t_mcmc, f_mcmc, e_mcmc = bin_data(t, flux_corr, PHASE_BINSIZE_DAYS)
    good = np.isfinite(t_mcmc) & np.isfinite(f_mcmc) & np.isfinite(e_mcmc) & (e_mcmc > 0)
    t_mcmc = t_mcmc[good]
    f_mcmc = f_mcmc[good]
    e_mcmc = e_mcmc[good]

    if RUN_MCMC:
        print("=" * 70)
        print("RUNNING FAST EMCEE ON BINNED DATA")
        print("=" * 70)
        print(f"{len(t_mcmc)} binned points for MCMC")

        initial = theta_best
        pos = initial + 1e-4 * np.random.randn(N_WALKERS, NDIM)

        sampler = emcee.EnsembleSampler(
            N_WALKERS, NDIM, log_probability, args=(t_mcmc, f_mcmc, e_mcmc)
        )
        sampler.run_mcmc(pos, N_STEPS, progress=True)

        chain = sampler.get_chain()
        np.save(OUTPUT_CHAIN_NPY, chain)

        flat_samples = sampler.get_chain(discard=BURNIN, thin=THIN, flat=True)
        theta_mcmc = np.median(flat_samples, axis=0)
        dt0_m, log_fp_m, c2_m, c4_m, baseline_m = theta_mcmc

        print("MCMC median parameters (binned data):")
        print(f"        t0 = {LEWIS['t0'] + dt0_m:.8f}")
        print(f"        fp = {np.exp(log_fp_m):.8f}")
        print(f"        c2 = {c2_m:.8e}")
        print(f"        c4 = {c4_m:.8e}")
        print(f"  baseline = {baseline_m:.8f}")

        theta_best = theta_mcmc
        model = astrophysical_model(t, theta_best)
        resid = flux_corr - model

        if HAVE_CORNER:
            labels = [r"$\Delta t_0$", r"$\log f_p$", r"$c_2$", r"$c_4$", "baseline"]
            fig = corner.corner(flat_samples, labels=labels, show_titles=True)
            fig.savefig(OUTPUT_CORNER_FIG, dpi=150)
            plt.close(fig)
            print(f"Saved {OUTPUT_CORNER_FIG}")

    summary_df = build_summary_df(theta_best, fitres, flux_corr, model)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    model_df = pd.DataFrame({
        "BJD_UTC": t,
        "flux_raw_trimmed": flux,
        "sensitivity_correction": sens,
        "flux_corrected": flux_corr,
        "model_flux": model,
        "residual": resid,
        "x_cent": x,
        "y_cent": y,
        "noise_pix": noisepix,
        "flux_err_fixed": flux_err,
    })
    model_df.to_csv(OUTPUT_MODEL_CSV, index=False)

    plot_phase_curve(t, flux_corr, model, theta_best)
    plot_zoom_events(t, flux_corr, model, theta_best)

    dt0, log_fp, c2, c4, baseline = theta_best

    print("=" * 70)
    print("FINAL 5-PARAMETER RESULTS (LSQ or MCMC median)")
    print(f"        t0 = {LEWIS['t0'] + dt0:.8f}")
    print(f"        fp = {np.exp(log_fp):.8f}")
    print(f"        c2 = {c2:.8e}")
    print(f"        c4 = {c4:.8e}")
    print(f"  baseline = {baseline:.8f}")
    print("=" * 70)
    print(f"Saved {OUTPUT_MODEL_CSV}")
    print(f"Saved {OUTPUT_SUMMARY_CSV}")
    print(f"Saved {OUTPUT_PHASE_FIG}")
    print(f"Saved {OUTPUT_ZOOM_FIG}")
    if RUN_MCMC:
        print(f"Saved {OUTPUT_CHAIN_NPY}")
        if HAVE_CORNER:
            print(f"Saved {OUTPUT_CORNER_FIG}")
    print("=" * 70)
    print(f"RMS residual = {np.std(resid):.6e}")
    print("=" * 70)


if __name__ == "__main__":
    main()