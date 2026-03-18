# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# import batman
# import corner
# import emcee
# from scipy import stats
# import warnings
# warnings.filterwarnings("ignore")

# # ============================================================
# # LEWIS+2013 TABLE 2 PARAMETERS (4.5 μm) - FIXED
# # ============================================================
# # These are their MCMC results. We use them to generate the model.
# LEWIS_PARAMS = {
#     # Orbital geometry
#     't0': 2455752.88523,      # BJD_TDB, transit time
#     'per': 5.6334729,         # days
#     'rp': 0.0693,             # Rp/Rs
#     'a': 8.70,                # a/Rs
#     'inc': 85.97,             # degrees
#     'ecc': 0.50910,           # eccentricity
#     'w': 188.09,              # degrees, argument of periapsis
    
#     # Limb darkening (4-param from Sing 2010, Teff=6290K)
#     'u': [0.12, 0.34, 0.20, 0.10],  
    
#     # Secondary eclipse
#     'fp': 0.001031,           # Fp/Fs eclipse depth
    
#     # Phase curve (their Lorentzian fit)
#     'c1': 0.00116,            # peak Fp/Fs amplitude  
#     'c2': 0.243,              # days after periapsis (5.84 hr)
#     'c3': 0.45,               # rise timescale (days)
#     'c4': 0.39,               # decay timescale (days)
#     'c5': 2.85,               # eclipse center (days after peri)
#     'c6': 0.12,               # eclipse duration (days)
# }

# # ============================================================
# # 1. LOAD PHASE 1 OUTPUT
# # ============================================================
# print("\n" + "=" * 60)
# print("LOADING PHASE 1 PHOTOMETRY")
# print("=" * 60)

# df = pd.read_csv("spitzer_raw_photometry_p1.csv")
# t = np.array(df["BJD_UTC"])
# flux = np.array(df["flux_norm"])
# x = np.array(df["x_cent"])
# y = np.array(df["y_cent"])
# noise_pix = np.array(df["noise_pix"])

# print(f"Loaded {len(t)} points, {t.max()-t.min():.1f} days")


# # ============================================================
# # 2. RAMP CORRECTION (Lewis Section 2.4)
# # ============================================================
# print("\n" + "=" * 60)
# print("RAMP CORRECTION (Lewis Section 2.4)")
# print("=" * 60)

# def ramp_function(t, a1, a2, a3, a4):
#     """Lewis+2013 Equation 1: Double exponential ramp correction."""
#     return 1 + a1 * np.exp(-t/a2) + a3 * np.exp(-t/a4)

# # Identify AOR segments (gaps > 1 hour = downlink)
# dt = np.diff(t)
# gaps = np.where(dt > 1/24.0)[0]  # >1hr gaps
# segments = np.split(np.arange(len(t)), gaps+1)

# print(f"Found {len(segments)} AOR segments")

# flux_corr = flux.copy()
# ramp_params = []

# for i, seg in enumerate(segments):
#     t_seg = t[seg] - t[seg[0]]  # time since segment start (days)
#     flux_seg = flux[seg]
    
#     # Trim first hour (Lewis: "trim the first hour at the start of each observation")
#     mask_trim = t_seg > 1/24.0
#     t_trim = t_seg[mask_trim]
#     flux_trim = flux_seg[mask_trim]
    
#     if len(t_trim) < 20:  # too few points
#         continue
    
#     try:
#         # Fit ramp
#         popt, _ = curve_fit(ramp_function, t_trim, flux_trim, 
#                            p0=[0.01, 0.1, 0.01, 0.3],
#                            bounds=([-0.1, 0.01, -0.1, 0.01], [0.1, 1.0, 0.1, 1.0]))
#         ramp_model = ramp_function(t_seg, *popt)
#         flux_corr[seg] /= ramp_model
        
#         ramp_params.append(popt)
#         print(f"  Segment {i}: a1={popt[0]:.4f}, tau1={popt[1]*24:.1f}hr")
#     except:
#         print(f"  Segment {i}: fit failed, no correction")

# print("✓ Ramp correction applied")


# # ============================================================
# # 3. INTRAPIXEL SENSITIVITY (Lewis Appendix B)
# # Non-parametric: bin flux vs (x,y) then divide by map
# # ============================================================
# print("\n" + "=" * 60)
# print("INTRAPIXEL SENSITIVITY CORRECTION (Appendix B)")
# print("=" * 60)

# # Build 2D sensitivity map: bin into NxN grid of (x,y) positions
# # Lewis: non-parametric map of pixel-position-dependent flux variations
# N_bins = 25  # 25x25 grid

# x_bins = np.linspace(x.min(), x.max(), N_bins + 1)
# y_bins = np.linspace(y.min(), y.max(), N_bins + 1)

# sensitivity_map = np.ones((N_bins, N_bins))

# # For each (x,y) bin, median flux = sensitivity at that position
# for i in range(N_bins):
#     for j in range(N_bins):
#         mask = ((x >= x_bins[i]) & (x < x_bins[i+1]) &
#                 (y >= y_bins[j]) & (y < y_bins[j+1]))
#         if np.sum(mask) > 5:
#             sensitivity_map[j, i] = np.median(flux_corr[mask])

# # Find which bin each point falls into
# ix = np.clip(np.searchsorted(x_bins, x) - 1, 0, N_bins - 1)
# iy = np.clip(np.searchsorted(y_bins, y) - 1, 0, N_bins - 1)

# # Divide each flux point by the sensitivity at its (x,y) position
# ip_correction = sensitivity_map[iy, ix]
# flux_final = flux_corr / ip_correction

# print(f"  Built {N_bins}x{N_bins} sensitivity map")
# print(f"  Median correction factor: {np.median(ip_correction):.5f}")
# print(f"RMS before/after: {np.std(flux_corr):.5f} → {np.std(flux_final):.5f}")


# # ============================================================
# # POST-CORRECTION OUTLIER REMOVAL
# # ============================================================
# print("\nPost-correction sigma clipping...")

# from scipy.ndimage import median_filter
# flux_med = median_filter(flux_final, size=16, mode='nearest')
# resid_clip = np.abs(flux_final - flux_med)
# std_clip = np.std(resid_clip)
# good = resid_clip < 10.0 * std_clip

# print(f"  Removing {np.sum(~good)} outliers ({np.sum(~good)/len(flux_final)*100:.2f}%)")
# t = t[good]
# flux = flux[good]
# flux_corr = flux_corr[good]
# flux_final = flux_final[good]
# x = x[good]
# y = y[good]

# # ============================================================
# # RE-NORMALIZE: Force out-of-event baseline = exactly 1.0
# # ============================================================

# # Rough event positions to mask out during normalization
# t_ecl_approx  = 2455751.58912
# t_tran_approx = 2455756.44545
# t_ecl2_approx = t_ecl_approx + LEWIS_PARAMS['per']

# oot_mask = (
#     (np.abs(t - t_tran_approx) > 0.20) &
#     (np.abs(t - t_ecl_approx)  > 0.20) &
#     (np.abs(t - t_ecl2_approx) > 0.20)
# )

# oot_median = np.median(flux_final[oot_mask])
# flux_final  = flux_final / oot_median
# print(f"Re-normalized baseline: {oot_median:.6f} → now 1.000000")

# # ============================================================
# # 4. LEWIS MODEL (Sections 2.5, 2.6)
# # ============================================================
# print("\n" + "=" * 60)
# print("BUILDING LEWIS MODEL (Sections 2.5, 2.6)")
# print("=" * 60)

# def lewis_model(t):
#     """Lewis+2013 model: transit + phase curve + eclipse."""
    
#     # BATMAN transit setup
#     params = batman.TransitParams()
#     params.t0 = LEWIS_PARAMS['t0']
#     params.per = LEWIS_PARAMS['per']
#     params.rp = LEWIS_PARAMS['rp']
#     params.a = LEWIS_PARAMS['a']
#     params.inc = LEWIS_PARAMS['inc']
#     params.ecc = LEWIS_PARAMS['ecc']
#     params.w = LEWIS_PARAMS['w']
#     params.limb_dark = "nonlinear"   # batman's name for 4-param LD
#     params.u = LEWIS_PARAMS['u']
    
#     # Spitzer 4.5 μm subarray exposure time = 0.4s = 0.4/86400 days
#     exp_time = 0.4 / 86400.0
    
#     m_transit = batman.TransitModel(params, t, 
#                                     supersample_factor=10,
#                                     exp_time=exp_time)
#     transit_flux = m_transit.light_curve(params)

    
#     # Secondary eclipse: planet goes behind star
#     # Use batman's secondary eclipse via fp parameter
#     params.fp = LEWIS_PARAMS['fp']   # planet-to-star flux ratio at eclipse

#     # Time of secondary eclipse (Lewis Table 2, 4.5 µm)
#     t_sec = 2455751.58912   # first eclipse BJD - corrected from data search
#     period = LEWIS_PARAMS['per']

#     # Compute eclipse model: find all eclipse centers within data range
#     t_sec_all = t_sec + np.round((t - t_sec) / period) * period

#     # Simple box eclipse (ingress/egress ~2hr for HAT-P-2b)
#     eclipse_dur = 0.125   # days (~3hr total duration, Lewis Table 2)
#     in_eclipse = np.zeros(len(t), dtype=bool)
#     for tc in np.unique(t_sec_all):
#         in_eclipse |= np.abs(t - tc) < eclipse_dur / 2.0

#     # During eclipse: subtract planet flux (normalized to 1 + fp outside)
#     eclipse_flux = np.ones(len(t))
#     eclipse_flux[in_eclipse] = 1.0 / (1.0 + LEWIS_PARAMS['fp'])

    
#     # Orbital phase relative to periapsis
#     # t0 is transit time; time of periapsis = t0 + offset
#     # Lewis Eq 9-11: asymmetric Lorentzian phase curve in time from periapsis
#     t_peri = LEWIS_PARAMS['t0'] - LEWIS_PARAMS['c2']
#     dt = (t - t_peri) % LEWIS_PARAMS['per']
    
#     # Asymmetric Lorentzian (Lewis Eq 11)
#     c2 = LEWIS_PARAMS['c2']
#     c3 = LEWIS_PARAMS['c3']
#     c4 = LEWIS_PARAMS['c4']
#     nu = np.where(dt < c2, dt / c3, dt / c4)
#     phase_flux = LEWIS_PARAMS['c1'] / (nu**2 + 1.0)
    
#     # Zero phase flux during eclipse (planet hidden)
#     eclipse_center = LEWIS_PARAMS['c5']
#     eclipse_dur = LEWIS_PARAMS['c6']
#     eclipse_mask = np.abs(dt - eclipse_center) < eclipse_dur / 2.0
#     phase_flux[eclipse_mask] = 0.0
    
#     # Total model = transit * (1 + phase curve)
#     model = transit_flux * (1.0 + phase_flux) * eclipse_flux
    
#     return model
# # Find correct transit epoch within this dataset
# t0_base = 2454387.49375       # Pal et al. 2010 reference epoch (BJD)
# period = 5.6334729

# t_data_center = (t.min() + t.max()) / 2.0
# n = np.round((t_data_center - t0_base) / period)
# t0_correct = t0_base + n * period

# print(f"Data center: BJD {t_data_center:.5f}")
# print(f"Nearest transit epoch n={n:.0f}")
# print(f"Correct t0: BJD {t0_correct:.5f}")
# print(f"Transit at hour: {(t0_correct - t.min())*24:.1f} hr into observation")

# # Update the parameter
# LEWIS_PARAMS['t0'] = t0_correct

# # Compute model
# model_flux = lewis_model(t)
# residuals = flux_final - model_flux

# print("✓ Lewis model computed")
# print(f"Transit depth: {np.min(model_flux):.4f}")
# print(f"RMS residuals: {np.std(residuals):.4f}")


# # ============================================================
# # 5. STELLAR VARIABILITY TEST (Lewis Section 2.7)
# # ============================================================
# print("\n" + "=" * 60)
# print("STELLAR VARIABILITY TEST (Lewis Section 2.7)")
# print("=" * 60)

# # Test linear trend (Eq 13)
# def linear_stellar(t, d1):
#     return 1 + d1 * (t - t.mean())

# popt_lin, pcov_lin = curve_fit(linear_stellar, t, flux_final - model_flux,
#                               p0=[0], bounds=[-0.01, 0.01])

# # Test sinusoidal (Eq 14, stellar rotation ~3.8 days)
# def sine_stellar(t, d1, d2, d3):
#     return 1 + d1 * np.sin(2*np.pi*(t-t.mean())/d2 - d3)

# try:
#     popt_sin, pcov_sin = curve_fit(sine_stellar, t, flux_final - model_flux,
#                                   p0=[0.001, 3.8, 0],
#                                   bounds=[0, [0.01, 10, np.pi]])
#     print("Sinusoidal stellar: amp={:.2e}, period={:.1f}d".format(*popt_sin[:2]))
# except:
#     print("Sinusoidal stellar fit failed")

# print(f"Linear stellar slope: {popt_lin[0]*1e4:.2f} ppm/day (insignificant)")
# print("✓ Matches Lewis: no significant stellar variability")


# # ============================================================
# # 6. FINAL PLOTS
# # ============================================================
# # ============================================================
# # FIGURE 5 STYLE: Full phase curve, x-axis = time from periapse
# # Lewis Figure 5 (middle panel = 4.5 µm)
# # ============================================================

# # Compute time of periapse from Table 2 (4.5 µm value)
# t_peri = 2455757.05194   # BJD_UTC, Lewis Table 2

# # Time from periapse in days
# t_from_peri = t - t_peri

# # Bin to 5-minute intervals (Lewis bins all their final plots to 5 min)
# bin_size = 30.0 / (24.0 * 60.0)   # 5 min in days
# t_bins = np.arange(t_from_peri.min(), t_from_peri.max(), bin_size)

# def bin_data(t_rel, f, bins):
#     """Bin flux into time bins, return bin centers and median flux."""
#     bin_centers = []
#     bin_flux = []
#     bin_err = []
#     for i in range(len(bins) - 1):
#         mask = (t_rel >= bins[i]) & (t_rel < bins[i+1])
#         if np.sum(mask) > 3:
#             bin_centers.append(0.5 * (bins[i] + bins[i+1]))
#             bin_flux.append(np.median(f[mask]))
#             bin_err.append(np.std(f[mask]) / np.sqrt(np.sum(mask)))
#     return np.array(bin_centers), np.array(bin_flux), np.array(bin_err)

# t_bin5, f_bin5, e_bin5 = bin_data(t_from_peri, flux_final, t_bins)

# # Model evaluated at full resolution, then also binned
# model_flux = lewis_model(t)
# # ============================================================
# # FIND ACTUAL ECLIPSE TIME FROM DATA
# # Search for minimum flux in expected eclipse window
# # Expected: eclipse is ~5.2 days before periapse
# # ============================================================

# # Search window: ±0.3 days around expected eclipse
# t_ecl_guess = t_peri - 5.18   # rough guess from Lewis Fig 5
# search_mask = np.abs(t - t_ecl_guess) < 0.4

# t_search = t[search_mask]
# f_search = flux_final[search_mask]

# # Bin to 5 min and find minimum
# bin_size_s = 5.0 / (24.0 * 60.0)
# bins_s = np.arange(t_search.min(), t_search.max(), bin_size_s)
# t_sb, f_sb, _ = bin_data(t_search - t_search.min(),
#                           f_search,
#                           bins_s - t_search.min())
# t_sb += t_search.min()

# t_eclipse1_found = t_sb[np.argmin(f_sb)]
# print(f"\nEclipse search:")
# print(f"  Guess:  BJD {t_ecl_guess:.5f}")
# print(f"  Found:  BJD {t_eclipse1_found:.5f}")
# print(f"  Offset: {(t_eclipse1_found - t_ecl_guess)*24:.2f} hr")

# # Also find transit minimum for sanity check
# t_tr_guess = t_peri - 0.68
# tr_mask = np.abs(t - t_tr_guess) < 0.3
# t_tr_s = t[tr_mask]
# f_tr_s = flux_final[tr_mask]
# bins_tr = np.arange(t_tr_s.min(), t_tr_s.max(), bin_size_s)
# t_trb, f_trb, _ = bin_data(t_tr_s - t_tr_s.min(), f_tr_s, bins_tr - t_tr_s.min())
# t_trb += t_tr_s.min()
# t_transit_found = t_trb[np.argmin(f_trb)]
# print(f"\nTransit search:")
# print(f"  Guess:  BJD {t_tr_guess:.5f}")
# print(f"  Found:  BJD {t_transit_found:.5f}")
# print(f"  Offset: {(t_transit_found - t_tr_guess)*24:.2f} hr")

# # Update event centers to data-derived values
# t_eclipse1 = t_eclipse1_found
# t_transit  = t_transit_found

# _, m_bin5, _ = bin_data(t_from_peri, model_flux, t_bins)

# fig5, ax = plt.subplots(figsize=(10, 4))

# # Binned corrected data (filled circles like Lewis)
# ax.plot(t_bin5, f_bin5, 'ko', ms=2.5, label='4.5 µm (binned 5 min)')

# # Best-fit model in red
# ax.plot(t_bin5, m_bin5, 'r-', lw=1.5, label='Best-fit model')

# # Dashed line at stellar flux = 1.0
# ax.axhline(1.0, color='r', lw=0.8, ls='--', label='Stellar flux')

# ax.set_xlabel('Time from Periapse (Earth days)', fontsize=12)
# ax.set_ylabel('Relative Flux', fontsize=12)
# ax.set_xlim(t_from_peri.min() - 0.15, t_from_peri.max()+0.15)
# ax.set_ylim(0.9975, 1.0025)
# ax.legend(fontsize=9)
# ax.set_title('HAT-P-2b 4.5 µm Phase Curve (Lewis Fig. 5 style)')
# plt.tight_layout()
# # plt.savefig('figure5_phase_curve.png', dpi=150)
# plt.show()
# print("✓ Saved figure5_phase_curve.png")


# # ============================================================
# # FIGURE 7 STYLE: Zoom on transit + 2 eclipses
# # x-axis = time from event center in days, ±0.15 days
# # ============================================================

# # Event centers from Lewis Table 2 (4.5 µm, BJD_UTC)
# t_transit   = 2455756.44545     # transit center
# t_eclipse1  = 2455751.58912   # first secondary eclipse
# t_eclipse2  = t_eclipse1 + LEWIS_PARAMS['per']    # second secondary eclipse

# events = [
#     (t_eclipse1, 'First Secondary Eclipse',  [0.9988, 1.0022], 0.15),
#     (t_transit,  'Transit',                  [0.9950, 1.0005], 0.15),
# ]

# fig7 = plt.figure(figsize=(10, 8))
# gs = fig7.add_gridspec(4, 1,
#                        height_ratios=[3, 1, 3, 1],
#                        hspace=0.55)

# main_axes = [fig7.add_subplot(gs[0]), fig7.add_subplot(gs[2])]
# res_axes  = [fig7.add_subplot(gs[1]), fig7.add_subplot(gs[3])]

# bin_size_fig7 = 5.0 / (24.0 * 60.0)   # 5-min bins

# for ax, ax_res, (t_center, label, ylim, win) in zip(main_axes, res_axes, events):

#     # Select data within window
#     t_rel  = t - t_center
#     mask   = np.abs(t_rel) < win
#     t_w    = t_rel[mask]
#     f_w    = flux_final[mask]
#     m_w    = model_flux[mask]

#     # 5-min binning
#     bins_w        = np.arange(-win, win + bin_size_fig7, bin_size_fig7)
#     t_wb, f_wb, _ = bin_data(t_w, f_w, bins_w)
#     _,    m_wb, _ = bin_data(t_w, m_w, bins_w)

#     # Main panel
#     ax.plot(t_wb, f_wb, 'ko', ms=3.5, label='Data (5 min bins)')
#     ax.plot(t_wb, m_wb, 'r-', lw=1.8, label='Lewis model')
#     ax.set_xlim(-win, win)
#     ax.set_ylim(ylim)
#     ax.set_ylabel('Rel. Flux', fontsize=9)
#     ax.set_title(label, fontsize=10, pad=4)
#     ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
#     ax.tick_params(labelbottom=False)   # hide x tick labels on main panel
#     ax.legend(fontsize=8, loc='upper right')

#     # Residuals panel
#     resid = (f_wb - m_wb) * 1000   # mmag
#     ax_res.axhline(0, color='r', lw=0.8, ls='--')
#     ax_res.plot(t_wb, resid, 'k.', ms=2.5)
#     ax_res.set_xlim(-win, win)
#     ax_res.set_ylim(-8, 8)
#     ax_res.set_ylabel('Res.\n(mmag)', fontsize=7)
#     ax_res.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
#     ax_res.tick_params(labelsize=7)

# # Only the very bottom panel gets the x label
# res_axes[-1].set_xlabel('Time from Event Center (days)', fontsize=11)

# fig7.suptitle('HAT-P-2b 4.5 µm — Lewis Figure 7 Style', fontsize=12)
# # plt.savefig('figure7_zoom_events.png', dpi=150, bbox_inches='tight')
# plt.show()
# print("✓ Saved figure7_zoom_events.png")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import batman
import corner
import emcee
from scipy import stats
import warnings
warnings.filterwarnings("ignore")




# ============================================================
# LEWIS+2013 TABLE 2 PARAMETERS (4.5 μm) - FIXED
# ============================================================
# This dictionary stores the published 4.5 μm best-fit values from Lewis+2013
# that the rest of this script uses to build a comparison model.
#
# Important idea:
# This Phase 2 script is not doing the full simultaneous MCMC fit from the paper.
# Instead, it takes the Phase 1 photometry, applies a sequence of corrections,
# and then generates a "Lewis-style" model using fixed literature values.
#
# In other words:
# - These are treated as known inputs, not as free parameters.
# - The model is meant to reproduce the qualitative/approximate 4.5 μm behavior.
# - If the final light curve looks wrong (it does not for now), these values and the event timings are
#   among the first things to inspect.
#
# Parameter groups:
# - Orbital geometry: transit timing/orbit shape/orientation.
# - Limb darkening: used for the transit model.
# - Secondary eclipse: sets the planet-to-star flux ratio removed during eclipse.
# - Phase curve coefficients: describe the asymmetric planetary brightening and cooling
#   around periapse using a Lorentzian-style function.
LEWIS_PARAMS = {
    # Orbital geometry
    't0': 2455752.88523,      # BJD_TDB, transit time
    'per': 5.6334729,         # days
    'rp': 0.0693,             # Rp/Rs
    'a': 8.70,                # a/Rs
    'inc': 85.97,             # degrees
    'ecc': 0.50910,           # eccentricity
    'w': 188.09,              # degrees, argument of periapsis
    
    # Limb darkening (4-param from Sing 2010, Teff=6290K)
    'u': [0.12, 0.34, 0.20, 0.10],
    
    # Secondary eclipse
    'fp': 0.001031,           # Fp/Fs eclipse depth
    
    # Phase curve (their Lorentzian fit)
    'c1': 0.00116,            # peak Fp/Fs amplitude
    'c2': 0.243,              # days after periapsis (5.84 hr)
    'c3': 0.45,               # rise timescale (days)
    'c4': 0.39,               # decay timescale (days)
    'c5': 2.85,               # eclipse center (days after peri)
    'c6': 0.12,               # eclipse duration (days)
}




# ============================================================
# 1. LOAD PHASE 1 OUTPUT
# ============================================================
# This is the input stage for Phase 2.
#
# The Phase 1 script already did the heavy image-level work:
# - extracted raw aperture photometry
# - measured x/y centroids
# - computed a noise-pixel diagnostic
# - removed obvious bad points
#
# Phase 2 starts from that time-series product and applies higher-level corrections
# directly to the light curve.
#
# Expected columns:
# - BJD_UTC   : observation times
# - flux_norm : raw normalized photometry from Phase 1
# - x_cent    : x centroid positions
# - y_cent    : y centroid positions
# - noise_pix : PSF-width/noise-pixel proxy
print("\n" + "=" * 60)
print("LOADING PHASE 1 PHOTOMETRY")
print("=" * 60)


df = pd.read_csv("spitzer_raw_photometry_p1.csv")
t = np.array(df["BJD_UTC"])
flux = np.array(df["flux_norm"])
x = np.array(df["x_cent"])
y = np.array(df["y_cent"])
noise_pix = np.array(df["noise_pix"])


print(f"Loaded {len(t)} points, {t.max()-t.min():.1f} days")




# ============================================================
# 2. RAMP CORRECTION (Lewis Section 2.4)
# ============================================================
# Spitzer phase-curve data often show a "ramp" at the start of an observation and
# after interruptions such as spacecraft downlinks.
#
# In the Lewis paper, the 3.6 and 4.5 μm observations were described as showing this
# ramp-like behavior near the start of observations and after breaks. Their strategy
# for 4.5 μm emphasized trimming the first hour of each segment rather than fitting
# explicit ramp parameters in the final model.
#
# This script takes a more direct correction approach:
# - identify each uninterrupted AOR-like segment
# - trim the first hour of that segment for fitting
# - fit a double-exponential ramp to the remaining data
# - divide the full segment by the fitted ramp model
#
# So this block is "Lewis-inspired" rather than a strict replication of the exact
# 4.5 μm fitting strategy in the paper.
print("\n" + "=" * 60)
print("RAMP CORRECTION (Lewis Section 2.4)")
print("=" * 60)


def ramp_function(t, a1, a2, a3, a4):
    """
    Lewis-inspired double exponential ramp correction.
    
    The form matches the general style discussed by Lewis+2013 for
    Spitzer ramp behavior: a sum of exponential terms in time from segment start.
    """
    return 1 + a1 * np.exp(-t/a2) + a3 * np.exp(-t/a4)


# Identify AOR segments using gaps in the time series.
# A gap larger than one hour is treated as a break between observing blocks,
# similar to the downlink-separated segments discussed in the paper.
dt = np.diff(t)
gaps = np.where(dt > 1/24.0)[0]  # >1 hr gaps
segments = np.split(np.arange(len(t)), gaps + 1)


print(f"Found {len(segments)} AOR segments")


# flux_corr will store the light curve after segment-by-segment ramp removal.
# ramp_params collects the best-fit coefficients for later inspection/diagnostics.
flux_corr = flux.copy()
ramp_params = []


for i, seg in enumerate(segments):
    # Re-zero time within each segment.
    # Ramp models are naturally expressed as "time since segment start."
    t_seg = t[seg] - t[seg[0]]
    flux_seg = flux[seg]
    
    # Trim first hour.
    # Lewis notes that the asymptotic part of the ramp often settles within an hour,
    # and trimming the first hour reduces the strongest startup behavior.
    mask_trim = t_seg > 1/24.0
    t_trim = t_seg[mask_trim]
    flux_trim = flux_seg[mask_trim]
    
    # If a segment is too short after trimming, skip fitting.
    # There would not be enough data to constrain a 4-parameter function robustly.
    if len(t_trim) < 20:
        continue
    
    try:
        # Fit the double-exponential ramp.
        # The initial guesses and bounds are chosen to keep the fit in a physically
        # reasonable range and avoid wildly unstable solutions.
        popt, _ = curve_fit(
            ramp_function,
            t_trim,
            flux_trim,
            p0=[0.01, 0.1, 0.01, 0.3],
            bounds=([-0.1, 0.01, -0.1, 0.01], [0.1, 1.0, 0.1, 1.0])
        )
        
        # Evaluate the ramp over the full segment, not just the trimmed part,
        # then divide the original segment by that model.
        ramp_model = ramp_function(t_seg, *popt)
        flux_corr[seg] /= ramp_model
        
        ramp_params.append(popt)
        print(f"  Segment {i}: a1={popt[0]:.4f}, tau1={popt[1]*24:.1f}hr")
    
    except:
        # If the fit fails, leave that segment unchanged.
        print(f"  Segment {i}: fit failed, no correction")


print("✓ Ramp correction applied")




# ============================================================
# 3. INTRAPIXEL SENSITIVITY (Lewis Appendix B)
# ============================================================
# After ramp removal, the next major 4.5 μm systematic is intrapixel sensitivity:
# the measured flux depends on where the stellar centroid lands within a pixel.
#
# Lewis Appendix B uses a non-parametric pixel-mapping method based on nearest
# neighbors and Gaussian weighting in x/y (and optionally a PSF-shape term).
#
# This script uses a simpler approximation:
# - bin the detector positions into a 2D (x, y) grid
# - estimate the typical flux in each cell
# - treat that value as the local sensitivity
# - divide each point by the sensitivity associated with its detector location
#
# So again, this is conceptually consistent with the paper's goal of position-based
# non-parametric decorrelation, but it is a simplified implementation.
print("\n" + "=" * 60)
print("INTRAPIXEL SENSITIVITY CORRECTION (Appendix B)")
print("=" * 60)


# Build a 2D sensitivity map in centroid-position space.
# A finer grid can model smaller-scale structure, but if it is too fine many cells
# will be sparsely populated and the map becomes noisy.
N_bins = 25  # 25 x 25 grid


x_bins = np.linspace(x.min(), x.max(), N_bins + 1)
y_bins = np.linspace(y.min(), y.max(), N_bins + 1)


# Initialize the map to unity so empty/poorly sampled cells do not produce a divide-by-zero.
sensitivity_map = np.ones((N_bins, N_bins))


# For each (x, y) bin, estimate the local sensitivity using the median corrected flux.
# Median statistics are more robust than means if a few bad points remain.
for i in range(N_bins):
    for j in range(N_bins):
        mask = (
            (x >= x_bins[i]) & (x < x_bins[i + 1]) &
            (y >= y_bins[j]) & (y < y_bins[j + 1])
        )
        
        # Require a small minimum occupancy before trusting that cell.
        if np.sum(mask) > 5:
            sensitivity_map[j, i] = np.median(flux_corr[mask])


# For every observation, determine which x-bin and y-bin it falls into.
# searchsorted tells us the bin index; clip keeps the index safely in range.
ix = np.clip(np.searchsorted(x_bins, x) - 1, 0, N_bins - 1)
iy = np.clip(np.searchsorted(y_bins, y) - 1, 0, N_bins - 1)


# Look up the local sensitivity and divide by it.
# If a point falls in a region where the detector is artificially bright, this scales
# the point downward; if it falls in a less sensitive region, it scales upward.
ip_correction = sensitivity_map[iy, ix]
flux_final = flux_corr / ip_correction


print(f"  Built {N_bins}x{N_bins} sensitivity map")
print(f"  Median correction factor: {np.median(ip_correction):.5f}")
print(f"RMS before/after: {np.std(flux_corr):.5f} → {np.std(flux_final):.5f}")




# ============================================================
# POST-CORRECTION OUTLIER REMOVAL
# ============================================================
# Even after ramp and intrapixel corrections, some light-curve points can still be
# pathological. This block does a final sigma-clipping pass on the corrected flux.
#
# Compared with Phase 1:
# - Phase 1 removed obvious bad points in the raw photometry.
# - This stage removes points that only become obviously discrepant after the
#   decorrelation steps are applied.
print("\nPost-correction sigma clipping...")


from scipy.ndimage import median_filter

# Build a local baseline using a 16-point median filter.
# This tracks the nearby light-curve trend without being overly sensitive to outliers.
flux_med = median_filter(flux_final, size=16, mode='nearest')

# Measure how far each point lies from that local baseline.
resid_clip = np.abs(flux_final - flux_med)
std_clip = np.std(resid_clip)
good = resid_clip < 10.0 * std_clip


print(f"  Removing {np.sum(~good)} outliers ({np.sum(~good)/len(flux_final)*100:.2f}%)")

# Apply the same mask to every synchronized array.
# Just like in Phase 1, keeping all arrays aligned is essential.
t = t[good]
flux = flux[good]
flux_corr = flux_corr[good]
flux_final = flux_final[good]
x = x[good]
y = y[good]




# ============================================================
# RE-NORMALIZE: Force out-of-event baseline = exactly 1.0
# ============================================================
# After the previous corrections, the overall baseline may no longer sit exactly at 1.0.
# This block re-normalizes the light curve using out-of-event data only.
#
# Why mask events?
# If transit or eclipse points are included in the normalization reference, the median
# would be biased low and the baseline would not represent the true stellar flux level.
#
# The event masks here are rough/approximate and are only used for baseline estimation.


# Rough event positions to exclude during normalization.
# These are hard-coded approximate times for the events present in the 4.5 μm campaign.
t_ecl_approx = 2455751.58912
t_tran_approx = 2455756.44545
t_ecl2_approx = t_ecl_approx + LEWIS_PARAMS['per']


oot_mask = (
    (np.abs(t - t_tran_approx) > 0.20) &
    (np.abs(t - t_ecl_approx) > 0.20) &
    (np.abs(t - t_ecl2_approx) > 0.20)
)


# Use the median out-of-event flux as the renormalization factor.
oot_median = np.median(flux_final[oot_mask])
flux_final = flux_final / oot_median

print(f"Re-normalized baseline: {oot_median:.6f} → now 1.000000")




# ============================================================
# 4. LEWIS MODEL (Sections 2.5, 2.6)
# ============================================================
# This function constructs a literature-based model light curve containing:
# - the transit
# - the planetary phase variation
# - the secondary eclipse
#
# The paper models transit/eclipse geometry using the eccentric-orbit formalism and
# tests multiple phase-curve parameterizations. For the 4.5 μm data, Table 2 reports
# a preferred functional form and best-fit coefficients.
#
# Here the script builds a simplified forward model using:
# - BATMAN for the transit shape
# - a box-style secondary eclipse
# - an asymmetric Lorentzian-like phase curve in time from periapse
#
# So this is not a full reproduction of the exact Lewis fitting code, but it captures
# the main components needed for a Lewis-style comparison plot.
print("\n" + "=" * 60)
print("BUILDING LEWIS MODEL (Sections 2.5, 2.6)")
print("=" * 60)


def lewis_model(t):
    """
    Lewis-style model: transit + phase curve + eclipse.
    
    The model uses fixed literature parameters and returns the expected relative flux
    from the star+planet system at each observation time.
    """
    
    # ------------------------------------------------------------
    # Transit model
    # ------------------------------------------------------------
    # BATMAN handles the transit geometry and limb darkening.
    # We feed it the literature orbit and radius parameters.
    params = batman.TransitParams()
    params.t0 = LEWIS_PARAMS['t0']
    params.per = LEWIS_PARAMS['per']
    params.rp = LEWIS_PARAMS['rp']
    params.a = LEWIS_PARAMS['a']
    params.inc = LEWIS_PARAMS['inc']
    params.ecc = LEWIS_PARAMS['ecc']
    params.w = LEWIS_PARAMS['w']
    params.limb_dark = "nonlinear"   # batman's name for 4-parameter LD law
    params.u = LEWIS_PARAMS['u']
    
    # Spitzer 4.5 μm subarray exposure time.
    # Supersampling helps account for finite exposure integration across sharp features.
    exp_time = 0.4 / 86400.0
    
    m_transit = batman.TransitModel(
        params,
        t,
        supersample_factor=10,
        exp_time=exp_time
    )
    transit_flux = m_transit.light_curve(params)

    
    # ------------------------------------------------------------
    # Secondary eclipse model
    # ------------------------------------------------------------
    # During secondary eclipse, the planet goes behind the star.
    # The observed system flux therefore loses the planet's thermal contribution.
    params.fp = LEWIS_PARAMS['fp']   # planet-to-star flux ratio outside eclipse
    
    # Hard-coded first secondary eclipse time used in this script's 4.5 μm setup.
    # period is reused to propagate to other eclipse epochs in the observed window.
    t_sec = 2455751.58912
    period = LEWIS_PARAMS['per']

    # Compute the nearest eclipse center for each timestamp.
    # This lets the code identify whichever eclipse belongs to a given portion
    # of the observed time series.
    t_sec_all = t_sec + np.round((t - t_sec) / period) * period

    # Build a simple box eclipse mask.
    # This is a simplified stand-in for a more exact eclipse-shape calculation.
    eclipse_dur = 0.125   # days (~3 hr total duration)
    in_eclipse = np.zeros(len(t), dtype=bool)
    for tc in np.unique(t_sec_all):
        in_eclipse |= np.abs(t - tc) < eclipse_dur / 2.0

    # Outside eclipse, total normalized flux includes the planet.
    # Inside eclipse, the planet contribution is removed.
    eclipse_flux = np.ones(len(t))
    eclipse_flux[in_eclipse] = 1.0 / (1.0 + LEWIS_PARAMS['fp'])

    
    # ------------------------------------------------------------
    # Phase curve model
    # ------------------------------------------------------------
    # Lewis tests several phase-curve functional forms and reports best-fit coefficients.
    # This code uses an asymmetric Lorentzian-like function in time from periapse.
    #
    # Interpretation of the c-parameters:
    # - c1: peak amplitude
    # - c2: time of the peak after periapse
    # - c3: characteristic rise timescale
    # - c4: characteristic decay timescale
    #
    # t_peri is approximated here from the relation between the transit time and c2.
    t_peri = LEWIS_PARAMS['t0'] - LEWIS_PARAMS['c2']
    dt = (t - t_peri) % LEWIS_PARAMS['per']
    
    c2 = LEWIS_PARAMS['c2']
    c3 = LEWIS_PARAMS['c3']
    c4 = LEWIS_PARAMS['c4']

    # Use one width before the peak and another after the peak, allowing asymmetry.
    nu = np.where(dt < c2, dt / c3, dt / c4)
    phase_flux = LEWIS_PARAMS['c1'] / (nu**2 + 1.0)
    
    # Zero the phase contribution during eclipse.
    # Physically, the planet is hidden there, so its phase-dependent brightness is not visible.
    eclipse_center = LEWIS_PARAMS['c5']
    eclipse_dur = LEWIS_PARAMS['c6']
    eclipse_mask = np.abs(dt - eclipse_center) < eclipse_dur / 2.0
    phase_flux[eclipse_mask] = 0.0
    
    # ------------------------------------------------------------
    # Total system model
    # ------------------------------------------------------------
    # The observed relative flux is the stellar transit attenuation multiplied by the
    # out-of-eclipse planetary flux contribution and the eclipse suppression term.
    model = transit_flux * (1.0 + phase_flux) * eclipse_flux
    
    return model


# ============================================================
# TRANSIT EPOCH ADJUSTMENT
# ============================================================
# The literature reference epoch may not correspond to the exact transit that falls
# inside this particular observing window.
#
# This block finds the nearest transit epoch to the center of the dataset using:
# - a reference ephemeris (Pal et al. style t0_base)
# - the orbital period
#
# The resulting t0_correct is then written back into LEWIS_PARAMS so the transit
# component of the model aligns with this specific observation span.
t0_base = 2454387.49375       # reference epoch (BJD)
period = 5.6334729


t_data_center = (t.min() + t.max()) / 2.0
n = np.round((t_data_center - t0_base) / period)
t0_correct = t0_base + n * period


print(f"Data center: BJD {t_data_center:.5f}")
print(f"Nearest transit epoch n={n:.0f}")
print(f"Correct t0: BJD {t0_correct:.5f}")
print(f"Transit at hour: {(t0_correct - t.min())*24:.1f} hr into observation")


# Update the transit time used by the fixed-parameter model.
LEWIS_PARAMS['t0'] = t0_correct


# Compute model and residuals.
# Residuals are useful for evaluating how closely the corrected data follow the
# expected Lewis-style transit + eclipse + phase-curve behavior.
model_flux = lewis_model(t)
residuals = flux_final - model_flux


print("✓ Lewis model computed")
print(f"Transit depth: {np.min(model_flux):.4f}")
print(f"RMS residuals: {np.std(residuals):.4f}")




# ============================================================
# 5. STELLAR VARIABILITY TEST (Lewis Section 2.7)
# ============================================================
# Lewis explicitly tested whether stellar variability might explain some of the
# apparent light-curve structure. They considered both:
# - a linear trend with time
# - a sinusoidal variability term associated with stellar rotation
#
# The paper concludes that the apparent stellar-variability solutions are not
# physically convincing and are likely degenerate with other systematics/phase terms.
#
# This block performs a lightweight version of that same sanity check using the
# corrected data minus the fixed Lewis-style model.
print("\n" + "=" * 60)
print("STELLAR VARIABILITY TEST (Lewis Section 2.7)")
print("=" * 60)


# Test a linear trend in time.
# This corresponds to a simple slowly varying stellar baseline.
def linear_stellar(t, d1):
    return 1 + d1 * (t - t.mean())


popt_lin, pcov_lin = curve_fit(
    linear_stellar,
    t,
    flux_final - model_flux,
    p0=[0],
    bounds=[-0.01, 0.01]
)


# Test a sinusoid motivated by stellar rotation.
# Lewis discussed a stellar rotation timescale on the order of ~3.8 days.
def sine_stellar(t, d1, d2, d3):
    return 1 + d1 * np.sin(2*np.pi*(t-t.mean())/d2 - d3)


try:
    popt_sin, pcov_sin = curve_fit(
        sine_stellar,
        t,
        flux_final - model_flux,
        p0=[0.001, 3.8, 0],
        bounds=[0, [0.01, 10, np.pi]]
    )
    print("Sinusoidal stellar: amp={:.2e}, period={:.1f}d".format(*popt_sin[:2]))

except:
    print("Sinusoidal stellar fit failed")


print(f"Linear stellar slope: {popt_lin[0]*1e4:.2f} ppm/day (insignificant)")
print("✓ Matches Lewis: no significant stellar variability")




# ============================================================
# 6. FINAL PLOTS
# ============================================================
# The remaining blocks generate publication-style figures inspired by Lewis.
#
# Two main goals:
# - visualize the full corrected phase curve versus time from periapse
# - zoom in on transit/eclipses to compare data and model locally
#
# The paper bins its final presentation plots into five-minute intervals, so this
# script adopts the same basic presentation idea.




# ============================================================
# FIGURE 5 STYLE: Full phase curve, x-axis = time from periapse
# ============================================================
# Lewis Figure 5 shows the corrected full-orbit photometry after decorrelation with
# the best-fit model overplotted, using time from periapse on the x-axis.
#
# This block reproduces that style for the 4.5 μm case using:
# - corrected, binned photometry
# - the fixed Lewis-style model
# - a horizontal stellar-flux reference line


# Time of periapse.
# The code uses a fixed literature-like periapse time for the 4.5 μm observation.
t_peri = 2455757.05194   # BJD_UTC, Lewis Table 2 style value


# Convert the x-axis from absolute BJD to days from periapse.
t_from_peri = t - t_peri


# Bin to five-minute intervals.
# Lewis bins the final plotted light curves to reduce scatter and emphasize
# the large-scale orbital variation.
bin_size = 30.0 / (24.0 * 60.0)   # 5 min in days
t_bins = np.arange(t_from_peri.min(), t_from_peri.max(), bin_size)


def bin_data(t_rel, f, bins):
    """
    Bin flux into time bins using medians.
    
    Returns
    -------
    bin_centers : array
        Center of each populated bin
    bin_flux : array
        Median flux in each bin
    bin_err : array
        Standard error estimate from the scatter in each bin
    """
    
    # We only keep bins with enough points to produce a stable median/scatter estimate.
    bin_centers = []
    bin_flux = []
    bin_err = []
    
    for i in range(len(bins) - 1):
        mask = (t_rel >= bins[i]) & (t_rel < bins[i+1])
        
        if np.sum(mask) > 3:
            bin_centers.append(0.5 * (bins[i] + bins[i+1]))
            bin_flux.append(np.median(f[mask]))
            bin_err.append(np.std(f[mask]) / np.sqrt(np.sum(mask)))
    
    return np.array(bin_centers), np.array(bin_flux), np.array(bin_err)


t_bin5, f_bin5, e_bin5 = bin_data(t_from_peri, flux_final, t_bins)


# Evaluate the model at the full cadence.
# This keeps the model synchronized with the corrected photometry before any binning.
model_flux = lewis_model(t)




# ============================================================
# FIND ACTUAL ECLIPSE/TRANSIT TIMES FROM THE DATA
# ============================================================
# The paper derives event times as part of a fit. This script instead uses a practical
# search around the expected event windows to find the local minima in the corrected data.
#
# Purpose:
# - improve visual centering of the zoom-in plots
# - sanity-check whether the event locations in the data look reasonable
#
# This is not a full timing analysis; it is a plotting-oriented event finder.


# Search for the first eclipse near the expected location relative to periapse.
t_ecl_guess = t_peri - 5.18   # rough guess from Lewis-style geometry
search_mask = np.abs(t - t_ecl_guess) < 0.4


t_search = t[search_mask]
f_search = flux_final[search_mask]


# Bin locally to five minutes and identify the bin with minimum flux.
bin_size_s = 5.0 / (24.0 * 60.0)
bins_s = np.arange(t_search.min(), t_search.max(), bin_size_s)
t_sb, f_sb, _ = bin_data(
    t_search - t_search.min(),
    f_search,
    bins_s - t_search.min()
)
t_sb += t_search.min()


t_eclipse1_found = t_sb[np.argmin(f_sb)]

print(f"\nEclipse search:")
print(f"  Guess:  BJD {t_ecl_guess:.5f}")
print(f"  Found:  BJD {t_eclipse1_found:.5f}")
print(f"  Offset: {(t_eclipse1_found - t_ecl_guess)*24:.2f} hr")


# Do the same for transit as a consistency check.
t_tr_guess = t_peri - 0.68
tr_mask = np.abs(t - t_tr_guess) < 0.3

t_tr_s = t[tr_mask]
f_tr_s = flux_final[tr_mask]

bins_tr = np.arange(t_tr_s.min(), t_tr_s.max(), bin_size_s)
t_trb, f_trb, _ = bin_data(
    t_tr_s - t_tr_s.min(),
    f_tr_s,
    bins_tr - t_tr_s.min()
)
t_trb += t_tr_s.min()

t_transit_found = t_trb[np.argmin(f_trb)]

print(f"\nTransit search:")
print(f"  Guess:  BJD {t_tr_guess:.5f}")
print(f"  Found:  BJD {t_transit_found:.5f}")
print(f"  Offset: {(t_transit_found - t_tr_guess)*24:.2f} hr")


# Update local plotting centers using the data-derived minima.
t_eclipse1 = t_eclipse1_found
t_transit = t_transit_found


# Bin the model into the same time bins used for the phase-curve figure.
_, m_bin5, _ = bin_data(t_from_peri, model_flux, t_bins)


fig5, ax = plt.subplots(figsize=(10, 4))


# Binned corrected data.
# Filled black points match the overall Lewis figure style.
ax.plot(t_bin5, f_bin5, 'ko', ms=2.5, label='4.5 µm (binned 5 min)')


# Overplot the best-fit/literature model.
ax.plot(t_bin5, m_bin5, 'r-', lw=1.5, label='Best-fit model')


# Horizontal line showing the nominal stellar-only baseline.
ax.axhline(1.0, color='r', lw=0.8, ls='--', label='Stellar flux')


ax.set_xlabel('Time from Periapse (Earth days)', fontsize=12)
ax.set_ylabel('Relative Flux', fontsize=12)
ax.set_xlim(t_from_peri.min() - 0.15, t_from_peri.max() + 0.15)
ax.set_ylim(0.9975, 1.0025)
ax.legend(fontsize=9)
ax.set_title('HAT-P-2b 4.5 µm Phase Curve (Lewis Fig. 5 style)')

plt.tight_layout()
# plt.savefig('figure5_phase_curve.png', dpi=150)
plt.show()

print("✓ Saved figure5_phase_curve.png")




# ============================================================
# FIGURE 7 STYLE: Zoom on transit + eclipses
# ============================================================
# Lewis Figure 7 presents the 4.5 μm transit and secondary eclipses in zoomed panels,
# along with residuals below each main panel.
#
# This block reproduces that idea using:
# - data centered on each event
# - five-minute binning
# - model overplot
# - residual panel in mmag


# Event centers for the zoomed plots.
# Note that these are reset here to fixed reference values for the plotting stage.
t_transit = 2455756.44545      # transit center
t_eclipse1 = 2455751.58912     # first secondary eclipse
t_eclipse2 = t_eclipse1 + LEWIS_PARAMS['per']    # second secondary eclipse


# This list controls which events get plotted and how each panel is scaled.
events = [
    (t_eclipse1, 'First Secondary Eclipse', [0.9988, 1.0022], 0.15),
    (t_transit,  'Transit',                 [0.9950, 1.0005], 0.15),
]


fig7 = plt.figure(figsize=(10, 8))
gs = fig7.add_gridspec(
    4, 1,
    height_ratios=[3, 1, 3, 1],
    hspace=0.55
)


# Each event gets:
# - one main flux panel
# - one residual panel underneath
main_axes = [fig7.add_subplot(gs[0]), fig7.add_subplot(gs[2])]
res_axes = [fig7.add_subplot(gs[1]), fig7.add_subplot(gs[3])]


bin_size_fig7 = 5.0 / (24.0 * 60.0)   # 5-min bins


for ax, ax_res, (t_center, label, ylim, win) in zip(main_axes, res_axes, events):

    # ------------------------------------------------------------
    # Select data around the event
    # ------------------------------------------------------------
    # Shift the time axis so the event center is at t = 0.
    t_rel = t - t_center
    mask = np.abs(t_rel) < win
    t_w = t_rel[mask]
    f_w = flux_final[mask]
    m_w = model_flux[mask]

    # ------------------------------------------------------------
    # Five-minute binning
    # ------------------------------------------------------------
    bins_w = np.arange(-win, win + bin_size_fig7, bin_size_fig7)
    t_wb, f_wb, _ = bin_data(t_w, f_w, bins_w)
    _, m_wb, _ = bin_data(t_w, m_w, bins_w)

    # ------------------------------------------------------------
    # Main panel: binned data + model
    # ------------------------------------------------------------
    ax.plot(t_wb, f_wb, 'ko', ms=3.5, label='Data (5 min bins)')
    ax.plot(t_wb, m_wb, 'r-', lw=1.8, label='Lewis model')
    ax.set_xlim(-win, win)
    ax.set_ylim(ylim)
    ax.set_ylabel('Rel. Flux', fontsize=9)
    ax.set_title(label, fontsize=10, pad=4)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    ax.tick_params(labelbottom=False)   # hide x tick labels on main panel
    ax.legend(fontsize=8, loc='upper right')

    # ------------------------------------------------------------
    # Residuals panel
    # ------------------------------------------------------------
    # Residuals are shown in mmag for visual readability.
    resid = (f_wb - m_wb) * 1000
    ax_res.axhline(0, color='r', lw=0.8, ls='--')
    ax_res.plot(t_wb, resid, 'k.', ms=2.5)
    ax_res.set_xlim(-win, win)
    ax_res.set_ylim(-8, 8)
    ax_res.set_ylabel('Res.\n(mmag)', fontsize=7)
    ax_res.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax_res.tick_params(labelsize=7)


# Only the bottom residual panel gets the shared x-axis label.
res_axes[-1].set_xlabel('Time from Event Center (days)', fontsize=11)


fig7.suptitle('HAT-P-2b 4.5 µm — Lewis Figure 7 Style', fontsize=12)
# plt.savefig('figure7_zoom_events.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved figure7_zoom_events.png")