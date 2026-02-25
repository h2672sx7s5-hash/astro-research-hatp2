"""
USE DIFFERENT PYTHON VERSION
"""

import os

# Make RadVel happy on Windows
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.environ.get('USERPROFILE', os.path.expanduser('~'))
import os
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.environ.get('USERPROFILE', os.path.expanduser('~'))

import numpy as np
import matplotlib.pyplot as plt
import radvel
from radvel import fitting
import warnings
warnings.filterwarnings('ignore')

# ================================
# LOAD DATA
# ================================
data = np.loadtxt('hat_p2_rv.txt')
t = data[:,0] + 2450000.0    # BJD
rv = data[:,1]
err = data[:,2]

time_base = np.median(t)

# ================================
# SETUP PARAMS (1 PLANET)
# basis: per, tc, secosw, sesinw, k
# ================================
nplanets = 1
basis = 'per tc secosw sesinw k'

# NOTE: no time_base here
params = radvel.Parameters(nplanets, basis=basis)

# HAT-P-2b initial guesses
per = 5.6334729
tc  = 2454529.674
e   = 0.502
wdeg = 188.8
wrad = np.deg2rad(wdeg)
secosw = np.sqrt(e) * np.cos(wrad)
sesinw = np.sqrt(e) * np.sin(wrad)

params['per1']    = radvel.Parameter(value=per)
params['tc1']     = radvel.Parameter(value=tc)
params['secosw1'] = radvel.Parameter(value=secosw)
params['sesinw1'] = radvel.Parameter(value=sesinw)
params['k1']      = radvel.Parameter(value=950.0)  # m/s

# Single gamma and jitter
params['gamma'] = radvel.Parameter(value=np.mean(rv))
params['jit']   = radvel.Parameter(value=3.0)

# Which parameters vary?
for k in params.keys():
    params[k].vary = True

# Fix period and eccentricity shape if you want
params['per1'].vary    = False
params['secosw1'].vary = False
params['sesinw1'].vary = False

# ================================
# BUILD MODEL & LIKELIHOOD
# ================================
mod = radvel.RVModel(params)
mod.time_base = time_base

like = radvel.likelihood.RVLikelihood(mod, t, rv, err)

# ================================
# MAXIMUM-LIKELIHOOD FIT
# ================================
like = fitting.maxlike_fitting(like)
best_params = like.params

K_best   = best_params['k1'].value
gamma    = best_params['gamma'].value
jit      = best_params['jit'].value
per_best = best_params['per1'].value

print("\nBEST-FIT (RadVel)")
print("===========================")
print(f"per1   = {per_best:.7f} d (fixed)")
print(f"K1     = {K_best:.2f} m/s")
print(f"gamma  = {gamma:.2f} m/s")
print(f"jit    = {jit:.2f} m/s")
print(f"e      = {e:.3f} (fixed)")
print(f"omega  = {wdeg:.2f} deg (fixed)")

# Model + residuals
rv_model = mod(t)
resid = rv - rv_model
chi2 = np.sum((resid/err)**2)
dof = len(t) - sum(p.vary for p in best_params.values())
chi2_red = chi2/dof

print(f"\nchi2   = {chi2:.2f}")
print(f"chi2_r = {chi2_red:.3f}")
print(f"RMS    = {np.std(resid):.2f} m/s")

# ================================
# PLOTS
# ================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# 1) RV vs time
ax = axes[0,0]
ax.errorbar(t, rv, yerr=err, fmt='o', ms=4, alpha=0.6, color='C0', label='Data')
ax.plot(t, rv_model, 'r-', lw=2, label='RadVel model')
ax.set_xlabel('BJD')
ax.set_ylabel('RV (m/s)')
ax.set_title('HAT-P-2b: RV vs Time')
ax.legend()
ax.grid(alpha=0.3)

# 2) Residuals vs time
ax = axes[0,1]
ax.errorbar(t, resid, yerr=err, fmt='o', ms=4, alpha=0.6, color='C0')
ax.axhline(0, color='r', ls='--', lw=1)
ax.set_xlabel('BJD')
ax.set_ylabel('Residuals (m/s)')
ax.set_title('Residuals vs Time')
ax.grid(alpha=0.3)

# 3) Phase-folded RV
ax = axes[1,0]
phase = np.mod((t - tc)/per, 1.0)
phase_model = np.linspace(0,1,500)
t_model_phase = tc + phase_model*per
rv_model_phase =  mod(t_model_phase)

ax.errorbar(phase, rv, yerr=err, fmt='o', ms=4, alpha=0.6, color='C0', label='Data')
ax.plot(phase_model, rv_model_phase, 'r-', lw=2, label='RadVel model')
ax.set_xlabel('Orbital phase')
ax.set_ylabel('RV (m/s)')
ax.set_title('Phase-Folded RV')
ax.set_xlim(0,1)
ax.legend()
ax.grid(alpha=0.3)

# 4) Residual histogram
ax = axes[1,1]
ax.hist(resid, bins=20, color='C0', alpha=0.7, edgecolor='k')
ax.axvline(0, color='r', ls='--', lw=1)
ax.set_xlabel('Residuals (m/s)')
ax.set_ylabel('Count')
ax.set_title('Residual Distribution')
stats = f"K={K_best:.1f} m/s\nRMS={np.std(resid):.1f} m/s\nchi2_r={chi2_red:.2f}"
ax.text(0.97, 0.97, stats, transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('hatp2_radvel.png', dpi=200)
plt.show()

# ============================================
# MULTI-INSTRUMENT FIT (HIRES vs HARPS-N)
# ============================================

import radvel.likelihood as rlike

data_all = np.loadtxt('hat_p2_rv.txt')
t_all    = data_all[:, 0] + 2450000.0
rv_all   = data_all[:, 1]
err_all  = data_all[:, 2]
inst_id  = data_all[:, 3].astype(int)  # 0 = HIRES, 1 = HARPS-N

mask_hires  = (inst_id == 0)
mask_harpsn = (inst_id == 1)

params_multi = radvel.Parameters(1, basis=basis)
params_multi['per1']    = radvel.Parameter(value=per, vary=False)
params_multi['tc1']     = radvel.Parameter(value=tc,  vary=True)
params_multi['secosw1'] = radvel.Parameter(value=secosw, vary=False)
params_multi['sesinw1'] = radvel.Parameter(value=sesinw, vary=False)
params_multi['k1']      = radvel.Parameter(value=950.0, vary=True)

params_multi['gamma_hires']  = radvel.Parameter(value=np.mean(rv_all[mask_hires]),  vary=True, linear=True)
params_multi['jit_hires']    = radvel.Parameter(value=3.0,                          vary=True)
params_multi['gamma_harpsn'] = radvel.Parameter(value=np.mean(rv_all[mask_harpsn]), vary=True, linear=True)
params_multi['jit_harpsn']   = radvel.Parameter(value=3.0,                          vary=True)

mod_multi = radvel.RVModel(params_multi)
mod_multi.time_base = time_base

like_hires = rlike.RVLikelihood(
    mod_multi,
    t_all[mask_hires], rv_all[mask_hires], err_all[mask_hires],
    suffix='_hires'
)
like_harpsn = rlike.RVLikelihood(
    mod_multi,
    t_all[mask_harpsn], rv_all[mask_harpsn], err_all[mask_harpsn],
    suffix='_harpsn'
)

like_hires.params['gamma_hires']   = params_multi['gamma_hires']
like_hires.params['jit_hires']     = params_multi['jit_hires']
like_harpsn.params['gamma_harpsn'] = params_multi['gamma_harpsn']
like_harpsn.params['jit_harpsn']   = params_multi['jit_harpsn']

comp_like = rlike.CompositeLikelihood([like_hires, like_harpsn])
comp_like = fitting.maxlike_fitting(comp_like)
best      = comp_like.params

print("\nMULTI-INSTRUMENT (HIRES + HARPS-N) FIT")
print("=======================================")
print(f"K1           = {best['k1'].value:.2f} m/s")
print(f"gamma_hires  = {best['gamma_hires'].value:.2f} m/s")
print(f"gamma_harpsn = {best['gamma_harpsn'].value:.2f} m/s")
print(f"jit_hires    = {best['jit_hires'].value:.2f} m/s")
print(f"jit_harpsn   = {best['jit_harpsn'].value:.2f} m/s")

rv_model_multi = mod_multi(t_all)
resid_multi    = rv_all - rv_model_multi
chi2_multi     = np.sum((resid_multi / err_all)**2)
dof_multi      = len(t_all) - sum(p.vary for p in best.values())
chi2r_multi    = chi2_multi / dof_multi

print(f"\nchi2_multi   = {chi2_multi:.2f}")
print(f"chi2_r_multi = {chi2r_multi:.3f}")
print(f"RMS_multi    = {np.std(resid_multi):.2f} m/s")

# -------- PHASE-FOLDED PLOT (MULTI-INSTRUMENT) --------
per_fit = best['per1'].value   # fixed, but take from params
tc_fit  = best['tc1'].value

phase = np.mod((t_all - tc_fit) / per_fit, 1.0)
# Also make a second copy shifted by +1 to show wrap-around nicely
phase_all = np.concatenate([phase, phase + 1.0])
rv_all_2  = np.concatenate([rv_all, rv_all])
err_all_2 = np.concatenate([err_all, err_all])
inst_2    = np.concatenate([inst_id, inst_id])

phase_model = np.linspace(0, 2, 800)
t_model_phase = tc_fit + phase_model * per_fit
rv_model_phase = mod_multi(t_model_phase)

fig2, ax2 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Top: phase-folded RV
mask_hires2  = (inst_2 == 0)
mask_harpsn2 = (inst_2 == 1)

ax2[0].errorbar(phase_all[mask_hires2],  rv_all_2[mask_hires2],
                yerr=err_all_2[mask_hires2],
                fmt='o', ms=4, alpha=0.7, label='HIRES')
ax2[0].errorbar(phase_all[mask_harpsn2], rv_all_2[mask_harpsn2],
                yerr=err_all_2[mask_harpsn2],
                fmt='s', ms=4, alpha=0.7, label='HARPS-N')

ax2[0].plot(phase_model, rv_model_phase, 'k-', lw=1.5, label='Multi-inst model')
ax2[0].set_ylabel('RV (m/s)')
ax2[0].set_title('HAT-P-2b: phase-folded RV (multi-instrument)')
ax2[0].legend()
ax2[0].grid(alpha=0.3)

# Bottom: phase-folded residuals
resid_all_2 = np.concatenate([resid_multi, resid_multi])

ax2[1].errorbar(phase_all[mask_hires2],  resid_all_2[mask_hires2],
                yerr=err_all_2[mask_hires2],
                fmt='o', ms=4, alpha=0.7, label='HIRES')
ax2[1].errorbar(phase_all[mask_harpsn2], resid_all_2[mask_harpsn2],
                yerr=err_all_2[mask_harpsn2],
                fmt='s', ms=4, alpha=0.7, label='HARPS-N')

ax2[1].axhline(0, color='r', ls='--', lw=1)
ax2[1].set_xlabel('Orbital phase')
ax2[1].set_ylabel('Residuals (m/s)')
ax2[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('hatp2_radvel_multi_phase.png', dpi=200)
plt.show()

# # ============================================
# # EVOLVING ONE-PLANET MODEL: e(t), omega(t)
# # ============================================

# import radvel.likelihood as rlike
# from copy import deepcopy
# from radvel import kepler as rvk
# from radvel import orbit as rvo

# # Reference epoch for evolution (use median time)
# t0 = time_base  # already computed earlier as median BJD

# class EvolvingRVModel(radvel.RVModel):
#     """
#     One-planet model where e and omega evolve linearly with time:
#         e(t)      = e0 + de_dt * (t - t0)/365.25
#         omega(t)  = omega0 + dom_dt * (t - t0)/365.25
#     using RadVel's documented kepler() and orbit utilities.
#     """
#     def __init__(self, params, t0):
#         super().__init__(params)
#         self.t0 = t0  # reference epoch in BJD

#     def __call__(self, t):
#         # Time relative to reference epoch (for evolution terms)
#         dt_days  = t - self.t0
#         dt_years = dt_days / 365.25

#         # Evolving eccentricity and argument of periastron
#         e0      = self.params['e0'].value        # baseline e at t0
#         de_dt   = self.params['de_dt'].value     # 1/yr
#         omega0  = self.params['omega0'].value    # deg at t0
#         dom_dt  = self.params['dom_dt'].value    # deg/yr

#         e_t = e0 + de_dt * dt_years
#         e_t = np.clip(e_t, 1e-5, 0.99)           # keep in (0,1)
#         omega_t_deg = omega0 + dom_dt * dt_years
#         omega_t = np.deg2rad(omega_t_deg)

#         # Orbital elements that we treat as constant
#         per = self.params['per1'].value
#         tc  = self.params['tc1'].value
#         k   = self.params['k1'].value

#         # Map transit time -> time of periastron using baseline e0, omega0
#         omega0_rad = np.deg2rad(omega0)
#         tp0 = rvo.timetrans_to_timeperi(tc, per, e0, omega0_rad)

#         # Mean anomaly at each time
#         n = 2.0 * np.pi / per
#         M = n * (t - tp0)

#         # Solve Kepler's equation for E(M, e_t)
#         E = rvk.kepler(M, e_t)

#         # True anomaly
#         cosE = np.cos(E)
#         sinE = np.sin(E)
#         sqrt1pe = np.sqrt(1.0 + e_t)
#         sqrt1me = np.sqrt(1.0 - e_t)

#         f = 2.0 * np.arctan2(sqrt1pe * sinE / 2.0,
#                              sqrt1me * cosE / 2.0)

#         # Radial velocity (planet contribution only; gammas handled in likelihood)
#         vr = k * (np.cos(f + omega_t) + e_t * np.cos(omega_t))

#         return vr

# # --- Build parameters for evolving model ---

# params_evol = radvel.Parameters(1, basis=basis)

# # Keep period fixed, tc free as before
# params_evol['per1'] = radvel.Parameter(value=per, vary=False)
# params_evol['tc1']  = radvel.Parameter(value=tc,  vary=True)

# # Replace fixed e, omega with evolving parameters
# params_evol['e0']      = radvel.Parameter(value=e,      vary=True)      # baseline e at t0
# params_evol['de_dt']   = radvel.Parameter(value=0.0,    vary=True)      # 1/yr
# params_evol['omega0']  = radvel.Parameter(value=wdeg,   vary=True)      # deg at t0
# params_evol['dom_dt']  = radvel.Parameter(value=0.0,    vary=True)      # deg/yr

# params_evol['k1'] = radvel.Parameter(value=950.0, vary=True)

# # Per-instrument gamma/jitter (start from previous best values if you want)
# params_evol['gamma_hires']  = radvel.Parameter(value=best['gamma_hires'].value,  vary=True, linear=True)
# params_evol['jit_hires']    = radvel.Parameter(value=best['jit_hires'].value,    vary=True)
# params_evol['gamma_harpsn'] = radvel.Parameter(value=best['gamma_harpsn'].value, vary=True, linear=True)
# params_evol['jit_harpsn']   = radvel.Parameter(value=best['jit_harpsn'].value,   vary=True)

# # --- Likelihood setup ---

# mod_evol = EvolvingRVModel(params_evol, t0=t0)

# like_hires_e = rlike.RVLikelihood(
#     mod_evol,
#     t_all[inst_id == 0], rv_all[inst_id == 0], err_all[inst_id == 0],
#     suffix='_hires'
# )
# like_harpsn_e = rlike.RVLikelihood(
#     mod_evol,
#     t_all[inst_id == 1], rv_all[inst_id == 1], err_all[inst_id == 1],
#     suffix='_harpsn'
# )

# like_hires_e.params['gamma_hires']   = params_evol['gamma_hires']
# like_hires_e.params['jit_hires']     = params_evol['jit_hires']
# like_harpsn_e.params['gamma_harpsn'] = params_evol['gamma_harpsn']
# like_harpsn_e.params['jit_harpsn']   = params_evol['jit_harpsn']

# comp_like_e = rlike.CompositeLikelihood([like_hires_e, like_harpsn_e])

# # --- Fit evolving model ---

# comp_like_e = fitting.maxlike_fitting(comp_like_e)
# best_e = comp_like_e.params

# print("\nEVOLVING ONE-PLANET FIT (e(t), omega(t))")
# print("=========================================")
# print(f"e0        = {best_e['e0'].value:.5f}")
# print(f"de/dt     = {best_e['de_dt'].value:.3e} 1/yr")
# print(f"omega0    = {best_e['omega0'].value:.3f} deg")
# print(f"domega/dt = {best_e['dom_dt'].value:.3e} deg/yr")
# print(f"K1        = {best_e['k1'].value:.2f} m/s")
# print(f"gamma_hires   = {best_e['gamma_hires'].value:.2f} m/s")
# print(f"gamma_harpsn  = {best_e['gamma_harpsn'].value:.2f} m/s")
# print(f"jit_hires     = {best_e['jit_hires'].value:.2f} m/s")
# print(f"jit_harpsn    = {best_e['jit_harpsn'].value:.2f} m/s")

# rv_model_e = mod_evol(t_all)
# resid_e    = rv_all - rv_model_e
# chi2_e     = np.sum((resid_e / err_all)**2)
# dof_e      = len(t_all) - sum(p.vary for p in best_e.values())
# chi2r_e    = chi2_e / dof_e

# print(f"\nchi2_e    = {chi2_e:.2f}")
# print(f"chi2_r_e  = {chi2r_e:.3f}")
# print(f"RMS_e     = {np.std(resid_e):.2f} m/s")

# # --- Phase-folded plot for evolving model ---

# per_fit_e = best_e['per1'].value
# tc_fit_e  = best_e['tc1'].value

# phase_e = np.mod((t_all - tc_fit_e) / per_fit_e, 1.0)
# phase_e_all = np.concatenate([phase_e, phase_e + 1.0])
# rv_all_2_e  = np.concatenate([rv_all, rv_all])
# err_all_2_e = np.concatenate([err_all, err_all])
# inst_2_e    = np.concatenate([inst_id, inst_id])

# phase_model_e = np.linspace(0, 2, 800)
# t_model_phase_e = tc_fit_e + phase_model_e * per_fit_e
# rv_model_phase_e = mod_evol(t_model_phase_e)

# fig3, ax3 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# mask_hires2_e  = (inst_2_e == 0)
# mask_harpsn2_e = (inst_2_e == 1)

# ax3[0].errorbar(phase_e_all[mask_hires2_e],  rv_all_2_e[mask_hires2_e],
#                 yerr=err_all_2_e[mask_hires2_e],
#                 fmt='o', ms=4, alpha=0.7, label='HIRES')
# ax3[0].errorbar(phase_e_all[mask_harpsn2_e], rv_all_2_e[mask_harpsn2_e],
#                 yerr=err_all_2_e[mask_harpsn2_e],
#                 fmt='s', ms=4, alpha=0.7, label='HARPS-N')

# ax3[0].plot(phase_model_e, rv_model_phase_e, 'k-', lw=1.5, label='Evolving model')
# ax3[0].set_ylabel('RV (m/s)')
# ax3[0].set_title('HAT-P-2b: phase-folded RV (evolving one-planet)')
# ax3[0].legend()
# ax3[0].grid(alpha=0.3)

# resid_all_2_e = np.concatenate([resid_e, resid_e])

# ax3[1].errorbar(phase_e_all[mask_hires2_e],  resid_all_2_e[mask_hires2_e],
#                 yerr=err_all_2_e[mask_hires2_e],
#                 fmt='o', ms=4, alpha=0.7, label='HIRES')
# ax3[1].errorbar(phase_e_all[mask_harpsn2_e], resid_all_2_e[mask_harpsn2_e],
#                 yerr=err_all_2_e[mask_harpsn2_e],
#                 fmt='s', ms=4, alpha=0.7, label='HARPS-N')

# ax3[1].axhline(0, color='r', ls='--', lw=1)
# ax3[1].set_xlabel('Orbital phase')
# ax3[1].set_ylabel('Residuals (m/s)')
# ax3[1].grid(alpha=0.3)

# plt.tight_layout()
# plt.savefig('hatp2_radvel_evolving_phase.png', dpi=200)
# plt.show()