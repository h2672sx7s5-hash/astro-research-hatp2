import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from datetime import datetime
import glob
import os
# htf5, csv
BASE_PATH = r"C:\Users\robob\Documents\astro research; stellar pulsations\Spitzer datasets"

def load_all_spitzer_data(base_path):
    """Load all BCD files from all AORs"""
    
    aor_dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    print(f"Found {len(aor_dirs)} AOR directories\n")
    
    all_times = []
    all_fluxes = []
    total_files = 0
    
    for aor in aor_dirs:
        aor_path = os.path.join(base_path, aor, f"r{aor}")
        
        if not os.path.exists(aor_path):
            continue
        
        bcd_files = glob.glob(os.path.join(aor_path, "**", "*_bcd.fits"), recursive=True)
        
        if not bcd_files:
            continue
        
        print(f"AOR {aor}: {len(bcd_files)} BCD files")
        
        for i, fits_file in enumerate(bcd_files):
            try:
                with fits.open(fits_file, memmap=True) as hdul:
                    header = hdul[0].header
                    image_data = hdul[0].data
                    
                    date_obs_str = header.get('DATE_OBS', None)
                    if not date_obs_str:
                        continue
                    
                    try:
                        time_obj = datetime.fromisoformat(date_obs_str)
                    except:
                        time_obj = datetime.strptime(date_obs_str[:19], '%Y-%m-%dT%H:%M:%S')
                    
                    if image_data.ndim == 3:
                        flux_value = np.nanmean(image_data)
                    else:
                        h, w = image_data.shape
                        flux_value = np.nanmean(image_data[h//4:3*h//4, w//4:3*w//4])
                    
                    all_times.append(time_obj)
                    all_fluxes.append(flux_value)
                    total_files += 1
                    
                    if (i + 1) % 500 == 0:
                        print(f"  Processed {i + 1}/{len(bcd_files)} files...")
                        
            except Exception as e:
                continue
        
        print(f"  ✓ Subtotal: {total_files} times loaded\n")
    
    if all_times:
        sort_idx = np.argsort(np.array([t.timestamp() for t in all_times]))
        return np.array(all_times)[sort_idx], np.array(all_fluxes)[sort_idx]
    
    return None, None

print("="*70)
print("LOADING ALL SPITZER DATA")
print("="*70 + "\n")

times, fluxes = load_all_spitzer_data(BASE_PATH)

if times is None:
    print("No data loaded!")
else:
    # Convert to DAYS (not seconds!)
    times_days = np.array([(t - times[0]).total_seconds() / 86400.0 for t in times])
    fluxes_norm = fluxes / np.median(fluxes)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total observations: {len(times)}")
    print(f"Time span: {times_days[-1]:.1f} days ({times_days[-1]/365.25:.2f} years)")
    print(f"Date range: {times[0].date()} to {times[-1].date()}")
    print(f"Flux range: {np.min(fluxes_norm):.4f} - {np.max(fluxes_norm):.4f}")
    
    # ========== PLOT ==========
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(times_days, fluxes_norm, 'o', color='blue', markersize=2, alpha=0.4)
    ax.set_xlabel('Time (days from first observation)', fontsize=12)
    ax.set_ylabel('Normalized Flux', fontsize=12)
    ax.set_title(f'Spitzer All AORs Combined ({len(times)} observations over {times_days[-1]/365.25:.1f} years)', 
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('spitzer_all_aors_combined.png', dpi=300)
    plt.show()
    
    print(f"\n✓ Plot saved: spitzer_all_aors_combined.png")
