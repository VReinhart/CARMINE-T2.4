import os
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Parameters: change these as needed
cmap = "viridis"

# EXAMPLE 1: era5-2km 
pilotarea = "Athens"
dataset_name = "era5-2km"
indicator = "cdd"
start_year = 1989
end_year = 2018
hist_start_year = None
hist_end_year = None
proj_start_year = None
proj_end_year = None
scenario = None
'''
# =======================================
# (Uncomment to use)
# EXAMPLE 2: eu-cordex-11 
pilotarea = "Athens"
dataset_name = "eu-cordex-11"
indicator = "cdd"
hist_start_year = 1981
hist_end_year = 2010
proj_start_year = 2021
proj_end_year = 2050
scenario = "rcp26"
start_year = hist_start_year
end_year = proj_end_year
'''
# Construct base path
base = f"/work/cmcc/gf27821/CARMINE/CARMINE-T2.4/{pilotarea}/INDICATORS"

def construct_filename(pilotarea, dataset_name, indicator, start_year, end_year,
                      hist_start_year=None, hist_end_year=None,
                      proj_start_year=None, proj_end_year=None, scenario=None):
    pilot_lower = pilotarea.lower()

    if dataset_name in ["cerra", "eobs"]:
        period_start = str(start_year)
        period_end = str(end_year)
        filename = f"{pilot_lower}_{dataset_name}_{indicator}_eu_{period_start}_{period_end}.nc"
    elif dataset_name == "era5-2km":
        period_start = str(start_year)
        period_end = str(end_year)
        filename = f"{pilot_lower}_{dataset_name}_{indicator}_{period_start}{period_end}.nc"
    elif dataset_name == "eu-cordex-11":
        filename = f"{pilot_lower}_{dataset_name}_{indicator}_{hist_start_year}-{hist_end_year}_{proj_start_year}-{proj_end_year}_{scenario}.nc"
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: cerra, eobs, era5-2km, eu-cordex-11")

    return filename

def get_title_and_savefig(pilotarea, dataset_name, var_name, start_year, end_year,
                         hist_start_year=None, hist_end_year=None,
                         proj_start_year=None, proj_end_year=None, scenario=None):
    if dataset_name == "eu-cordex-11":
        title = f"Average Map of {var_name} from {hist_start_year}-{hist_end_year} & {proj_start_year}-{proj_end_year}\n" \
                f"over {pilotarea} in {dataset_name} ({scenario})"
        savefig_name = f"{pilotarea}_{var_name}_{dataset_name}_{hist_start_year}-{hist_end_year}_{proj_start_year}-{proj_end_year}_{scenario}.png"
    else:
        title = f"Average Map of {var_name} from {start_year} to {end_year}\n" \
                f"over {pilotarea} in {dataset_name}"
        savefig_name = f"{pilotarea}_{var_name}_{dataset_name}_{start_year}-{end_year}.png"

    return title, savefig_name

# Construct file path
filename = construct_filename(pilotarea, dataset_name, indicator, start_year, end_year,
                             hist_start_year=hist_start_year, hist_end_year=hist_end_year,
                             proj_start_year=proj_start_year, proj_end_year=proj_end_year,
                             scenario=scenario)
file = f"{base}/{filename}"
FUA_SHP = "/work/cmcc/gg21021/shapefile/UI-boundaries-FUA/FUA_Boundaries.shp"
print(f"Loading file: {file}")

# Load dataset
ds = xr.open_dataset(file, decode_timedelta=True)
print(f"Dataset variables: {list(ds.data_vars)}")
print(f"Dataset coords: {list(ds.coords)}")

# Get data and coordinates (KEEP 2D SHAPE for curvilinear grid)
var_name = 'CDD'
var = ds[var_name]
lon_2d = ds['lon'].values  # 2D array (5,3)
lat_2d = ds['lat'].values  # 2D array (5,3)
print(f"Data variable: '{var_name}' (shape: {var.shape})")
print(f"lon shape: {lon_2d.shape}, lat shape: {lat_2d.shape}")

# Process data
if np.issubdtype(var.dtype, np.timedelta64):
    data = var.astype("timedelta64[D]").astype(float).values
else:
    data = var.values
data = np.squeeze(data)

# FIXED: Calculate extent using 2D coordinates
non_nan_mask = ~np.isnan(data)
lon_valid = lon_2d[non_nan_mask]
lat_valid = lat_2d[non_nan_mask]

# Convert 0-360 lon to -180/+180 if needed
lon_valid = np.where(lon_valid > 180, lon_valid - 360, lon_valid)

lon_min, lon_max = lon_valid.min(), lon_valid.max()
lat_min, lat_max = lat_valid.min(), lat_valid.max()

# Add padding
lon_pad = (lon_max - lon_min) * 0.1
lat_pad = (lat_max - lat_min) * 0.1
lon_min -= lon_pad
lon_max += lon_pad
lat_min -= lat_pad
lat_max += lat_pad

print(f"Data extent: LON [{lon_min:.3f}, {lon_max:.3f}], LAT [{lat_min:.3f}, {lat_max:.3f}]")

# Load FUA
try:
    fua_gdf = gpd.read_file(FUA_SHP).to_crs(epsg=4326)
    fua_shape = fua_gdf[fua_gdf["FUA_NAME"] == "Athina"]
    fua_found = not fua_shape.empty
    print(f"FUA '{pilotarea}': {'Found' if fua_found else 'Not found'}")
except Exception as e:
    fua_shape = None
    fua_found = False
    print(f"Warning: Could not load FUA shapefile: {e}")

# Plot
plt.figure(figsize=(12, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True, dms=True, alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# FIXED: pcolormesh with 2D curvilinear coordinates (same shape as data)
lon_plot_2d = np.where(lon_2d > 180, lon_2d - 360, lon_2d)
im = ax.pcolormesh(lon_plot_2d, lat_2d, data, cmap=cmap, transform=ccrs.PlateCarree(),
                   vmin=np.nanpercentile(data, 5), vmax=np.nanpercentile(data, 95))

if fua_found:
    fua_shape.boundary.plot(ax=ax, edgecolor='k', linewidth=1, transform=ccrs.PlateCarree())
    print(f"FUA bounds: {fua_shape.total_bounds}")

cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, pad=0.05)
cbar.set_label(f'{var_name} (days)', fontsize=12)

ax.set_extent([lon_valid.min(), lon_valid.max(), lat_valid.min(), lat_valid.max()], crs=ccrs.PlateCarree())

title, savefig_name = get_title_and_savefig(pilotarea, dataset_name, var_name, start_year, end_year,
                                           hist_start_year, hist_end_year, proj_start_year, proj_end_year, scenario)
plt.title(title, fontsize=16, pad=20)
plt.xlabel('Longitude [degrees]')
plt.ylabel('Latitude [degrees]')

plt.savefig(savefig_name, dpi=300, bbox_inches='tight')
plt.show()
print(f"Map saved as: {savefig_name}")
