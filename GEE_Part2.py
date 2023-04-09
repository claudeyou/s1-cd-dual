"""
Created on Fri Mar 17 17:15:10 2023

@author: hoyeong
@Python: 3.9.16v

### GEE Part2 - Hypothesis Testing ###
"""

#%% Import libraries.

import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from scipy.stats import norm, gamma, f, chi2
from pyproj import Transformer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import IPython.display as disp
import numpy as np
import glob
import folium

#%% Find all TIF file / list
fp = glob.glob(r'/Users/hoyeong/Documents/PYTHON/ChangeDetection/CDT/S1_GRD/*.tif')
fp.sort()

# Read and process each GeoTIFF file
im_list = []

for file in [fp[0], fp[1]]:
    with rasterio.open(file) as src:
        vv_band = src.read(1)
        im_list.append(vv_band)

#%% AOI_sub https://geojson.io/

aoi_sub = {"coordinates": [
          [
            [
              129.33486887065214,
              35.94604676520781
            ],
            [
              129.33486887065214,
              35.933629878792274
            ],
            [
              129.35682593712005,
              35.933629878792274
            ],
            [
              129.35682593712005,
              35.94604676520781
            ],
            [
              129.33486887065214,
              35.94604676520781
            ]
          ]
        ],
        "type": "Polygon"
      }

#%% Transform into Same CRS

crs1 = CRS.from_epsg(4326)  # Source CRS
crs2 = CRS.from_epsg(32652)   # Target CRS 
aoi_sub = transform_geom(crs1, crs2, aoi_sub)

#%% Clip each image in im_list

im_list_clip = []

for file in [fp[0], fp[1]]:
    with rasterio.open(file) as src:
        clip_img, out_transform = mask(src, [aoi_sub], crop=True, indexes=1)
        im_list_clip.append(clip_img)

#%% Match the shape of Clipped Image (IF not the same shape)

# Match to min rows and columns.
if im_list_clip[0].shape != im_list_clip[1].shape:
    min_rows = min(im_list_clip[0].shape[0], im_list_clip[1].shape[0])
    min_cols = min(im_list_clip[0].shape[1], im_list_clip[1].shape[1])

    im1_sub = im_list_clip[0][:min_rows, :min_cols]
    im2_sub = im_list_clip[1][:min_rows, :min_cols]


elif im_list_clip[0].shape == im_list_clip[1].shape:
    im1_sub = im_list_clip[0]
    im2_sub = im_list_clip[1]

#%% Ratio Image

ratio = np.divide(im1_sub, im2_sub, where=np.logical_and(im2_sub != 0, im1_sub != 0), 
                  out=np.full_like(im1_sub, np.nan))

# Check the number of nan values
# nan_count = np.sum(np.isnan(ratio))

plt.figure(figsize = (12, 8))
plt.imshow(ratio, cmap='gray', vmin=0, vmax=10)
plt.colorbar()
plt.show()

#%% Making Histogram

ratio_flat = ratio.flatten() # 1-D Array
ratio_flat_rmnan = ratio_flat[~np.isnan(ratio_flat)] # remove nan value for statistical analysis

mean = np.mean(ratio_flat_rmnan)
variance = np.var(ratio_flat_rmnan)
y,x = np.histogram(ratio_flat_rmnan, bins = 500, range=(0,5)) # result = tuple

y = y / np.sum(y) # normalize
x = x[:-1]

plt.grid()
plt.plot(x, y, '.', label='data')
plt.legend()
plt.show()

#%% Overlaying F distribution

m = 5.0
plt.grid()
plt.plot(x, y, '.', label='data')
plt.plot(x, f.pdf(x, 2*m, 2*m) / 100, '-r', label='F-dist')
plt.legend()
plt.show()

print(mean, m/(m-1))
print(variance, m*(2*m-1)/(m-1)**2/(m-2))

#%% Display on Folium map

im_list = []

for file in [fp[0], fp[1]]:
    with rasterio.open(file) as src:
        raster_data = {
            'band_data': src.read(1), # read(1) : VV / read(2): VH
            'meta': src.meta,
            'bounds': src.bounds,
        }
        im_list.append(raster_data)

im1 = im_list[0]['band_data']
im2 = im_list[1]['band_data']

# Reproject if the shape or CRS not same.
# if im1.shape != im2.shape or im1_bounds != im2_bounds:
#     im2_resampled = np.empty_like(im1)
#     reproject(
#         im2, im2_resampled,
#         src_transform=im2_transform,
#         dst_transform=im1_transform,
#         src_crs=src.crs,
#         dst_crs=src.crs,
#         resampling=Resampling.nearest
#     )
#     im2 = im2_resampled

ratio = im1 / im2

def colormap_ratio(x):
    # Normalize ratio values between 0 and 20
    ratio_min, ratio_max = 0, 20
    normalized_x = min(max(x, ratio_min), ratio_max) / (ratio_max - ratio_min)
    
    # Map values to black and white color
    color = (normalized_x, normalized_x, normalized_x, 1)
    return color

# CRS transform
transformer = Transformer.from_crs(im_list[0]['meta']['crs'], 'EPSG:4326')

left, bottom = transformer.transform( im_list[0]['bounds'].left,
                                      im_list[0]['bounds'].bottom)
right, top = transformer.transform( im_list[0]['bounds'].right,
                                    im_list[0]['bounds'].top)

bounds = [
    [ left,  bottom],
    [right,     top]
]

# Display the result on a Folium map
# lat, long 
# opacity: transparency
location = [(top + bottom) / 2,
            (left + right) / 2][::-1]
mp = folium.Map(location=location, zoom_start=12)
mp.add_child(folium.raster_layers.ImageOverlay(ratio, bounds=bounds,
                                               opacity=0.7, colormap=colormap_ratio, 
                                               name = 'Ratio_map'))
mp.add_child(folium.LayerControl())
display(mp)
mp.save(r'/Users/hoyeong/Documents/PYTHON/ChangeDetection/CDT/Ratio_map.html')

#%% Statistical Testing

# Decision Threshold
dt = dt = f.ppf(0.0005, 2*m, 2*m)

# LRT statistics.
q1 = im1 / im2
q2 = im2 / im1

# Change map with 0 = no change, 1 = decrease, 2 = increase in intensity.
c_map = np.zeros_like(im1)
c_map[q2 < dt] = 1
c_map[q1 < dt] = 2

# Mask no-change pixels.
c_map = np.ma.masked_array(c_map, mask=(c_map == 0))
c_map = c_map.astype(int)

colors = ['black', 'blue', 'red']
cmap = ListedColormap(colors)

# Convert the masked array to a colored RGBA image
c_map = cmap(c_map)

mp = folium.Map(
    location=location, tiles='Stamen Toner',
    zoom_start=13)
folium.TileLayer('OpenStreetMap').add_to(mp)
mp.add_child(folium.raster_layers.ImageOverlay(ratio, bounds=bounds,
                                               opacity=1, colormap=colormap_ratio,
                                               name = 'Ratio_map'))
mp.add_child(folium.raster_layers.ImageOverlay(c_map, bounds=bounds,
                                               opacity=0.7, name='Change Map'))
mp.add_child(folium.LayerControl())
mp.save(r'/Users/hoyeong/Documents/PYTHON/ChangeDetection/CDT/Change_map.html')

#%% Bivariate change detection
'''Using VV, VH band together'''

def det(img):
    '''VV*VH'''
    return np.multiply(img['vv_band'], img['vh_band'])
    
im_list = []

for file in [fp[0], fp[1]]:
    with rasterio.open(file) as src:
        raster_data = {
            'band_data': {'vv_band': src.read(1),
                          'vh_band': src.read(2)}, # read(1) : VV / read(2): VH
            'meta': src.meta,
            'bounds': src.bounds,
        }
        im_list.append(raster_data)

im1 = im_list[0]['band_data']
im2 = im_list[1]['band_data']
im_sum = {'vv_band': im1['vv_band'] + im2['vv_band'],
          'vh_band': im1['vh_band'] + im2['vh_band']}

# number of looks
m = 5
m2logQ = (np.log(det(im1)) + np.log(det(im2)) - 2 * np.log(det(im_sum)) + 4 * np.log(2)) * (-2 * m)

#%% m2logQ and chi2pdf

m2logQ_flat = m2logQ.flatten()
y,x = np.histogram(m2logQ_flat, bins = 200, range=(0,20)) # result = tuple

y = y / np.sum(y) # normalize
x = x[:-1]

plt.plot(x, y, '.', label='data')
plt.plot(x, chi2.pdf(x, 2)/10, '-r', label='chi square')
plt.legend()
plt.grid()
plt.show()

#%% P-value

m2logq = (np.log(det(im1)) + np.log(det(im2)) - 2 * np.log(det(im_sum)) + 4 * np.log(2)) * (-2 * m)

# p_value
p_value = 1 - chi2.cdf(m2logq, 2)

pmap = plt.get_cmap('bone')
p_value = pmap(p_value)

mp = folium.Map(location=location, zoom_start=12)
mp.add_child(folium.raster_layers.ImageOverlay(p_value, bounds=bounds,
                                               opacity=0.7, name='P-Value'))
mp.add_child(folium.LayerControl())
mp.save(r'/Users/hoyeong/Documents/PYTHON/ChangeDetection/CDT/pvalue.html')
