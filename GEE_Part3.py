"""
Created on Fri Mar 24 18:41:10 2023

@author: hoyeong
@Python: 3.9.16v
"""


### GEE Part3 - Multitemporal Change detection ###

#%% ==================================== Import Library =========================================
import rasterio
from rasterio.windows import Window
from pyproj import Transformer, Proj
from scipy.interpolate import griddata
from scipy.ndimage import median_filter
from skimage.metrics import peak_signal_noise_ratio
import atsf_threshold.atsf_spatial as atsf_th
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import glob
import folium
from scipy.stats import norm, gamma, f, chi2

#%% ====================================== Functions =============================================

def det(img):
    """Calculates VV*VH"""
    return np.multiply(img['band_data']['vv_band'], img['band_data']['vh_band'])


def getvv(im_list):
    'VV band only in band_data'
    vv = []
    for img in im_list:
        raster_data = {
            'band_data': {'vv_band': img['band_data']['vv_band']}, 
            'angle': img['angle'],
            'meta': img['meta'],
            'bounds': img['bounds'],
            'name': img['name']
        }
        vv.append(raster_data)
    return vv


def longlat2window(lon,lat,dataset):
    p = Proj(dataset.crs)
    t = dataset.transform
    xmin, ymin = p(lon[0], lat[0])
    xmax, ymax = p(lon[1], lat[1])
    col_min, row_min = ~t*(xmin, ymin)
    col_max, row_max = ~t*(xmax, ymax)
    return Window.from_slices(rows=(np.floor(row_max), np.ceil(row_min)),
                              cols=(np.floor(col_min), np.ceil(col_max)))


def subset_raster(fp, lon, lat):
    im_list_clip=[]
    for file in fp:
        with rasterio.open(file) as src:
            p = Proj(src.crs)
            xmin, ymin = p(lon[0], lat[0])
            xmax, ymax = p(lon[1], lat[1])
            window = longlat2window(lon,lat,src)
            new_transform = src.window_transform(window)
            raster = {'band_data': {'vv_band': np.array(src.read(1,window=window)),
                                    'vh_band': np.array(src.read(2,window=window))},
                      'angle': np.array(src.read(3,window=window)), 
                      'meta': {'driver': src.meta['driver'],
                               'dtype': src.meta['dtype'],
                               'nodata': {'vv_band': np.sum(np.isnan(np.array(src.read(1,window=window)))),
                                          'vh_band': np.sum(np.isnan(np.array(src.read(2,window=window))))},
                               'width': np.array(src.read(1,window=window)).shape[1],
                               'height': np.array(src.read(1,window=window)).shape[0],
                               'count': 3,
                               'crs': src.meta['crs'],
                               'transform': new_transform},
                      'bounds': [xmin,ymin,xmax,ymax],
                      'name': src.name.split('/')[-1].split('.')[0]}
        im_list_clip.append(raster)
    return im_list_clip


def match(im_list):
    '''Match shape of arrays in im_list[i]['band_data']['vv/vh_band]'''
    for band in im_list[0]['band_data'].keys():
        min_rows, min_cols = im_list[0]['band_data'][band].shape[0], im_list[0]['band_data'][band].shape[1]

        for j in range(len(im_list)):
            if im_list[j]['band_data'][band].shape[0] < min_rows:
                min_rows = im_list[j]['band_data'][band].shape[0]

            if im_list[j]['band_data'][band].shape[1] < min_cols:
                min_cols = im_list[j]['band_data'][band].shape[1]

        for j in range(len(im_list)):
            im_list[j]['band_data'][band] = im_list[j]['band_data'][band][:min_rows, :min_cols]
            im_list[j]['meta']['width'] = min_cols
            im_list[j]['meta']['height'] = min_rows
    return im_list


def rgb_compostite(im_list, index, clip_range, band=1):
    '''Stack 3 bands'''
    if band == 1:
        '''VV band'''
        rgb_image = np.stack((im_list[index[0]]['band_data']['vv_band'],
                              im_list[index[1]]['band_data']['vv_band'],
                              im_list[index[2]]['band_data']['vv_band']), axis = -1)
    elif band == 2:
        '''VH band'''
        rgb_image = np.stack((im_list[index[0]]['band_data']['vh_band'],
                              im_list[index[1]]['band_data']['vh_band'],
                              im_list[index[2]]['band_data']['vh_band']), axis = -1)
    for i in range(rgb_image.shape[-1]):
        rgb_image[:, :, i] = interpolate_nan(rgb_image[:, :, i])
    
    rgb_image = 10*np.log10(rgb_image) # to decibels
    rgb_image = np.clip(rgb_image, clip_range[0], clip_range[1])
    rgb_image = (rgb_image + 20) / 20
    return rgb_image


def folium_transform(im_list):
    '''
    returns Bounds and Location in EPSG:4326
    To be more accurate, use coordinates from geoJSON.
    Can be slightly changed when tranforming crs if it is not 4326 formation
    '''
    if im_list[0]['meta']['crs'] != 'EPSG:4326':
        transformer = Transformer.from_crs(im_list[0]['meta']['crs'], 'EPSG:4326')

        minx, miny = im_list[0]['bounds'][0], im_list[0]['bounds'][1]
        maxx, maxy = im_list[0]['bounds'][2], im_list[0]['bounds'][3]
        for i in range(len(im_list)):
            if im_list[i]['bounds'][0] < minx:
                minx = im_list[i]['bounds'][0]
    
            if im_list[i]['bounds'][1] < miny:
                miny = im_list[i]['bounds'][1]
    
            if im_list[i]['bounds'][2] > maxx:
                maxx = im_list[i]['bounds'][2]
    
            if im_list[i]['bounds'][3] > maxy:
                maxy = im_list[i]['bounds'][3]

        minx, miny = transformer.transform(minx, miny)
        maxx, maxy = transformer.transform(maxx, maxy)

        bounds = [[minx,  miny],
                  [maxx,  maxy]][::-1]

        location = [(miny + maxy) / 2,
                    (minx + maxx) / 2][::-1]
        
    else:
        minx, miny = im_list[0]['bounds'][0], im_list[0]['bounds'][1]
        maxx, maxy = im_list[0]['bounds'][2], im_list[0]['bounds'][3]
        bounds = [[miny,  minx],
                  [maxy,  maxx]]

        location = [(miny + maxy) / 2,
                    (minx + maxx) / 2]
    return bounds, location


def omnibus(im_list, m=4.4):
    """Calculates the omnibus test statistic, monovariate(vv) case."""
    def log(current):
        return np.log(current)
    
    img = []
    k = len(im_list)
    klogk = k*np.log(k)

    for i in range(0,k):
        img.append(im_list[i]['band_data']['vv_band'])

    sumlogs = np.sum([log(i) for i in img], axis=0)
    logsum = np.log(np.sum(img, axis=0)) * k
    return (klogk + sumlogs - logsum) * (-2 * m)


def interpolate_nan(array):
    """Replace NaN values in the array with the interpolation."""
    x, y = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))
    valid_points = ~np.isnan(array)
    coords_valid = np.array((x[valid_points], y[valid_points])).T
    values_valid = array[valid_points]

    coords_all = np.array((x.flatten(), y.flatten())).T

    arr = griddata(coords_valid, values_valid, coords_all, method='nearest').reshape(array.shape)
    return arr


def sample_vv_imgs(j, im_list):
    """Samples the test statistics Rj """

    vv_list = [img['band_data']['vv_band'] for img in im_list]
    sj = vv_list[j - 1]
    jfact = ((j ** j)) / ((j-1) ** (j-1))
    sumj = np.sum(vv_list[:j], axis=0)
    sumjm1 = np.sum(vv_list[:j-1], axis=0)
    Rj = (((sumjm1 ** (j-1)) * sj * jfact) / ((sumj ** j))) ** 5

    sample = Rj.flatten()
    np.random.seed(123)
    sample = np.random.choice(sample, size=1000)
    return sample

#%% ============================================== File load ==================================================

fp = glob.glob(r'A:/PYTHON/ChangeDetection/S1_GRD/*.tif')
fp.sort()

im_list = []

for file in fp:
    '''Create im_list'''
    with rasterio.open(file) as src:
        raster_data = {
            'band_data': {'vv_band': src.read(1),
                          'vh_band': src.read(2)},
            'angle': src.read(3), # read(1) : VV | read(2): VH | read(3): Angle
            'meta': src.meta,
            'bounds': src.bounds,
            'name': src.name.split('/')[-1].split('.')[0]
        }
        im_list.append(raster_data)

# Check shape of im_list
for i in range(len(im_list)):
    print(im_list[i]['band_data']['vv_band'].shape)
print('\n')
for i in range(len(im_list)):
    print(im_list[i]['band_data']['vh_band'].shape)

# Match the shape of Images in im_list IF SHAPE DOESN'T MATCH
im_list = match(im_list)

#%% ============================================== Timestamplist ==============================================

timestamplist = []
for i in range(len(im_list)):
    timestamp = im_list[i]['name'].split('_')[-3:]
    timestamp = 'T'+ timestamp[-3] + timestamp[-2] + timestamp[-1]
    timestamplist.append(timestamp)

#%% ============================================== RGB Composite ==============================================

# Get VV band from im_list
vv_list = getvv(im_list)

# Stack 10,11,12 bands in vv_list
rgb_image = rgb_compostite(vv_list, [10,11,12], [-20,0], band=1)

# Folium map Display
bounds, location = folium_transform(vv_list)

mp = folium.Map(location=location, zoom_start=12)
mp.add_child(folium.raster_layers.ImageOverlay(rgb_image, bounds=bounds,
                                               opacity=1, name = 'RGB Composite'))
mp.add_child(folium.LayerControl())
mp.save(r'A:/PYTHON/ChangeDetection/CDT/hyyou/Folium/RGB_Composite.html')

#%% ======================================== Rate of False positive ============================================

alpha = 0.01
print('%.2f%%' %((1-(1-alpha)**len(im_list))*100))

#%% ================================== An omnibus test for change(VV only) =====================================

# No change area
aoi_sub = {"coordinates": [
          [
            [
              129.3352679337704,
              35.944751176829314
            ],
            [
              129.3352679337704,
              35.93625423648341
            ],
            [
              129.3574237991255,
              35.93625423648341
            ],
            [
              129.3574237991255,
              35.944751176829314
            ],
            [
              129.3352679337704,
              35.944751176829314
            ]
          ]
        ],
        "type": "Polygon"
      }

# Airport
aoi_sub = {
        "coordinates": [
          [
            [
              129.40532492028984,
              35.993667864492494
            ],
            [
              129.40532492028984,
              35.982117435563936
            ],
            [
              129.43682038773488,
              35.982117435563936
            ],
            [
              129.43682038773488,
              35.993667864492494
            ],
            [
              129.40532492028984,
              35.993667864492494
            ]
          ]
        ],
        "type": "Polygon"
      }
# Agriculture
aoi_sub = {
        "coordinates": [
          [
            [
              129.31901389465418,
              35.99882031088046
            ],
            [
              129.31901389465418,
              35.983422619426705
            ],
            [
              129.35852181280455,
              35.983422619426705
            ],
            [
              129.35852181280455,
              35.99882031088046
            ],
            [
              129.31901389465418,
              35.99882031088046
            ]
          ]
        ],
        "type": "Polygon"
      }

coordinates = aoi_sub['coordinates'][0]

min_lon = min(coord[0] for coord in coordinates)
max_lon = max(coord[0] for coord in coordinates)
min_lat = min(coord[1] for coord in coordinates)
max_lat = max(coord[1] for coord in coordinates)

im_list_sub = subset_raster(fp, lon=(min_lon, max_lon),
                                lat=(min_lat, max_lat))

im_list_sub = match(im_list_sub)
im_list_sub = getvv(im_list_sub)
rgb_image_sub = rgb_compostite(im_list_sub, [10,11,12], [-20,0], band=1)
bounds, location = folium_transform(im_list_sub)

mp = folium.Map(location=location, zoom_start=12)
mp.add_child(folium.raster_layers.ImageOverlay(rgb_image_sub, bounds=bounds,
                                               opacity=1, name = 'RGB_SUB Composite'))
mp.add_child(folium.LayerControl())
mp.save(r'A:/PYTHON/ChangeDetection/CDT/hyyou/Folium/RGB_SUB Composite.html')

#%% ======================================= Omnibus Histogram ====================================================

k = 6
omn = omnibus(im_list_sub[0:k], m=5)
omn = omn.flatten()

y,x = np.histogram(omn, bins = 200, range=(0,40)) 
y = y / np.sum(y) # normalize
x = x[:-1]

plt.plot(x, y, '.', label='data')
plt.plot(x, chi2.pdf(x, k-1)/5, '-r', label='chi square')
plt.legend()
plt.grid()
plt.show()

#%% ===================================== Change map for full series =============================================

k = len(vv_list[:16]); alpha = 0.01

p_value = 1 - chi2.cdf(omnibus(vv_list[:16],m=5), k-1)
c_map = (p_value < alpha).astype(int)

colors = [(0,0,0,0), 'red'] # transparent, red
cmap_color = ListedColormap(colors)
c_map = cmap_color(c_map)

bounds, location = folium_transform(vv_list)
mp = folium.Map(location=location, zoom_start=11)
mp.add_child(folium.raster_layers.ImageOverlay(c_map, bounds=bounds,
                                               opacity=1, name = 'Change Map'))
mp.add_child(folium.LayerControl())
mp.save(r'A:/PYTHON/ChangeDetection/CDT/hyyou/Folium/Omnibus Change Map.html')

#%% ==================================== Rj Correlation matrix =================================================

samples = [sample_vv_imgs(j, im_list_sub) for j in range(2, 9)]
np.set_printoptions(precision=2, suppress=True)
print(np.corrcoef(samples))

#%% ============================== Pre-calculating the P value array ===========================================

def log_det_sum(im_list, j):
    """Returns log of determinant of the sum of the first j images in im_list."""
    vv = 0
    vh = 0
    for i in range(j):
        vv = vv + im_list[i]['band_data']['vv_band']
        vh = vh + im_list[i]['band_data']['vh_band']
    
    sumj = {'band_data': {'vv_band': vv,
                          'vh_band': vh}}
    return np.log(det(sumj))


def log_det(im_list, j):
    """Returns log of the determinant of the jth image in im_list."""
    img = im_list[j - 1]
    return np.log(det(img))


def pval(im_list, j, m=4.4):
    """Calculates -2logRj for im_list and returns P value and -2logRj."""
    m2logRj = ((log_det_sum(im_list, j - 1) * (j - 1)
                + log_det(im_list, j)
                + 2 * j * np.log(j)
                - 2 * (j - 1) * np.log(j - 1)
                - log_det_sum(im_list, j) * j) * (-2 * m))
    pv = 1 - chi2.cdf(m2logRj, 2)
    return pv, m2logRj #()


def p_values(im_list):
    """Pre-calculates the P-value array for a list of images."""
    k = len(im_list)

    def ells_map(ell):
        """Arranges calculation of pval for combinations of k and j."""
        # Slice the series from k-l+1 to k (image indices start from 0).
        ell = int(ell)
        im_list_ell = im_list[k - ell: k]

        def js_map(j):
            """Applies pval calculation for combinations of k and j."""
            j = int(j)
            pv1, m2logRj1 = pval(im_list_ell, j)
            return {'pv': pv1, 'm2logRj': m2logRj1}

        # Map over j=2,3,...,l.
        js = list(range(2, ell + 1))
        pv_m2logRj = [js_map(j) for j in js]

        # Calculate m2logQl from collection of m2logRj images.
        m2logRj_sum = 0
        for item in pv_m2logRj:
            m2logRj_sum = m2logRj_sum + item['m2logRj']
        m2logQl = m2logRj_sum
        pvQl = 1 - chi2.cdf(m2logQl, (ell - 1) * 2)
        pvs = [item['pv'] for item in pv_m2logRj] + [pvQl]
        return pvs

    # Map over l = k to 2.
    ells = list(range(k, 1, -1))
    pv_arr = [ells_map(ell) for ell in ells]

    # Return the P value array ell = k,...,2, j = 2,...,l.
    return pv_arr

#%% Filtering the P values

def filter_j(prev, current):
    """Calculates change maps; iterates over j indices of pv_arr."""
    pv = current
    pvQ = prev['pvQ']
    i = prev['i']
    j = prev['j']
    alpha = prev['alpha']
    cmap = prev['cmap']
    smap = prev['smap']
    fmap = prev['fmap']
    bmap = prev['bmap']
    cmapj = np.multiply(0, cmap['band_data']['map'])
    cmapj = cmapj + (i + j - 1)
    # Check Rj : pv < alpha
    # Check Ql : pvQ < alpha
    # Check Row i : cmap == i-1
    tst = np.logical_and(np.logical_and(pv < alpha, pvQ < alpha), cmap['band_data']['map'] == (i - 1))

    # Update cmap, fmap, smap, bmap
    cmap['band_data']['map'] = np.where(tst, cmapj, cmap['band_data']['map'])
    fmap['band_data']['map'] = np.where(tst, fmap['band_data']['map'] + 1, fmap['band_data']['map'])
    if i == 1:
        smap['band_data']['map'] = np.where(tst, cmapj, smap['band_data']['map'])
    else:
        smap['band_data']['map'] = smap['band_data']['map']
    idx = i + j - 2
    bmap['band_data']['map'][idx][:, :] = np.where(tst, 1, bmap['band_data']['map'][idx][:, :])

    return {'i': i, 'j': j + 1, 'alpha': alpha, 'pvQ': pvQ, 
            'cmap': cmap, 'smap': smap, 'fmap': fmap, 'bmap': bmap}


def filter_i(prev, current):
    """Arranges calculation of change maps; iterates over row-indices of pv_arr."""
    pvs = current[0:-1]
    pvQ = current[-1]
    i = prev['i']
    alpha = prev['alpha']
    median = prev['median']

    # Filter Ql p value if desired.
    if median==True:
        pvQ = median_filter(pvQ, size=5)
    else:
        pvQ = pvQ

    cmap = prev['cmap']
    smap = prev['smap']
    fmap = prev['fmap']
    bmap = prev['bmap']

    first = {'i': i, 'j': 1, 'alpha': alpha ,'pvQ': pvQ,
             'cmap': cmap, 'smap': smap, 'fmap': fmap, 'bmap': bmap}
    result = reduce(filter_j, pvs, first)
    return {'i': i + 1, 'alpha': alpha, 'median': median,
            'cmap': result['cmap'], 'smap': result['smap'],
            'fmap': result['fmap'], 'bmap': result['bmap']}


def change_maps(im_list, median=False, alpha=0.01):
    """Calculates thematic change maps."""
    k = len(im_list)

    # Pre-calculate the P value array.
    pv_arr = p_values(im_list)
    # Filter P values for change maps
    cmap = {'band_data': {'map': np.zeros(im_list[0]['band_data']['vv_band'].shape)}, 
            'meta': {'dtype': {'type': 'PixelType', 'precision': 'np.float32'},
                     'width': im_list[0]['band_data']['vv_band'].shape[0],
                     'height': im_list[0]['band_data']['vv_band'].shape[1],
                     'crs': im_list[0]['meta']['crs'],
                     'transform': im_list[0]['meta']['transform']},
            'bounds': im_list[0]['bounds']}
    bmap = {'band_data': {'map': np.zeros((k-1,) + im_list[0]['band_data']['vv_band'].shape)}, 
            'meta': {'dtype': {'type': 'PixelType', 'precision': 'np.float32'},
                     'width': im_list[0]['band_data']['vv_band'].shape[0],
                     'height': im_list[0]['band_data']['vv_band'].shape[1],
                     'crs': im_list[0]['meta']['crs'],
                     'transform': im_list[0]['meta']['transform']},
            'bounds': im_list[0]['bounds']}
    smap = {'band_data': {'map': np.zeros(im_list[0]['band_data']['vv_band'].shape)}, 
            'meta': {'dtype': {'type': 'PixelType', 'precision': 'np.float32'},
                     'width': im_list[0]['band_data']['vv_band'].shape[0],
                     'height': im_list[0]['band_data']['vv_band'].shape[1],
                     'crs': im_list[0]['meta']['crs'],
                     'transform': im_list[0]['meta']['transform']},
            'bounds': im_list[0]['bounds']}
    fmap = {'band_data': {'map': np.zeros(im_list[0]['band_data']['vv_band'].shape)}, 
            'meta': {'dtype': {'type': 'PixelType', 'precision': 'np.float32'},
                     'width': im_list[0]['band_data']['vv_band'].shape[0],
                     'height': im_list[0]['band_data']['vv_band'].shape[1],
                     'crs': im_list[0]['meta']['crs'],
                     'transform': im_list[0]['meta']['transform']},
            'bounds': im_list[0]['bounds']}
    alpha = np.full(im_list[0]['band_data']['vv_band'].shape, alpha)
    first = {'i': 1, 'alpha': alpha, 'median': median,
             'cmap': cmap, 'smap': smap, 'fmap': fmap, 'bmap': bmap}
    result = reduce(filter_i, pv_arr, first)

    # Post-process bmap for change direction
    bmap = result['bmap']
    avimg = im_list[0]
    avimgcnt = np.full(im_list[0]['band_data']['vv_band'].shape, k)
    j = 0
    i = np.full(im_list[0]['band_data']['vv_band'].shape, 1)
    first = {'avimg': avimg, 'avimgcnt': avimgcnt, 'bmap': bmap, 'j': j, 'i': i}
    result2 = reduce(dmap_iter, im_list[1:], first)
    result['bmap'] = result2['bmap']
    result['avimg'] = result2['avimg']
    result['avimgcnt'] = result2['avimgcnt']
    return result


def dmap_iter(prev, current):
    """Reclassifies values in directional change maps."""
    j = prev['j']
    image = current['band_data']
    avimg = prev['avimg']
    avimgcnt = prev['avimgcnt']
    diff = {'band_data': {'vv_band': image['vv_band'] - avimg['band_data']['vv_band'],
                          'vh_band': image['vh_band'] - avimg['band_data']['vh_band']}}
    # Get Positive/Negative definiteness
    posd = np.logical_and(diff['band_data']['vv_band'] > 0, det(diff) > 0)
    negd = np.logical_and(diff['band_data']['vv_band'] < 0, det(diff) > 0)
    bmap = prev['bmap']
    k = len(bmap['band_data']['map'])
    bmapj = bmap['band_data']['map'][j][:,:]
    dmap = np.arange(1,4)
    bmapj[:, :] = np.where(bmapj, dmap[2], bmapj) # 3:indefinite difference
    bmapj[:, :] = np.where(np.logical_and(bmapj, posd), dmap[0], bmapj) # 1: positive definite difference
    bmapj[:, :] = np.where(np.logical_and(bmapj, negd), dmap[1], bmapj) # 2: negative definite difference
    bmap['band_data']['map'][j][:,:] = bmapj[:,:]
    # Update avimg with provisional means.
    i = prev['i'] + 1
    avimg = {'band_data': {'vv_band': avimg['band_data']['vv_band'] 
                           + (image['vv_band'] - avimg['band_data']['vv_band']) / i,
                           'vh_band': avimg['band_data']['vh_band'] 
                           + (image['vh_band'] - avimg['band_data']['vh_band']) / i}}
    # Reset avimg to current image and set i=1 if change occurred.
    avimg = {'band_data': {'vv_band': np.where(bmapj, image['vv_band'], avimg['band_data']['vv_band']),
                           'vh_band': np.where(bmapj, image['vh_band'], avimg['band_data']['vh_band'])}}
    avimgcnt = np.where(bmapj, k - j, avimgcnt)
    i = np.where(bmapj, 1, i)
    return {'avimg': avimg, 'avimgcnt': avimgcnt, 'bmap': bmap, 'j': j+1, 'i': i}



#%% ======================= Run algorithm for cmap, smap, fmap ==========================

result = change_maps(im_list, median=True, alpha=0.05)
cmap = result['cmap']['band_data']['map']
smap = result['smap']['band_data']['map']
fmap = result['fmap']['band_data']['map']

#%% ======================== Cmap, Smap, Fmap on Matplotlib =====================

maps = [cmap, smap, fmap]
map_names = ['cmap', 'smap', 'fmap']
colors = ['black', 'blue', 'cyan', 'yellow', 'red']
boundaries = [0, 3.75, 7.5, 11.25, 15]

norm = Normalize(vmin=0, vmax=15)
cmap_color = LinearSegmentedColormap.from_list('custom', colors, N=100)
cmap_color.set_over(colors[-1])
cmap_color.set_under(colors[0])

for i, map in enumerate(maps):

    # Display the array using the imshow function with the custom colormap and colorbar
    fig, ax = plt.subplots()
    im = ax.imshow(map, cmap=cmap_color, norm=norm)
    cbar = fig.colorbar(im, ticks=boundaries, label='Count', ax=ax)
    ax.set_title('{}'.format(map_names[i]))  # Use .format() method for title
    plt.tight_layout()

    # Save the entire plot
    output_path = 'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/{}.png'.format(map_names[i])
    plt.savefig(output_path, dpi=300)

    # Show the plot
    plt.show()

#%% =========================== Cmap, Smap, Fmap on Folium ============================
palette = [(0.0, 'black'), (0.25, 'blue'), (0.5, 'cyan'), (0.75, 'yellow'), (1.0, 'red')]
cmap_color = LinearSegmentedColormap.from_list('cmap', palette)

# Map the cmap, smap, and fmap values to the range [0, 1]
cmap_ = cmap / 15.0
smap_ = smap / 15.0
fmap_ = fmap / 15.0

# Apply the colormap to the values
cmap_fol = cmap_color(cmap_)
smap_fol = cmap_color(smap_)
fmap_fol = cmap_color(fmap_)

bounds, location = folium_transform(im_list)
mp = folium.Map(location=location, zoom_start=11)
mp.add_child(folium.raster_layers.ImageOverlay(cmap_fol, bounds=bounds,
                                               opacity=1, name='cmap'))
mp.add_child(folium.raster_layers.ImageOverlay(smap_fol, bounds=bounds,
                                               opacity=1, name='smap'))
mp.add_child(folium.raster_layers.ImageOverlay(fmap_fol, bounds=bounds,
                                               opacity=1, name='fmap'))

mp.add_child(folium.LayerControl())
mp.save(r'A:/PYTHON/ChangeDetection/CDT/hyyou/Folium/Sequential Omnibus test.html')


#%% ================= Run algorithm for bmap (alpha changed) ===============

# Run change_map
result = change_maps(im_list, median=True, alpha=0.01)
cmap = result['cmap']['band_data']['map']
smap = result['smap']['band_data']['map']
fmap = result['fmap']['band_data']['map']
bmap = result['bmap']['band_data']['map']
cmaps = {'band': {'cmap': cmap,
                  'smap': smap,
                  'fmap': fmap,
                  'bmap': {timestamplist[i]: bmap[i-1] for i in range(1, len(timestamplist[:16]))}}}

#%% ====================== Bmap on Folium ======================
cmaps_mask = {}
for key, value in cmaps['band'].items():
    if key == 'bmap':
        bmap_masked = {k: np.ma.masked_array(v, mask=(v == 0)) for k, v in value.items()}
        cmaps_mask[key] = bmap_masked
    else:
        cmaps_mask[key] = np.ma.masked_array(value, mask=(value == 0))
# Convert it into int()
for key, value in cmaps_mask.items():
    if key == 'bmap':
        for k, v in value.items():
            cmaps_mask[key][k] = v.astype(int)
    else:
        cmaps_mask[key] = value.astype(int)

# 1: positive definite difference (RED)
# 2: negative definite difference (CYAN)
# 3: indefinite difference        (YELLOW)

palette = ['black', 'red', 'cyan', 'yellow']
color_list = ListedColormap(palette)

bounds, location = folium_transform(im_list)
mp = folium.Map(location=location, zoom_start=11)
for i, timestamp in enumerate(timestamplist[1:16], 1):
    colored_cmap = color_list(cmaps_mask['bmap'][timestamp])
    mp.add_child(folium.raster_layers.ImageOverlay(colored_cmap, bounds=bounds, opacity=1, name=timestamp))
mp.add_child(folium.LayerControl())
mp.save(r'A:/PYTHON/ChangeDetection/CDT/hyyou/Folium/Sequential_test_Bmap.html')

#%% ======================== Display ATSF =====================================

# In log scale
log_ATSF = 10 * np.log10(result['avimg']['band_data']['vv_band'])
log_origin = 10 * np.log10(im_list[15]['band_data']['vv_band'])

# Thresholding
log_ATSF = np.clip(log_ATSF, -20, 0)
log_origin = np.clip(log_origin, -20, 0)

# Scaling
log_ATSF = ((log_ATSF + 20) * (255 / 20)).astype(np.uint8)
log_origin = ((log_origin + 20) * (255 / 20)).astype(np.uint8)

# Display log_ATSF in the first subplot
fig, axes = plt.subplots()
axes.imshow(log_ATSF, cmap='gray')
axes.set_title('ATSF')
plt.axis('on')
plt.tight_layout()
output_path = 'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/4326/ATSF_4326.png'
plt.savefig(output_path, dpi=300)
plt.show()

# Display log_origin in the second subplot
fig, axes = plt.subplots()
axes.imshow(log_origin, cmap='gray')
axes.set_title('Original')
plt.axis('on')
plt.tight_layout()
output_path = 'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/4326/Original_4326.png'
plt.savefig(output_path, dpi=300)
plt.show()

#%% =============================== Display ATSF count =================================

ATSF_cnt = result['avimgcnt']

# Normalize the values to the range 0-1
ATSF_cnt = (ATSF_cnt - 1) / (len(im_list[:16]) - 1)

# Scale the normalized values to the range 0-255
ATSF_cnt_ = (ATSF_cnt * 255).astype(np.uint8)

fig, axes = plt.subplots()
img = axes.imshow(ATSF_cnt_, cmap='gray')
axes.set_title('Count')
plt.axis('on')
plt.tight_layout()

# Add colorbar
cbar = plt.colorbar(img, ax=axes, ticks=np.linspace(0, 255, 16))
cbar.set_label('Image Count', labelpad=10)
cbar.ax.set_yticklabels(range(1, 17))

output_path = 'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/Count.png'
plt.savefig(output_path, dpi=300)
plt.show()

#%% =========================== ATSF in RGB ===============================
def atsfrgb(vv_arr,vh_arr):
    '''
    ---------
    vv_arr: 'VV' band numpy array
    vh_arr: 'VH' band numpy array
    Returns
    ---------
    RGB Image using (vv,vv,vh)
    '''
    def equalize_hist(channel):
        hist, bin_edges = np.histogram(channel, bins=256, range=(0, 1))
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]  # normalize
        equalized_channel = np.interp(channel, bin_edges[:-1], cdf)
        return equalized_channel
    
    # 10*nplog10
    R = 10 * np.log10(vv_arr)
    G = 10 * np.log10(vv_arr)
    B = 10 * np.log10(vh_arr)
    # Normalize
    R = (R - np.min(R)) / (np.max(R) - np.min(R))
    G = (G - np.min(G)) / (np.max(G) - np.min(G))
    B = (B - np.min(B)) / (np.max(B) - np.min(B))
    # Histogram Equalization
    R = equalize_hist(R)
    G = equalize_hist(G)
    B = equalize_hist(B)
    # Stack
    RGB = np.stack((R, G, B), axis=-1)
    return RGB

ATSF_RGB = atsfrgb(result['avimg']['band_data']['vv_band'], result['avimg']['band_data']['vh_band'])
fig, axes = plt.subplots()
axes.imshow(ATSF_RGB)
axes.set_title('ATSF_RGB')
plt.axis('on')
plt.tight_layout()
output_path = 'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/ATSF_RGB4326.png'
plt.savefig(output_path, dpi=300)
plt.show()

Original_RGB = atsfrgb(im_list[15]['band_data']['vv_band'], im_list[15]['band_data']['vh_band'])
fig, axes = plt.subplots()
axes.imshow(Original_RGB)
axes.set_title('Original_RGB')
plt.axis('on')
plt.tight_layout()
output_path = 'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/Original_RGB4326.png'
plt.savefig(output_path, dpi=300)
plt.show()

#%% ================== Display bmap on ATSF ========================

log_im_list=[]
for i in range(len(im_list_sub)):
    log_img = 10 * np.log10(im_list_sub[i]['band_data']['vv_band'])
    log_img = np.clip(log_img, -20, 0)
    log_img = ((log_img + 20) * (255 / 20)).astype(np.uint8)
    log_im_list.append(log_img)

for i in range(1,len(im_list_sub)):
    fig, ax = plt.subplots()
    ax.imshow(log_im_list[i], cmap='gray')
    # 1: positive definite difference (RED)
    # 2: negative definite difference (CYAN)
    # 3: indefinite difference        (YELLOW)
    colors = ['black', 'red', 'cyan', 'yellow']
    cmap_bmap = ListedColormap(colors)

    alpha_bmap = np.where(cmaps['band']['bmap'][timestamplist[i]] > 0, 0.5, 0) 
    ax.imshow(cmaps['band']['bmap'][timestamplist[i]], cmap=cmap_bmap, alpha=alpha_bmap)

    plt.tight_layout()
    plt.title('%s-%s' %(timestamplist[i-1], timestamplist[i]))
    output_path = '/Users/hoyeong/Documents/PYTHON/ChangeDetection/CDT/Image/Sub/%s_%s.png' %(timestamplist[i-1], timestamplist[i])
    plt.savefig(output_path, dpi=300)
    plt.show()


fig, ax = plt.subplots()
ax.imshow(log_ATSF, cmap='gray')
plt.tight_layout()
plt.title('ATSF')
output_path = 'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/Sub/ATSF_sub.png'
plt.savefig(output_path, dpi=300)
plt.show()


#%% =================================== Spatial Filtering =========================================

# Gamma map filter------------------------------
import spatial_filter.filters as sf

gamma_filtered = {}
gamma_filtered = sf.gamma_filter(im_list_sub[15],7)

gammamap = 10 * np.log10(gamma_filtered['vv_band'])
gammamap = np.clip(gammamap, -20, 0)
gammamap = ((gammamap + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(gammamap, cmap='gray')
axes.set_title('GAMMA Filter')
plt.axis('on')
plt.tight_layout()
# output_path = r'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/SpatialFiltered/Gammamap Filter.png'
# plt.savefig(output_path, dpi=300)
plt.show()

# Boxcar filter--------------------------------

boxcar_filtered = {}
boxcar_filtered = sf.boxcar(im_list_sub[15], 7)

boxcar = 10 * np.log10(boxcar_filtered['vv_band'])
boxcar = np.clip(boxcar, -20, 0)
boxcar = ((boxcar + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(boxcar, cmap='gray')
axes.set_title('BoxCar Filter')
plt.axis('on')
plt.tight_layout()
# output_path = r'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/SpatialFiltered/Boxcar Filter.png'
# plt.savefig(output_path, dpi=300)
plt.show()

# Lee filter-------------------------------------

lee_filtered = {}
lee_filtered = sf.leefilter(im_list_sub[15], 7)

lee = 10 * np.log10(lee_filtered['vv_band'])
lee = np.clip(lee, -20, 0)
lee = ((lee + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(lee, cmap='gray')
axes.set_title('Lee Filter')
plt.axis('on')
plt.tight_layout()
# output_path = r'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/SpatialFiltered/Lee Filter.png'
# plt.savefig(output_path, dpi=300)
plt.show()

# Frost filter-------------------------------------

frost_filtered = {}
frost_filtered['vv_band'] = sf.frost_filter(im_list_sub[-1]['band_data']['vv_band'],win_size=7)
frost_filtered['vh_band'] = sf.frost_filter(im_list_sub[-1]['band_data']['vh_band'],win_size=7)

frost = 10 * np.log10(frost_filtered['vv_band'])
frost = np.clip(frost, -20, 0)
frost = ((frost + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(frost, cmap='gray')
axes.set_title('Frost Filter')
plt.axis('on')
plt.tight_layout()
# output_path = r'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/SpatialFiltered/Frost Filter.png'
# plt.savefig(output_path, dpi=300)
plt.show()

# RefinedLee filter----------------------------------

leesigma_filtered = {}
leesigma_filtered = sf.leesigma(im_list_sub[15], 7)

leesigma = 10 * np.log10(leesigma_filtered['band_data']['vv_band'])
leesigma = np.clip(leesigma, -20, 0)
leesigma = ((leesigma + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(leesigma, cmap='gray')
axes.set_title('Lee Sigma Filter')
plt.axis('on')
plt.tight_layout()
# output_path = r'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/SpatialFiltered/Lee Sigma Filter.png'
# plt.savefig(output_path, dpi=300)
plt.show()

# 3D Mesh Grid---------------------------------------
def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def plot_3d_mesh(image, title):
    import os
    fig = plt.figure()
    X, Y = np.meshgrid(range(image.shape[1]), range(image.shape[0]))
    Z = image
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma', edgecolors='k', linewidth=0.1)
    ax.set_title(title)
    ax.set_zlim(0, 3)
    ax.view_init(elev=45, azim=225)
    plt.axis('on')
    plt.tight_layout()
    output_folder = r'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/SpatialFiltered/' 
    file_name = f"{title}_3D.png"
    output_path = os.path.join(output_folder, file_name)
    plt.subplots_adjust(bottom=0.07)
    plt.savefig(output_path, dpi=300)
    plt.show()

plot_3d_mesh(normalize_image(log_origin), "Original Image")
plot_3d_mesh(normalize_image(gammamap), "GAMMA Filter")
plot_3d_mesh(normalize_image(boxcar), "BoxCar Filter")
plot_3d_mesh(normalize_image(lee), "Lee Filter")
plot_3d_mesh(normalize_image(frost), "Frost Filter")
plot_3d_mesh(normalize_image(leesigma), "Lee sigma Filter")

#%% =================================== ATSF Thresholding =====================================

import atsf_threshold.atsf_spatial as atsf_th

atsf_thresholded = atsf_th.atsf_gamma_filter(result['avimg'], 
                                             7, 
                                             result['avimgcnt'],
                                             5)

# Gray Scale --------------------------
atsf_t_vv = 10 * np.log10(atsf_thresholded['vv_band'])  
atsf_t_vv = np.clip(atsf_t_vv, -20, 0)
atsf_t_vv = ((atsf_t_vv + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(atsf_t_vv, cmap='gray')
axes.set_title('ATSF Thresholded')
plt.axis('on')
plt.tight_layout()
output_path = '/Users/hoyeong/Documents/PYTHON/ChangeDetection/CDT/Image/ATSF_Thresholded_Farm.png'
plt.savefig(output_path, dpi=300)
plt.show()

# RGB Scale -----------------------------
Threshold_RGB = atsfrgb(atsf_thresholded['vv_band'], atsf_thresholded['vh_band'])
fig, axes = plt.subplots()
axes.imshow(Threshold_RGB)
axes.set_title('ATSF_Thresholded RGB')
plt.axis('on')
plt.tight_layout()
output_path = '/Users/hoyeong/Documents/PYTHON/ChangeDetection/CDT/Image/ATSF_Thresholded_RGB.png'
plt.savefig(output_path, dpi=300)
plt.show()

#%% ===================================== Evaluation ========================================

# PSNR -----------------------------

from skimage.metrics import peak_signal_noise_ratio
def psnr(reference, filtered):
    """
    Compute the PSNR between two SAR images.
    
    Args:
    image1: Reference Image in numpy array
    image2: Filtered Image in numpy array
    
    Returns:
    psnr_value: float, the PSNR value between reference and filtered
    """
    if reference.shape != filtered.shape:
        raise ValueError("Images must have the same dimensions")
        
    reference = reference.astype(np.float64)
    filtered = filtered.astype(np.float64)
    psnr_value = peak_signal_noise_ratio(reference, filtered, data_range=reference.max() - reference.min())
    return psnr_value


reference_image = im_list_sub[15]['band_data']['vv_band']
atsf_image = result['avimg']['band_data']['vv_band']
gamma_image = gamma_filtered['vv_band']
boxcar_image = boxcar_filtered['vv_band']
frost_image = frost_filtered['vv_band']
leesigma_image = leesigma_filtered['band_data']['vv_band']

print('############')
print('### PSNR')
print('############')

psnr_boxcar = psnr(reference_image, boxcar_image)
psnr_frost = psnr(reference_image, frost_image)
psnr_leesigma = psnr(reference_image, leesigma_image)
psnr_gamma = psnr(reference_image, gamma_image)
psnr_atsf = psnr(reference_image, atsf_image)

print("BOXCAR PSNR:", psnr_boxcar)
print("FROST PSNR:", psnr_frost)
print("LEESIGMA PSNR:", psnr_leesigma)
print("GAMMA PSNR:", psnr_gamma)
print("ATSF PSNR:", psnr_atsf)

#%% ATSF+RABASAR =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import copy
import glob
from RABASAR.rabasar.enl import (get_enl_img, 
                                 get_enl_mode, 
                                 get_enl_mask)
from RABASAR.rabasar.ratio_denoise import (admm_ratio_denoise)

def temporal_average(im_list):
    band_names = ['vv_band', 'vh_band']
    averaged = {'band_data':{}}
    for band in band_names:
        # Stack corresponding bands from all images
        stacked_bands = np.stack([img['band_data'][band] for img in im_list])
        # Calculate the mean along the new axis (axis=0) created by stacking
        averaged['band_data'][band] = np.mean(stacked_bands, axis=0)
    averaged['meta'] = {'driver': 'GTiff',
                         'dtype': 'float32',
                         'width': averaged['band_data']['vv_band'].shape[1],
                         'height':averaged['band_data']['vv_band'].shape[0],
                         'crs': im_list[0]['meta']['crs'],
                         'transform': im_list[0]['meta']['transform']}
    averaged['bounds'] = im_list[0]['bounds']
    averaged['name'] = 'Temporal Average'
    return averaged


def ratio_img(img, reference):
    '''
    It forms a ratio image for RABASAR
    
    Prameters:
    -----------
    img: Image with Speckle in certain time t
    refernce: SuperImage containing vv and vh band
    
    Return:
    -----------
    Ratio Image: img / reference
    '''
    ratio = copy.deepcopy(img)
    for band in ['vv_band', 'vh_band']:
        ratio['band_data'][band] = np.divide(img['band_data'][band], reference['band_data'][band])
    return ratio

superimage = temporal_average(im_list)

superimage_vv = 10 * np.log10(superimage['band_data']['vv_band'])
superimage_vv = np.clip(superimage_vv, -20, 0)
superimage_vv = ((superimage_vv + 20) * (255 / 20)).astype(np.uint8)

superimage_vh = 10 * np.log10(superimage['band_data']['vh_band'])
superimage_vh = np.clip(superimage_vh, -20, 0)
superimage_vh = ((superimage_vh + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(superimage_vv, cmap='gray')
axes.set_title('SuperImage')
plt.axis('on')
plt.tight_layout()
# output_path = '/Users/hoyeong/Documents/PYTHON/ChangeDetection/CDT/RABASAR/Image/SuperImage.png'
# plt.savefig(output_path, dpi=300)
plt.show()

# ------------------------------------------------------

ratio_image_mean = ratio_img(im_list[-1], superimage)

ratio_vv_mean = 10 * np.log10(ratio_image_mean['band_data']['vv_band'])
ratio_vv_mean = np.clip(ratio_vv_mean, -20, 0)
ratio_vv_mean = ((ratio_vv_mean + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(ratio_vv_mean, cmap='gray')
axes.set_title('Ratio Image using MEAN')
plt.axis('on')
plt.tight_layout()
# output_path = r'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/ratio_image_filtered_mean.png'
# plt.savefig(output_path, dpi=300)
plt.show()

# -------------------------------------------------------

ratio_image_atsf = ratio_img(im_list[-1], result['avimg'])

ratio_vv_atsf = 10 * np.log10(ratio_image_atsf['band_data']['vv_band'])
ratio_vv_atsf = np.clip(ratio_vv_atsf, -20, 0)
ratio_vv_atsf = ((ratio_vv_atsf + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(ratio_vv_atsf, cmap='gray')
axes.set_title('Ratio Image using ATSF')
plt.axis('on')
plt.tight_layout()
output_path = r'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/ratio_image_atsf.png'
plt.savefig(output_path, dpi=300)
plt.show()

#%% ---------------------------------------------------

enl_mask_mean = get_enl_mask(im_list[-1]['band_data']['vv_band'], db_min=-13)
enl_img_mean = get_enl_img(im_list[-1]['band_data']['vv_band'], 31, enl_max=20, mask=enl_mask_mean)
L_mean = round(get_enl_mode(enl_img_mean))

enl_ref_img_mean = get_enl_img(superimage['band_data']['vv_band'], 31, enl_max=1000, mask=enl_mask_mean)
Lm_mean = round(get_enl_mode(enl_ref_img_mean))

print(L_mean,Lm_mean)

plt.hist(enl_ref_img_mean[~np.isnan(enl_ref_img_mean)], bins=50)
plt.xlabel('ENL')
plt.ylabel('frequency')
plt.show()
# ------------------------

enl_mask_atsf = get_enl_mask(im_list[-1]['band_data']['vv_band'], db_min=-13)
enl_img_atsf = get_enl_img(im_list[-1]['band_data']['vv_band'], 31, enl_max=20, mask=enl_mask_atsf)
L_atsf = round(get_enl_mode(enl_img_atsf))

enl_ref_img_atsf = get_enl_img(superimage['band_data']['vv_band'], 31, enl_max=1000, mask=enl_mask_atsf)
Lm_atsf = round(get_enl_mode(enl_ref_img_atsf))

print(L_atsf,Lm_atsf)

plt.hist(enl_ref_img_atsf[~np.isnan(enl_ref_img_atsf)], bins=50)
plt.xlabel('ENL')
plt.ylabel('frequency')
plt.show()

#%% Mean

ratio_filtered_mean, res_list_mean = admm_ratio_denoise(ratio_image_mean['band_data']['vv_band'],
                                                        L_mean,
                                                        Lm_mean,
                                                        regularizer='bm3d',
                                                        regularizer_params={'weight':0.01},
                                                        max_admm_iterations=10,
                                                        newton_iterations=3)


ratio_fil_mean = 10 * np.log10(ratio_filtered_mean)
ratio_fil_mean = np.clip(ratio_fil_mean, -20, 0)
ratio_fil_mean = ((ratio_fil_mean + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(ratio_fil_mean, cmap='gray')
axes.set_title('Denoised Ratio Image_MEAN')
plt.axis('on')
plt.tight_layout()
output_path = r'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/Ratio_filtered_mean.png'
plt.savefig(output_path, dpi=300)
plt.show()

# ATSF

ratio_filtered_atsf, res_list_atsf = admm_ratio_denoise(ratio_image_atsf['band_data']['vv_band'],
                                                        L_atsf,
                                                        Lm_atsf,
                                                        regularizer='bm3d',
                                                        regularizer_params={'weight':0.01},
                                                        max_admm_iterations=10,
                                                        newton_iterations=3)


ratio_fil_atsf = 10 * np.log10(ratio_filtered_atsf)
ratio_fil_atsf = np.clip(ratio_fil_atsf, -20, 0)
ratio_fil_atsf = ((ratio_fil_atsf + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(ratio_fil_atsf, cmap='gray')
axes.set_title('Denoised Ratio Image_ATSF')
plt.axis('on')
plt.tight_layout()
output_path = r'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/Ratio_filtered_atsf.png'
plt.savefig(output_path, dpi=300)
plt.show()

#%%

final_img_atsf = ratio_filtered_atsf*result['avimg']['band_data']['vv_band']

final_vv_atsf = 10 * np.log10(final_img_atsf)
final_vv_atsf = np.clip(final_vv_atsf, -20, 0)
final_vv_atsf = ((final_vv_atsf + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(final_vv_atsf, cmap='gray')
axes.set_title('RABASAR_ATSF')
plt.axis('on')
plt.tight_layout()
output_path = r'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/RABASAR_ATSF.png'
plt.savefig(output_path, dpi=300)
plt.show()

orignal_vv = 10 * np.log10(im_list[-1]['band_data']['vv_band'])
orignal_vv = np.clip(orignal_vv, -20, 0)
orignal_vv = ((orignal_vv + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(orignal_vv, cmap='gray')
axes.set_title('Original Image')
plt.axis('on')
plt.tight_layout()
output_path = r'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/Original Image.png'
plt.savefig(output_path, dpi=300)
plt.show()