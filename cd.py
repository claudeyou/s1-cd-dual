"""
Created on Fri Mar 24 18:41:10 2023

@author: hoyeong
@Python: 3.9.16v
"""
#%% Library --------------------------------------
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


#%% Functions --------------------------------------
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