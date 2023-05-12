'''
#############
## RABASAR ##
#############

author: @hoyeong
'''

#%% Import Library --------------------------------

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import copy
import glob
import cd
from tqdm import tqdm
from RABASAR.rabasar.enl import (get_enl_img, 
                         get_enl_mode, 
                         get_enl_mask)
from RABASAR.rabasar.ratio_denoise import (admm_ratio_denoise)

#%% Functions ------------------------------------

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


#%% Read Files --------------------------------------

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

im_list = cd.match(im_list)

#%% Average Image-----------------------------------

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

#%% Making Ratio Image-------------------------------

ratio_image = ratio_img(im_list[15], superimage)

ratio_vv = 10 * np.log10(ratio_image['band_data']['vv_band'])
ratio_vv = np.clip(ratio_vv, -20, 0)
ratio_vv = ((ratio_vv + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(ratio_vv, cmap='gray')
axes.set_title('Ratio Image VV')
plt.axis('on')
plt.tight_layout()
output_path = r'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/ratio_image_filtered_15index.png'
plt.savefig(output_path, dpi=300)
plt.show()

#%% ENL Calculation -----------------------------

enl_mask = get_enl_mask(im_list[0]['band_data']['vv_band'], db_min=-13)
enl_img = get_enl_img(im_list[0]['band_data']['vv_band'], 31, enl_max=20, mask=enl_mask)
L = round(get_enl_mode(enl_img))

enl_ref_img = get_enl_img(superimage['band_data']['vv_band'], 31, enl_max=1000, mask=enl_mask)
Lm = round(get_enl_mode(enl_ref_img))

print(L,Lm)

plt.hist(enl_ref_img[~np.isnan(enl_ref_img)], bins=50)
plt.xlabel('ENL')
plt.ylabel('frequency')
plt.show()

#%% Filtering Ratio Image--------------------------

ratio_filtered, res_list = admm_ratio_denoise(ratio_image['band_data']['vv_band'],
                                              L,
                                              Lm,
                                              regularizer='bm3d',
                                              regularizer_params={'weight':0.1},
                                              max_admm_iterations=10,
                                              newton_iterations=3)


ratio_fil = 10 * np.log10(ratio_filtered)
ratio_fil = np.clip(ratio_fil, -20, 0)
ratio_fil = ((ratio_fil + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(ratio_fil, cmap='gray')
axes.set_title('Denoised Ratio Image')
plt.axis('on')
plt.tight_layout()
output_path = r'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/Ratio_filtered.png'
plt.savefig(output_path, dpi=300)
plt.show()

#%% Final Result-----------------------------------

final_img = ratio_filtered*superimage['band_data']['vv_band']

final_vv = 10 * np.log10(final_img)
final_vv = np.clip(final_vv, -20, 0)
final_vv = ((final_vv + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(final_vv, cmap='gray')
axes.set_title('RABASAR_weight_0.10 Image')
plt.axis('on')
plt.tight_layout()
output_path = r'A:/PYTHON/ChangeDetection/CDT/hyyou/Image/RABASAR_weight1.png'
plt.savefig(output_path, dpi=300)
plt.show()

orignal_vv = 10 * np.log10(im_list[0]['band_data']['vv_band'])
orignal_vv = np.clip(orignal_vv, -20, 0)
orignal_vv = ((orignal_vv + 20) * (255 / 20)).astype(np.uint8)

fig, axes = plt.subplots()
axes.imshow(orignal_vv, cmap='gray')
axes.set_title('Original Image')
plt.axis('on')
plt.tight_layout()
output_path = '/Users/hoyeong/Documents/PYTHON/ChangeDetection/CDT/RABASAR/Image/Original Image.png'
plt.savefig(output_path, dpi=300)
plt.show()