import numpy as np
from tqdm import tqdm
from scipy.ndimage import uniform_filter, convolve, generic_filter
from scipy.stats import variation

#%=================Gamma map===================
def gamma_filter(image, kernel_size):
    """
    Gamma Maximum a-posterior Filter applied to one image. It is implemented as described in 
    Lopes A., Nezry, E., Touzi, R., and Laur, H., 1990.  
    Maximum A Posteriori Speckle Filtering and First Order texture Models in SAR Images.  
    International  Geoscience  and  Remote  Sensing  Symposium (IGARSS).
    Parameters
    ----------
    image : numpy array
        Image to be filtered
    KERNEL_SIZE : positive odd integer
        Neighbourhood window size
    Returns
    -------
        Filtered Image
    """
    enl = 5
    band_names = ['vv_band', 'vh_band']
    
    def get_stats(img, kernel_size):
        mean = np.empty(img.shape)
        std_dev = np.empty(img.shape)

        pad_size = kernel_size // 2
        padded_img = np.pad(img, pad_size, mode='reflect')

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                neighborhood = padded_img[i:i + kernel_size, j:j + kernel_size]
                mean[i, j] = np.mean(neighborhood)
                std_dev[i, j] = np.std(neighborhood)

        return mean, std_dev
    
    def apply_gammamap(band_name):
        img_band = image['band_data'][band_name]
        z, sigz = get_stats(img_band, kernel_size)

        ci = sigz / z
        cu = 1.0 / np.sqrt(enl)
        cmax = np.sqrt(2.0) * cu
        
        alpha = (1 + cu ** 2) / (ci ** 2 - cu ** 2)

        q = np.maximum(0, z ** 2 * (z * alpha - enl - 1) ** 2 + 4 * alpha * enl * img_band * z)
        r_hat = (z * (alpha - enl - 1) + np.sqrt(q)) / (2 * alpha)

        z_hat = np.where(ci <= cu, z, 0)
        r_hat = np.where((ci > cu) & (ci < cmax), r_hat, 0)
        x = np.where(ci >= cmax, img_band, 0)

        return z_hat + r_hat + x

    filtered_bands = {}
    for band_name in band_names:
        filtered_bands[band_name] = apply_gammamap(band_name)
    
    return filtered_bands

#%======================Boxcar=======================

def boxcar(image, kernel_size):
    """
    Apply boxcar filter on every image in the collection.
    Parameters
    ----------
    image : dict
        Image to be filtered
    kernel_size : positive odd integer
        Neighborhood window size
    Returns
    -------
    dict
        Filtered Image
    """
    # Get the band data
    vv_band = image['band_data']['vv_band']
    vh_band = image['band_data']['vh_band']
    
    # Apply uniform filter
    vv_filtered = uniform_filter(vv_band, size=kernel_size)
    vh_filtered = uniform_filter(vh_band, size=kernel_size)
    
    # Update the band data in the image
    filtered_bands = {}
    filtered_bands['vv_band'] = vv_filtered
    filtered_bands['vh_band'] = vh_filtered
    return filtered_bands


#%======================RefinedLEE=====================


def refined_lee(image, kernel_size=7):
    vv_band = image['band_data']['vv_band']
    vh_band = image['band_data']['vh_band']

    def apply_filter(band_data, kernel_size):
        kernel = np.ones((kernel_size, kernel_size))

        mean = uniform_filter(band_data, size=(kernel_size, kernel_size))
        variance = generic_filter(band_data, np.var, size=(kernel_size, kernel_size))

        sample_weights = np.zeros((kernel_size, kernel_size))
        sample_weights[kernel_size//2::2, :] = 1

        sample_mean = convolve(mean, sample_weights)
        sample_var = generic_filter(variance, np.var, footprint=sample_weights)
        
        # Compute gradients and directions
        gradients = np.abs(sample_mean - np.roll(sample_mean, shift=1, axis=0))
        gradients += np.abs(sample_mean - np.roll(sample_mean, shift=1, axis=1))
        max_gradient = np.max(gradients, axis=(0, 1))

        # Implement the rest of the algorithm using the provided steps
        rect_kernel = np.ones((kernel_size, kernel_size))
        rect_kernel[:kernel_size//2, :] = 0
        rect_kernel[-kernel_size//2:, :] = 0

        diag_kernel = np.eye(kernel_size)
        np.fill_diagonal(diag_kernel[:, ::-1], 1)

        dir_mean_rect = convolve(mean, rect_kernel)
        dir_var_rect = convolve(variance, rect_kernel)

        dir_mean_diag = convolve(mean, diag_kernel)
        dir_var_diag = convolve(variance, diag_kernel)

        dir_mean = np.where(dir_mean_rect > dir_mean_diag, dir_mean_rect, dir_mean_diag)
        dir_var = np.where(dir_var_rect > dir_var_diag, dir_var_rect, dir_var_diag)

        sigma_v = np.sort(sample_var.ravel())[:5].mean()

        var_x = (dir_var - dir_mean * dir_mean * sigma_v) / (sigma_v + 1.0)
        b = var_x / dir_var
        result = dir_mean + b * (band_data - dir_mean)

        return result

    vv_filtered = apply_filter(vv_band, kernel_size)
    vh_filtered = apply_filter(vh_band, kernel_size)
    return {'vv_band': vv_filtered, 'vh_band': vh_filtered}


#%======================lee filter=====================

def leefilter(image, KERNEL_SIZE):
    """
    Lee Filter applied to one image.
    Parameters
    ----------
    image : dict
        Image dictionary containing band data as NumPy arrays
    KERNEL_SIZE : positive odd integer
        Neighbourhood window size
    Returns
    -------
    dict
        Filtered Image dictionary containing band data as NumPy arrays
    """

    # S1-GRD images are multilooked 5 times in range
    enl = 4.4
    # Compute the speckle standard deviation
    eta = 1.0 / np.sqrt(enl)

    output_bands = {}

    for band_name, band_data in image['band_data'].items():
        # Estimate mean and variance using a local neighborhood window
        mean = uniform_filter(band_data, size=KERNEL_SIZE)
        var = uniform_filter(band_data**2, size=KERNEL_SIZE) - mean**2

        # Estimate weight
        varx = (var - mean**2 * eta**2) / (1 + eta**2)
        b = varx / var

        # If b is negative, set it to zero
        b[b < 0] = 0
        output_band = (1 - b) * np.abs(mean) + b * band_data
        output_bands[band_name] = output_band

    return output_bands

#%======================Frost filter=====================

def frost_filter(img, damping_factor=2.0, win_size=3):
    img_filtered = np.zeros_like(img)
    N, M = img.shape
    win_offset = win_size // 2

    for i in range(N):
        for j in range(M):
            x_start = max(0, i - win_offset)
            x_end = min(N, i + win_offset + 1)
            y_start = max(0, j - win_offset)
            y_end = min(M, j + win_offset + 1)

            window = img[x_start:x_end, y_start:y_end]
            coef_var = variation(window, None)
            window_mean = window.mean()
            sigma_zero = coef_var / (window_mean + 1e-7)  # To avoid division by zero
            factor_A = damping_factor * sigma_zero

            weights_array = np.exp(-factor_A * np.abs(window - window.mean()))
            img_filtered[i, j] = np.sum(weights_array * window) / np.sum(weights_array)

    return img_filtered


#%======================Lee Sigma======================


def leesigma(image, KERNEL_SIZE):
    Tk = 7
    sigma = 0.9
    enl = 4
    target_kernel = 3

    bandNames = list(image['band_data'].keys())

    # Helper function for percentile
    def percentile(arr, q):
        return np.percentile(arr, q)

    # Compute the 98 percentile intensity
    z98 = {band: percentile(image['band_data'][band], 98) for band in bandNames}

    # Select the strong scatterers to retain
    brightPixel = {band: image['band_data'][band] >= z98[band] for band in bandNames}
    K = {band: uniform_filter(brightPixel[band].astype(int), target_kernel) for band in bandNames}
    retainPixel = {band: K[band] >= Tk for band in bandNames}

    # Compute the a-priori mean within a 3x3 local window
    eta = 1.0 / np.sqrt(enl)
    eta_img = {band: np.full_like(image['band_data'][band], eta) for band in bandNames}

    mean = {band: uniform_filter(image['band_data'][band], target_kernel) for band in bandNames}
    var = {band: uniform_filter(image['band_data'][band]**2, target_kernel) - mean[band]**2 for band in bandNames}

    varx = {band: (var[band] - np.abs(mean[band])**2 * eta_img[band]**2) / (1 + eta_img[band]**2) for band in bandNames}
    b = {band: varx[band] / var[band] for band in bandNames}
    xTilde = {band: (1 - b[band]) * np.abs(mean[band]) + b[band] * image['band_data'][band] for band in bandNames}

    # LUT for sigma range
    LUT = {0.5: {'I1': 0.694, 'I2': 1.385, 'eta': 0.1921},
           0.6: {'I1': 0.630, 'I2': 1.495, 'eta': 0.2348},
           0.7: {'I1': 0.560, 'I2': 1.627, 'eta': 0.2825},
           0.8: {'I1': 0.480, 'I2': 1.804, 'eta': 0.3354},
           0.9: {'I1': 0.378, 'I2': 2.094, 'eta': 0.3991},
           0.95: {'I1': 0.302, 'I2': 2.360, 'eta': 0.4391}}

    sigma_data = LUT[sigma]
    I1 = {band: sigma_data['I1'] * xTilde[band] for band in bandNames}
    I2 = {band: sigma_data['I2'] * xTilde[band] for band in bandNames}
    nEta = sigma_data['eta']
    # Apply MMSE filter for pixels in the sigma range
    mask = {band: np.logical_or(image['band_data'][band] >= I1[band], image['band_data'][band] <= I2[band]) for band in bandNames}
    z = {band: np.where(mask[band], image['band_data'][band], np.nan) for band in bandNames}

    mean_z = {band: uniform_filter(np.nan_to_num(z[band]), KERNEL_SIZE) for band in bandNames}
    var_z = {band: uniform_filter(np.nan_to_num(z[band])**2, KERNEL_SIZE) - mean_z[band]**2 for band in bandNames}

    varx_z = {band: (var_z[band] - np.abs(mean_z[band])**2 * nEta**2) / (1 + nEta**2) for band in bandNames}
    b_z = {band: varx_z[band] / var_z[band] for band in bandNames}
    new_b = {band: np.where(b_z[band] < 0, 0, b_z[band]) for band in bandNames}
    xHat = {band: (1 - new_b[band]) * np.abs(mean_z[band]) + new_b[band] * z[band] for band in bandNames}

    # Remove the applied masks and merge the retained pixels and the filtered pixels
    xHat_merged = {band: np.where(retainPixel[band], image['band_data'][band], xHat[band]) for band in bandNames}
    output = {'band_data': xHat_merged}
    return output


