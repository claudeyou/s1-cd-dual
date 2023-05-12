import numpy as np

def atsf_gamma_filter(image, kernel_size, avimgcnt, threshold):
    enl = 4.4
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

        filtered_band = z_hat + r_hat + x
        return np.where(avimgcnt < threshold, filtered_band, img_band)

    filtered_bands = {}
    for band_name in band_names:
        filtered_bands[band_name] = apply_gammamap(band_name)
    
    return filtered_bands
