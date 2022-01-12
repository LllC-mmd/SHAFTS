import numpy as np
import cv2
import skimage.morphology as skmorph
import skimage.feature as skfeature
import skimage.util as sku

import multiprocessing
# import matplotlib.pyplot as plt


def get_confusion_matrix(val_true, val_pred, num_quantile=20, normed=False):
    q_list = np.linspace(0.0, 1.0, num_quantile+1)
    q_val_list = [np.quantile(val_true, q) for q in q_list]

    # ---convert the continuous height value into categorical variable
    # ------to achieve this, we divide the target interval as follows:
    # ---------label 0: (-\infty, q0), label 1: [q0, q1), ..., label-N: [q_{N-1}, qN), label-N+1: [qN, +\infty)
    # ------Thus, we actually have (N+2) intervels
    val_true_categorical = np.zeros_like(val_true)
    val_pred_categorical = np.zeros_like(val_pred)
    for i in range(0, num_quantile):
        tmp_true_test = np.logical_and(val_true >= q_val_list[i], val_true < q_val_list[i+1])
        tmp_pred_test = np.logical_and(val_pred >= q_val_list[i], val_pred < q_val_list[i+1])
        val_true_categorical[tmp_true_test] = i+1
        val_pred_categorical[tmp_pred_test] = i+1

    val_true_categorical = np.where(val_true >= q_val_list[-1], num_quantile+1, val_true_categorical)
    val_pred_categorical = np.where(val_pred < q_val_list[0], 0, val_pred_categorical)
    val_pred_categorical = np.where(val_pred >= q_val_list[-1], num_quantile+1, val_pred_categorical)

    # ---generate the confusion matrix
    num_interval = num_quantile + 2
    label = num_interval * val_true_categorical + val_pred_categorical
    count = np.bincount(label.astype(np.int), minlength=num_interval**2)
    # ------the i-th row and j-th column of the confusion matrix is the amount of label i predicted as j
    confusion_matrix = np.reshape(count, (num_interval, num_interval))

    if normed:
        c_sum = np.reshape(np.sum(confusion_matrix, axis=1), (num_interval, 1))
        c_sum = np.concatenate([c_sum for i in range(0, num_interval)], axis=1)
        confusion_matrix = confusion_matrix / c_sum
        confusion_matrix = np.where(np.isnan(confusion_matrix), 0.0, confusion_matrix)

    return confusion_matrix


# ---instance normalization, see:
# ------Ulyanov, D., Vedaldi, A., & Lempitsky, V. S. (2016).
# ------Instance Normalization: The Missing Ingredient for Fast Stylization. ArXiv Preprint ArXiv:1607.08022.
def rgb_rescale(dta, q1=0.98, q2=0.02, vmin=0.0, vmax=1.0, axis=(-1, -2)):
    val_min = np.min(dta, axis=axis)
    val_min = np.expand_dims(val_min, axis=axis)

    val_high = np.quantile(dta, q1, axis=axis)
    val_high = np.expand_dims(val_high, axis=axis)
    val_low = np.quantile(dta, q2, axis=axis)
    val_low = np.expand_dims(val_low, axis=axis)
    dta_rescale = (dta - val_min) * (vmax - vmin) / (val_high - val_low) + vmin

    dta_clipped = np.clip(dta_rescale, vmin, vmax)
    # ------set NaN value to be vmin
    dta_clipped[~np.isfinite(dta_clipped)] = vmin

    return dta_clipped


def rgb_rescale_band(dta, q1=0.98, q2=0.02, vmin=0.0, vmax=1.0):
    dta_1d = dta.flatten()
    dta_1d_loc = ~np.isnan(dta_1d)
    dta_1d_tmp = dta_1d[dta_1d_loc]

    val_min = np.min(dta_1d_tmp)

    val_high = np.quantile(dta_1d_tmp, q1)
    val_low = np.quantile(dta_1d_tmp, q2)

    dta_rescale = (dta - val_min) * (vmax - vmin) / (val_high - val_low) + vmin

    dta_clipped = np.clip(dta_rescale, vmin, vmax)

    return dta_clipped


def get_backscatterCoef(raw_s1_dta):
    coef = np.power(10.0, raw_s1_dta / 10.0)
    coef = np.where(coef > 1.0, 1.0, coef)
    return coef


def get_VVH(vv_coef, vh_coef, gamma=5.0):
    return vv_coef * np.power(gamma, vh_coef)


def get_luminance(red, green, blue, reduced=True):
    # ref to: https://en.wikipedia.org/wiki/Relative_luminance
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    if reduced:
        luminance = np.mean(luminance, axis=(-1, -2))
    return luminance


def get_normalized_index(x, y, reduced=True):
    n_id = (x - y) / (x + y)
    if reduced:
        n_id = np.mean(n_id, axis=(-1, -2))
    n_id = np.where(np.isnan(n_id), 0.0, n_id)
    return n_id


# ************************* Raw Band Statistics *************************
def get_band_statistics_func(method_list):
    band_stat_ref = {"Mean": lambda x: np.mean(x, axis=(-1, -2)), "Std": lambda x: np.std(x, axis=(-1, -2)),
                     "Max": lambda x: np.max(x, axis=(-1, -2)), "Min": lambda x: np.min(x, axis=(-1, -2)),
                     "50pt": lambda x: np.median(x, axis=(-1, -2)), "25pt": lambda x: np.quantile(x, 0.25, axis=(-1, -2)),
                     "75pt": lambda x: np.quantile(x, 0.75, axis=(-1, -2))}

    stat_dict = {}
    for m in method_list:
        if m in band_stat_ref.keys():
            stat_dict[m] = band_stat_ref[m]
        else:
            raise NotImplementedError("Unknown Statistics")

    return stat_dict


def get_band_statistics(method_list):
    pass


# ************************* Morphological Features based on Morphological Profiles (MP) *************************
# ---Ref to: Pesaresi, M., & Benediktsson, J. A. (2001).
# ------A new approach for the morphological segmentation of high-resolution satellite imagery.
# ------IEEE Transactions on Geoscience and Remote Sensing, 39(2), 309–320.
def get_opening_img(img, SE_size):
    if SE_size == 0:
        return img
    else:
        neighbor = np.zeros((SE_size, SE_size))
        center = int((SE_size-1)/2)
        neighbor[center, :] = 1
        neighbor[:, center] = 1
        return skmorph.opening(img, selem=neighbor)


def get_opening_img_cv(img, SE_size):
    if SE_size == 0:
        return img
    else:
        kernel = np.ones((SE_size, SE_size), dtype=np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def get_opening_reconstruction_img(img, SE_size):
    if SE_size == 0:
        img_erosion = img
    else:
        neighbor = np.zeros((SE_size, SE_size))
        center = int((SE_size - 1) / 2)
        neighbor[center, :] = 1
        neighbor[:, center] = 1
        img_erosion = skmorph.erosion(img, selem=neighbor)

    img_rec = skmorph.reconstruction(seed=img_erosion, mask=img, selem=np.ones((SE_size, SE_size)), method="dilation")
    return img_rec


# ---reconstruction of the erosion (marker) under the original image (mask)
def get_opening_reconstruction_img_cv(img, SE_size, max_iter=10000):
    if SE_size == 0:
        return img
    else:
        # ------set the image after erosion as the marker
        kernel = np.ones((SE_size, SE_size), dtype=np.uint8)
        img_marker = cv2.erode(img, kernel)

        c = 0
        while True:
            img_rec = cv2.dilate(img_marker, kernel)
            img_rec = np.where(img < img_rec, img, img_rec)

            if np.all(img_marker == img_rec) or (c >= max_iter):
                break

            img_marker = img_rec
            c += 1

        return img_rec


def get_closing_img(img, SE_size):
    if SE_size == 0:
        return img
    else:
        neighbor = np.zeros((SE_size, SE_size))
        center = int((SE_size-1)/2)
        neighbor[center, :] = 1
        neighbor[:, center] = 1
        return skmorph.closing(img, selem=neighbor)


def get_closing_img_cv(img, SE_size):
    if SE_size == 0:
        return img
    else:
        kernel = np.ones((SE_size, SE_size), dtype=np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def get_closing_reconstruction_img(img, SE_size):
    if SE_size == 0:
        img_dilation = img
    else:
        neighbor = np.zeros((SE_size, SE_size))
        center = int((SE_size - 1) / 2)
        neighbor[center, :] = 1
        neighbor[:, center] = 1
        img_dilation = skmorph.dilation(img, selem=neighbor)

    img_rec = skmorph.reconstruction(seed=img_dilation, mask=img, selem=np.ones((SE_size, SE_size)), method="erosion")
    return img_rec


# ---dual reconstruction of the dilation (marker) above the original image (mask)
def get_closing_reconstruction_img_cv(img, SE_size, max_iter=10000):
    if SE_size == 0:
        return img
    else:
        # ------set the image after dilation as the marker
        kernel = np.ones((SE_size, SE_size), dtype=np.uint8)
        img_marker = cv2.dilate(img, kernel)

        c = 0
        while True:
            img_rec = cv2.erode(img_marker, kernel)
            img_rec = np.where(img > img_rec, img, img_rec)

            if np.all(img_marker == img_rec) or (c >= max_iter):
                break

            img_marker = img_rec
            c += 1

        return img_rec


# ---calculated the mean of Differential Morphological Profiles
def get_DMP_mean(img, SE_size_1, SE_size_2, method):
    # ---implementation based on Scikit-image
    '''
    morph_func_ref = {"Opening": get_opening_img, "OpeningByReconstruction": get_opening_reconstruction_img,
                      "Closing": get_closing_img, "ClosingByReconstruction": get_closing_reconstruction_img}
    '''
    # ---implementation based on OpenCV
    morph_func_ref = {"Opening": get_opening_img_cv, "OpeningByReconstruction": get_opening_reconstruction_img_cv,
                      "Closing": get_closing_img_cv, "ClosingByReconstruction": get_closing_reconstruction_img_cv}

    if method in morph_func_ref.keys():
        morph_func = morph_func_ref[method]
        img_1 = morph_func(img, SE_size_1)
        img_2 = morph_func(img, SE_size_2)
    else:
        raise NotImplementedError("Unknown Morphological Operator")

    img_diff = np.mean(np.abs(img_1 - img_2))

    return img_diff


def get_DMP_mean_batch(img_list, size_list, method, num_cpu=None):
    # ------if num_cpu is specified, use multiprocessing
    if num_cpu is not None and num_cpu != 1:
        # ------check available CPUs
        num_cpu_available = multiprocessing.cpu_count()
        if num_cpu_available < num_cpu:
            print("Use %d CPUs which is available now" % num_cpu_available)
            num_cpu = num_cpu_available

        pool = multiprocessing.Pool(processes=num_cpu)
        num_cpu = min(num_cpu, len(img_list))
        img_sub = np.array_split(img_list, num_cpu)
        arg_list = [(img_sub[i], size_list, method) for i in range(0, num_cpu)]
        res_list = pool.starmap(get_DMP_mean_batch, arg_list)
        pool.close()
        pool.join()

        DMP_list = np.concatenate(res_list, axis=-1)
    # ------if num_cpu is not specified, use a direct iteration
    else:
        n_pair = len(size_list) - 1
        DMP_list = [get_DMP_mean(img=img, SE_size_1=size_list[i], SE_size_2=size_list[i+1], method=method)
                     for i in range(0, n_pair) for img in img_list]
        DMP_list = np.reshape(DMP_list, (n_pair, -1))
    return DMP_list


# ************************* Texture Features based on Grey-Level Co-occurrence Matrix (GLCM) *************************
# ---Ref to: Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973).
# ------Textural Features for Image Classification.
# ------IEEE Transactions on Systems Man and Cybernetics, 3(6), 610–621.
def get_avg_GLCM(img, window_length, angle, bin_width=32, normed=True):
    center = int((img.shape[0] - 1) / 2)
    l = int((window_length - 1) / 2)
    img_sample = img[center-l:center+l+1, center-l:center+l+1]
    img_sample = sku.img_as_ubyte(np.clip(img_sample, 0, 1)) // bin_width

    glcm = skfeature.texture.greycomatrix(image=img_sample, distances=[1], angles=angle, levels=256//bin_width)
    glcm_avg = np.sum(glcm[:, :, 0, :], axis=-1)

    if normed:
        glcm_avg = glcm_avg / np.sum(glcm_avg)

    return glcm_avg


def get_avg_GLCM_batch(img_list, window_length, angle, normed=True, num_cpu=None):
    # ------if num_cpu is specified, use multiprocessing
    if num_cpu is not None and num_cpu != 1:
        # ------check available CPUs
        num_cpu_available = multiprocessing.cpu_count()
        if num_cpu_available < num_cpu:
            print("Use %d CPUs which is available now" % num_cpu_available)
            num_cpu = num_cpu_available

        pool = multiprocessing.Pool(processes=num_cpu)
        num_cpu = min(num_cpu, len(img_list))
        img_sub = np.array_split(img_list, num_cpu)
        arg_list = [(img_sub[i], window_length, angle, True) for i in range(0, num_cpu)]
        res_list = pool.starmap(get_avg_GLCM_batch, arg_list)
        pool.close()
        pool.join()

        glcm_list = np.concatenate(res_list)
    # ------if num_cpu is not specified, use a direct iteration
    else:
        glcm_list = np.array([get_avg_GLCM(img=img, window_length=window_length, angle=angle, normed=normed) for img in img_list])

    return glcm_list


def get_GLCM_mean(glcm_list):
    return np.mean(glcm_list, axis=(-1, -2))


def get_GLCM_variance(glcm_list):
    return np.var(glcm_list, axis=(-1, -2))


def get_GLCM_homogeneity(glcm_list):
    h, w = glcm_list[0].shape
    i_mat = np.vstack([np.full((1, w), h_i) for h_i in range(0, h)])
    j_mat = np.hstack([np.full((h, 1), w_i) for w_i in range(0, w)])

    homogeneity = np.sum(glcm_list / (1.0 + (i_mat - j_mat)**2), axis=(-1, -2))

    return homogeneity


def get_GLCM_contrast(glcm_list):
    h, w = glcm_list[0].shape
    i_mat = np.vstack([np.full((1, w), h_i) for h_i in range(0, h)])
    j_mat = np.hstack([np.full((h, 1), w_i) for w_i in range(0, w)])

    contrast = np.sum(glcm_list * (i_mat - j_mat)**2, axis=(-1, -2))

    return contrast


def get_GLCM_dissimilarity(glcm_list):
    h, w = glcm_list[0].shape
    i_mat = np.vstack([np.full((1, w), h_i) for h_i in range(0, h)])
    j_mat = np.hstack([np.full((h, 1), w_i) for w_i in range(0, w)])

    dissimilarity = np.sum(glcm_list * np.abs(i_mat - j_mat), axis=(-1, -2))

    return dissimilarity


def get_GLCM_entropy(glcm_list):
    glcm_lnx = - glcm_list * np.log(glcm_list)
    # by convention, we have 0 * log(0) = 0, which is assigned to be NaN in numpy
    glcm_lnx = np.where(np.isnan(glcm_lnx), 0.0, glcm_lnx)
    entropy = np.sum(glcm_lnx, axis=(-1, -2))

    return entropy


def get_GLCM_2nd_moment(glcm_list):
    m = np.sum(np.square(glcm_list), axis=(-1, -2))
    return m


def get_GLCM_correlation(glcm_list):
    h, w = glcm_list[0].shape  # h = w = n_level

    i_sample = np.vstack([np.linspace(0, h-1, h) for n in range(0, len(glcm_list))])  # shape(i_sample) = (n_matrix, n_level)
    i_prob = np.sum(glcm_list, axis=-1)   # shape(i_prob) = (n_matrix, n_level)
    i_mean = np.average(i_sample, weights=i_prob, axis=-1)  # shape(i_mean) = (n_matrix, )
    i_std = np.average((i_sample - np.reshape(i_mean, (-1, 1)))**2, weights=i_prob, axis=-1)  # shape(i_std) = (n_matrix, )

    j_sample = np.vstack([np.linspace(0, w-1, w) for n in range(0, len(glcm_list))])
    j_prob = np.sum(glcm_list, axis=-2)
    j_mean = np.average(j_sample, weights=j_prob, axis=-1)
    j_std = np.average((j_sample - np.reshape(j_mean, (-1, 1)))**2, weights=j_prob, axis=-1)

    i_mat = np.vstack([np.full((1, w), h_i) for h_i in range(0, h)])
    j_mat = np.hstack([np.full((h, 1), w_i) for w_i in range(0, w)])

    corr = (np.sum(i_mat*j_mat*glcm_list, axis=(-1, -2)) - i_mean * j_mean) / (i_std * j_std)
    # when std(X) = 0, we must have corr(X, Y) = 0.0
    corr = np.where(np.isnan(corr), 0.0, corr)

    return corr


def get_texture_statistics(method_list):
    texture_func_ref = {"Mean": get_GLCM_mean, "Variance": get_GLCM_variance, "Homogeneity": get_GLCM_homogeneity,
                        "Contrast": get_GLCM_contrast, "Dissimilarity": get_GLCM_dissimilarity, "Entropy": get_GLCM_entropy,
                        "2nd_moment": get_GLCM_2nd_moment, "Correlation": get_GLCM_correlation}

    func_dict = {}
    for m in method_list:
        if m in texture_func_ref.keys():
            func_dict[m] = texture_func_ref[m]
        else:
            raise NotImplementedError("Unknown Texture Function")

    return func_dict


if __name__ == "__main__":
    '''
    # ---an example for different Morphological Ops
    y, x = np.mgrid[:20:0.5, :20:0.5]
    bumps = np.sin(x) + np.sin(y)

    se_size = 7
    kernel = np.ones((se_size, se_size), dtype=np.uint8)

    # ---implementation based on Scikit-image
    bumps_opening = get_opening_img(bumps, se_size)
    delta_opening = bumps_opening - bumps
    bumps_opening_rec = get_opening_reconstruction_img(bumps, se_size)
    delta_opening_rec = bumps_opening_rec - bumps
    bumps_closing = get_closing_img(bumps, se_size)
    delta_closing = bumps_closing - bumps
    bumps_closing_rec = get_closing_reconstruction_img(bumps, se_size)
    delta_closing_rec = bumps_closing_rec - bumps

    # ---implementation based on OpenCV
    bumps_opening = get_opening_img_cv(bumps, se_size)
    delta_opening = bumps_opening - bumps
    bumps_opening_rec = get_opening_reconstruction_img_cv(bumps, se_size)
    delta_opening_rec = bumps_opening_rec - bumps
    bumps_closing = get_closing_img_cv(bumps, se_size)
    delta_closing = bumps_closing - bumps
    bumps_closing_rec = get_closing_reconstruction_img_cv(bumps, se_size)
    delta_closing_rec = bumps_closing_rec - bumps

    fig, ax = plt.subplots(4, 3, figsize=(12, 12))
    col_label = ["img", "MorphOps(img)", "MorphOps(img) - IMG"]
    row_label = ["Opening", "OpeningByReconstruction", "Closing", "ClosingByReconstruction"]

    for a, col in zip(ax[0], col_label):
        a.set_title(col, size='large')

    for a, row in zip(ax[:, 0], row_label):
        a.set_ylabel(row, size='large', rotation=90)

    ax00 = ax[0, 0].imshow(bumps, vmin=-2.0, vmax=2.0)
    fig.colorbar(ax00, ax=ax[0, 0])
    ax01 = ax[0, 1].imshow(bumps_opening, vmin=-2.0, vmax=2.0)
    fig.colorbar(ax01, ax=ax[0, 1])
    ax02 = ax[0, 2].imshow(delta_opening, vmin=-2.0, vmax=2.0)
    fig.colorbar(ax02, ax=ax[0, 2])

    ax10 = ax[1, 0].imshow(bumps, vmin=-2.0, vmax=2.0)
    fig.colorbar(ax10, ax=ax[1, 0])
    ax11 = ax[1, 1].imshow(bumps_opening_rec, vmin=-2.0, vmax=2.0)
    fig.colorbar(ax11, ax=ax[1, 1])
    ax12 = ax[1, 2].imshow(delta_opening_rec, vmin=-2.0, vmax=2.0)
    fig.colorbar(ax12, ax=ax[1, 2])

    ax20 = ax[2, 0].imshow(bumps, vmin=-2.0, vmax=2.0)
    fig.colorbar(ax20, ax=ax[2, 0])
    ax21 = ax[2, 1].imshow(bumps_closing, vmin=-2.0, vmax=2.0)
    fig.colorbar(ax21, ax=ax[2, 1])
    ax22 = ax[2, 2].imshow(delta_closing, vmin=-2.0, vmax=2.0)
    fig.colorbar(ax22, ax=ax[2, 2])

    ax30 = ax[3, 0].imshow(bumps, vmin=-2.0, vmax=2.0)
    fig.colorbar(ax30, ax=ax[3, 0])
    ax31 = ax[3, 1].imshow(bumps_closing_rec, vmin=-2.0, vmax=2.0)
    fig.colorbar(ax31, ax=ax[3, 1])
    ax32 = ax[3, 2].imshow(delta_closing_rec, vmin=-2.0, vmax=2.0)
    fig.colorbar(ax32, ax=ax[3, 2])

    # plt.show()
    plt.savefig("mp_ops_cv.png", dpi=400)
    '''


