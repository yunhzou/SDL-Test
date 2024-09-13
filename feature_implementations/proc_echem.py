from datetime import datetime
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def filter_outlier(raw_array, outlier_filt: float = 2):
    # filter outliers of 2nd column of a 2D array
    mean = np.mean(raw_array[:,1])
    std = np.std(raw_array[:, 1])
    for i in range(0, raw_array.shape[0]):
        if (raw_array[i, 1] < mean - std * outlier_filt) or (raw_array[i, 1] > mean + std * outlier_filt):
            raw_array[i, :] = [np.nan, np.nan]
    return raw_array

def avg_vi(cv, block_size: int = 16, outlier_filt: float = 2):
    # calculate the block average to denoize the CV plot
    new_cv_len = cv.shape[0] // block_size
    new_cv = np.zeros((new_cv_len, 2))
    for i in range(0, new_cv_len):
        filtered_block = filter_outlier(cv[i * block_size : (i+1) * block_size, :], outlier_filt)
        avg_voltage = np.nanmean(filtered_block[:, 0])
        avg_current = np.nanmean(filtered_block[:, 1])
        new_cv[i] = [avg_voltage, avg_current]
    return new_cv

def gaussian(x, A, mu, sigma, bkg):
    return bkg + A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fit_gauss(data):
    max_current_pos = np.argmax(data[:, 2])

    init_guess = [data[max_current_pos, 2] - data[0, 2], data[max_current_pos, 1], np.std(data[:, 1]), data[0, 2]]
    try:
        gau_opt, gau_cov = curve_fit(gaussian, data[:, 1], data[:, 2], p0=init_guess)
        return gau_opt
    except:
        return init_guess

def proc_dpv(data, decay_ms:int = 500, pulse_ms:int = 50, pulse_from_end: int = 4, decay_from_end: int =50):
    drop_pts = decay_ms + pulse_ms
    num_periods = data.shape[0] // drop_pts

    dpv = np.empty((num_periods, 6))
    for i in range (0, num_periods):
        decay_end_point = drop_pts * i + decay_ms
        drop_end_point = drop_pts * (i + 1)
        dpv_time = data[decay_end_point-1, 0]
        dpv_cycle = data[decay_end_point-1, 3]
        dpv_exp = data[decay_end_point-1, 4]
        dpv_v_apply = data[decay_end_point-1, 5]
        dpv_current = np.mean(data[decay_end_point-decay_from_end:decay_end_point-1, 2]) - \
        np.mean(data[drop_end_point-pulse_from_end:drop_end_point-1, 2])
        dpv_voltage = np.mean(data[decay_end_point-decay_from_end:decay_end_point-1, 1])
        dpv[i] = [dpv_time, dpv_voltage, dpv_current, dpv_cycle, dpv_exp, dpv_v_apply]
    dpv = dpv[(dpv[:,1] >= -1.25) & (dpv[:, 1] <= 1.25)]
    return dpv

def dpv_phasing(data):
    """
    split ONE cycle of dpv to positive and negative phase
    :param data:
    :return:
    """
    phase_up = data[data[:, 2] >= 0]
    phase_down = data[data[:, 2] < 0]
    return phase_up, phase_down
