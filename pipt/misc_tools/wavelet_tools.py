"""
Sparse representation of seismic data using wavelet compression.

Copyright (c) 2019-2022 NORCE, All Rights Reserved. 4DSEIS
"""
import pywt
import numpy as np
import sys
from copy import deepcopy
import warnings


class SparseRepresentation:

    # Initialize
    def __init__(self, options):
        # options: dim, actnum, level, wname, colored_noise, threshold_rule, th_mult,
        #          use_hard_th, keep_ca, inactive_value
        self.options = options
        self.num_grid = np.prod(self.options['dim'])

        self.est_noise = np.array([])
        self.est_noise_level = []  # this is a scalar for each subband
        self.mask = []
        self.threshold = []
        self.num_total_coeff = None
        self.cd_leading_index = None
        self.cd_leading_coeff = None
        self.ca_leading_index = None
        self.ca_leading_coeff = None

    # Function for image compression. If the function is called without threshold, then the leading indices must
    # be defined in the class. Typically, this is done by running the compression on true data with a given threshold.
    def compress(self, data, th_mult=None):
        if ('inactive_value' not in self.options) or (self.options['inactive_value'] is None):
            self.options['inactive_value'] = np.mean(data)
        signal = np.zeros(self.num_grid)
        signal[~self.options['mask']] = self.options['inactive_value']
        signal[self.options['mask']] = data
        if 'order' not in self.options:
            self.options['order'] = 'C'
        if 'min_noise' not in self.options:
            self.options['min_noise'] = 1.0e-9
        signal = signal.reshape(self.options['dim'], order=self.options['order'])

        # Wavelet decomposition
        wdec = pywt.wavedecn(signal, self.options['wname'], 'symmetric', int(self.options['level']))

        wdec_rec = deepcopy(wdec)

        # Perform thresholding if the threshold is given as input.
        do_thresholding = False
        est_noise_level = None
        true_data = False
        if th_mult is not None and th_mult >= 0:
            do_thresholding = True
            if not self.est_noise.size:  # assume that the true data is input
                true_data = True
            else:
                self.est_noise = np.array([])  # this will be rebuilt

        # Initialize
        # Note: the keys below are organized the same way as in Matlab.
        if signal.ndim == 3:
            keys = ['daa', 'ada', 'dda', 'aad', 'dad', 'add', 'ddd']
            details = 'ddd'
        elif signal.ndim == 2:
            keys = ['da', 'ad', 'dd']
            details = 'dd'
        if true_data:
            for level in range(0, int(self.options['level'])+1):
                num_subband = 1 if level == 0 else len(keys)
                for subband in range(0, num_subband):
                    if level == 0:
                        self.mask.append(False)
                        self.est_noise_level.append(0)
                        self.threshold.append(0)
                    else:
                        if subband == 0:
                            self.mask.append({})
                            self.est_noise_level.append({})
                            self.threshold.append({})
                        self.mask[level][keys[subband]] = False
                        self.est_noise_level[level][keys[subband]] = 0
                        self.threshold[level][keys[subband]] = 0

        # In the white noise case estimated std is based on the high (hhh) subband only
        if true_data and not self.options['colored_noise']:
            subband_hhh = wdec[-1][details].flatten()
            est_noise_level = np.median(
                np.abs(subband_hhh - np.median(subband_hhh))) / 0.6745  # estimated noise std
            est_noise_level = np.maximum(est_noise_level, self.options['min_noise'])
            # Threshold based on universal rule
            current_threshold = th_mult * \
                np.sqrt(2 * np.log(np.size(data))) * est_noise_level
            self.threshold[0] = current_threshold

        # Loop over all levels and subbands (including the lll subband)
        ca_in_vec = np.array([])
        cd_in_vec = np.array([])
        for level in range(0, int(self.options['level'])+1):
            num_subband = 1 if level == 0 else len(keys)
            for subband in range(0, num_subband):
                coeffs = wdec[level] if level == 0 else wdec[level][keys[subband]]
                coeffs = coeffs.flatten()

                # In the colored noise case estimated std is based on all subbands
                if true_data and self.options['colored_noise']:
                    est_noise_level = np.median(
                        np.abs(coeffs - np.median(coeffs))) / 0.6745
                    est_noise_level = np.maximum(
                        est_noise_level, self.options['min_noise'])
                    # threshold based on universal rule
                    if self.options['threshold_rule'] == 'universal':
                        current_threshold = np.sqrt(
                            2 * np.log(np.size(coeffs))) * est_noise_level
                    # threshold based on bayesian rule
                    elif self.options['threshold_rule'] == 'bayesian':
                        std_data = np.linalg.norm(coeffs, 2) / len(coeffs)
                        current_threshold = est_noise_level**2 / \
                            np.sqrt(np.abs(std_data**2 - est_noise_level**2))
                    else:
                        print('Thresholding rule not implemented')
                        sys.exit(1)
                    current_threshold = th_mult * current_threshold
                    if level == 0:
                        self.threshold[level] = current_threshold
                    else:
                        self.threshold[level][keys[subband]] = current_threshold
                    # self.threshold = np.append(self.threshold, current_threshold)

                # Perform thresholding
                if do_thresholding:
                    if level == 0 or not self.options['colored_noise']:
                        current_threshold = self.threshold[0]
                    else:
                        current_threshold = self.threshold[level][keys[subband]]
                    if level > 0 or (level == 0 and not self.options['keep_ca']):
                        if self.options['use_hard_th']:  # use hard thresholding
                            zero_index = np.abs(coeffs) < current_threshold
                            coeffs[zero_index] = 0
                        else:  # use soft thresholding
                            coeffs = np.sign(
                                coeffs) * np.maximum(np.abs(coeffs) - current_threshold, 0)
                            zero_index = coeffs == 0
                    else:
                        zero_index = np.zeros(coeffs.size).astype(bool)

                    # Construct the mask for each subband and estimate the noise level
                    if level == 0:
                        self.mask[level] = np.invert(zero_index) + self.mask[level]
                        if true_data:
                            self.est_noise_level[level] = est_noise_level
                    else:
                        self.mask[level][keys[subband]] = np.invert(
                            zero_index) + self.mask[level][keys[subband]]
                        if true_data:
                            self.est_noise_level[level][keys[subband]] = est_noise_level

                    # Build the noise for the compressed signal
                    if level == 0:
                        num_el = np.sum(self.mask[level])
                        current_noise_level = self.est_noise_level[level]
                        self.est_noise = np.append(
                            self.est_noise, current_noise_level * np.ones(num_el))
                    else:
                        num_el = np.sum(self.mask[level][keys[subband]])
                        current_noise_level = self.est_noise_level[level][keys[subband]]
                        self.est_noise = np.append(
                            self.est_noise, current_noise_level * np.ones(num_el))

                if level == 0:
                    ca_in_vec = coeffs
                else:
                    cd_in_vec = np.append(cd_in_vec, coeffs)

        # Compute the leading indices and compressed signal
        if do_thresholding:
            self.num_total_coeff = ca_in_vec.size + cd_in_vec.size
            if self.cd_leading_index is not None:
                self.cd_leading_index = np.union1d(
                    self.cd_leading_index, np.nonzero(cd_in_vec)[0])
            else:
                self.cd_leading_index = np.nonzero(cd_in_vec)[0]
            self.cd_leading_coeff = cd_in_vec[self.cd_leading_index]
            if self.options['keep_ca']:
                if self.ca_leading_index is None:
                    self.ca_leading_index = np.arange(wdec[0].size)
            else:
                if self.ca_leading_index is None:
                    self.ca_leading_index = np.nonzero(ca_in_vec)[0]
                else:
                    self.ca_leading_index = np.union1d(
                        self.ca_leading_index, np.nonzero(ca_in_vec)[0])
            self.ca_leading_coeff = ca_in_vec[self.ca_leading_index]
            compressed_data = np.append(self.ca_leading_coeff, self.cd_leading_coeff)
        else:
            if self.ca_leading_index is None or self.cd_leading_index is None:
                print('Leading indices not defined')
                sys.exit(1)
            compressed_data = np.append(
                ca_in_vec[self.ca_leading_index], cd_in_vec[self.cd_leading_index])

        # Construct wdec_rec
        for level in range(0, int(self.options['level']) + 1):
            if level == 0:
                shape = wdec_rec[level].shape
                coeff = wdec_rec[level].flatten()
                coeff[~self.mask[level]] = 0
                wdec_rec[level] = coeff.reshape(shape)
            else:
                num_subband = len(keys)
                for subband in range(0, num_subband):
                    shape = wdec_rec[level][keys[subband]].shape
                    coeff = wdec_rec[level][keys[subband]].flatten()
                    coeff[~self.mask[level][keys[subband]]] = 0
                    wdec_rec[level][keys[subband]] = coeff.reshape(shape)

        return compressed_data, wdec_rec

    # Reconstruct the current compressed dataset.
    def reconstruct(self, wdec_rec):

        if wdec_rec is None:
            print('No signal to reconstruct')
            sys.exit(1)

        # reconstruct from wavelet coefficients
        data_rec = pywt.waverecn(wdec_rec, self.options['wname'], 'symmetric')
        data_rec = data_rec[tuple(slice(0, s) for s in self.options['dim'])] 

        data_rec = data_rec.flatten(order=self.options['order'])
        data_rec = data_rec[self.options['mask']]

        return data_rec
