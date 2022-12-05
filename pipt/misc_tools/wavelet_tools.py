import pywt
import numpy as np
import sys
from copy import deepcopy
import warnings


# Define the class for sparse representation of seismic data using wavelet compression.
class SparseRepresentation:

    # Initialize
    def __init__(self, options):
        # options: dim, actnum, level, wname, colored_noise, threshold_rule, th_mult,
        #          use_hard_th, keep_ca, inactive_value
        # dim must be given as (nz,ny,nx)
        self.options = options
        self.num_grid = np.prod(self.options['dim'])

        self.threshold = np.array([])
        self.est_noise = np.array([])
        self.num_total_coeff = None
        self.cd_leading_index = None
        self.cd_leading_coeff = None
        self.ca_leading_index = None
        self.ca_leading_coeff = None

    # Function to doing image compression. If the function is called without threshold, then the leading indices must
    # be defined in the class. Typically this is done by running the compression on true data with a given threshold.
    def compress(self, data, th_mult=None):
        if self.options['inactive_value'] is None:
            self.options['inactive_value'] = np.mean(data)
        signal = np.zeros(self.num_grid)
        signal[~self.options['actnum']] = self.options['inactive_value']
        signal[self.options['actnum']] = data
        signal = signal.reshape(self.options['dim'])
        signal = signal.transpose((2, 1, 0))  # get the signal back into its original shape (nx,ny,nz)
        # pywt throws a warning in case of single-dimentional entries in the shape of the signal.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wdec = pywt.wavedecn(signal, self.options['wname'], 'symmetric', int(self.options['level']))
        wdec_rec = deepcopy(wdec)
        
        # Perform thresholding if the value is given as input.
        do_thresholding = False
        current_threshold = None
        est_noise_level = None
        if th_mult is not None and th_mult >= 0:
            do_thresholding = True

        # In the white noise case estimated std is based on the high (hhh) subband only
        if do_thresholding and not self.options['colored_noise']:
            subband_hhh = wdec[-1]['ddd'].flatten()
            est_noise_level = np.median(np.abs(subband_hhh - np.median(subband_hhh))) / 0.6745  # estimated noise std
            est_noise_level = np.maximum(est_noise_level, 1e-9)
            # Threshold based on universal rule
            current_threshold = th_mult * np.sqrt(2 * np.log(np.size(data))) * est_noise_level
            self.threshold = np.append(self.threshold, current_threshold)

        # Loop over all levels and subbands (including the lll subband)
        # Note: the keys below are organized the same way as in Matlab.
        keys = ['daa', 'ada', 'dda', 'aad', 'dad', 'add', 'ddd']  
        ca_in_vec = np.array([])
        cd_in_vec = np.array([])
        for level in range(0, int(self.options['level'])+1):
            num_subband = 1 if level == 0 else len(keys)
            for subband in range(0, num_subband):
                coeffs = wdec[level] if level == 0 else wdec[level][keys[subband]]
                coeffs = coeffs.flatten()

                # In the colored noise case estimated std is based on all subbands
                if do_thresholding and self.options['colored_noise']:
                    est_noise_level = np.median(np.abs(coeffs - np.median(coeffs))) / 0.6745
                    est_noise_level = np.maximum(est_noise_level, 1e-9)
                    if self.options['threshold_rule'] == 'universal':  # threshold based on universal rule
                        current_threshold = np.sqrt(2 * np.log(np.size(coeffs))) * est_noise_level
                    elif self.options['threshold_rule'] == 'bayesian':  # threshold based on bayesian rule
                        std_data = np.linalg.norm(coeffs, 2) / len(coeffs)
                        current_threshold = est_noise_level**2 / np.sqrt(np.abs(std_data**2 - est_noise_level**2))
                    else:
                        print('Thresholding rule not implemented')
                        sys.exit()
                    current_threshold = th_mult * current_threshold
                    self.threshold = np.append(self.threshold, current_threshold)

                # Perform thresholding
                if do_thresholding:
                    zero_index = []
                    if level > 0 or (level == 0 and not self.options['keep_ca']):
                        if self.options['use_hard_th']:  # use hard thresholding
                            zero_index = np.abs(coeffs) < current_threshold
                            coeffs[zero_index] = 0
                        else:  # use soft thresholding
                            coeffs = np.sign(coeffs) * np.maximum(np.abs(coeffs) - current_threshold, 0)
                            zero_index = coeffs == 0
                    num_el = len(coeffs) - np.count_nonzero(zero_index)
                    self.est_noise = np.append(self.est_noise, est_noise_level*np.ones(num_el))

                if level == 0:
                    ca_in_vec = coeffs
                else:
                    cd_in_vec = np.append(cd_in_vec, coeffs)

        if do_thresholding:
            self.num_total_coeff = ca_in_vec.size + cd_in_vec.size
            self.cd_leading_index = np.nonzero(cd_in_vec)[0]
            self.cd_leading_coeff = cd_in_vec[self.cd_leading_index]
            if self.options['keep_ca']:
                self.ca_leading_index = np.arange(wdec[0].size)
            else:
                self.ca_leading_index = np.nonzero(ca_in_vec)[0]
            self.ca_leading_coeff = ca_in_vec[self.ca_leading_index]
            compressed_data = np.append(self.ca_leading_coeff, self.cd_leading_coeff)
        else:
            if self.ca_leading_index is None or self.cd_leading_index is None:
                print('Leading indices not defined')
                sys.exit()
            compressed_data = np.append(ca_in_vec[self.ca_leading_index], cd_in_vec[self.cd_leading_index])
            ca_non_leading_index = np.arange(len(ca_in_vec))
            ca_non_leading_index = np.delete(ca_non_leading_index, self.ca_leading_index)
            ca_in_vec[ca_non_leading_index] = 0
            cd_non_leading_index = np.arange(len(cd_in_vec))
            cd_non_leading_index = np.delete(cd_non_leading_index, self.cd_leading_index)
            cd_in_vec[cd_non_leading_index] = 0

        # construct wdec_rec
        ind = 0
        for level in range(0, int(self.options['level']) + 1):
            if level == 0:
                shape = wdec_rec[level].shape
                wdec_rec[level] = ca_in_vec.reshape(shape)
            else:
                num_subband = len(keys)
                for subband in range(0, num_subband):
                    shape = wdec_rec[level][keys[subband]].shape
                    coeff = cd_in_vec[ind:ind+np.prod(shape)]
                    wdec_rec[level][keys[subband]] = coeff.reshape(shape)
                    ind += np.prod(shape)

        return compressed_data, wdec_rec

    # Reconstruct the current compressed dataset. 
    def reconstruct(self, wdec_rec):

        if wdec_rec is None:
            print('No signal to reconstruct')
            sys.exit()

        data_rec = pywt.waverecn(wdec_rec, self.options['wname'], 'symmetric')
        data_rec = data_rec.transpose((2, 1, 0))  # flip the axes
        dim = self.options['dim']
        data_rec = data_rec[0:dim[0], 0:dim[1], 0:dim[2]]  # severe issure here
        data_rec = data_rec.flatten()
        data_rec = data_rec[self.options['actnum']]

        return data_rec
