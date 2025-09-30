# Convenience functions
import numpy as np
import random
from python_speech_features import mfcc
from scipy import signal
from scipy.signal.windows import hann


def py_speech_preprocessing(
    raw_signal,
    sample_rate,
    tf_desired_samples=64000,
    tf_window_size_samples=480,
    tf_sample_rate=16000,
    tf_window_size_ms=30.0,
    tf_window_stride_ms=20.0,
    tf_dct_coefficient_count=10,
):
    # Resample
    num_target_samples = round(tf_sample_rate / sample_rate * len(raw_signal))
    resampled_data = signal.resample(raw_signal, num_target_samples)
    # Rescale
    rescaled_data = resampled_data / np.max(resampled_data)
    # randomize padding
    pad_length = tf_desired_samples - rescaled_data.shape[-1]
    pad_right = random.randint(0, pad_length)
    pad_left = pad_length - pad_right
    # Pad
    padded_data = np.pad(
        rescaled_data,
        [[pad_left, pad_right]],
        mode="constant",
    )
    # Calculate MFCC features
    nfft = int(2 ** np.ceil(np.log2(tf_window_size_samples)))
    mfcc_feat_py = mfcc(
        padded_data,
        tf_sample_rate,
        winlen=tf_window_size_ms / 1000.0,
        winstep=tf_window_stride_ms / 1000.0,
        numcep=tf_dct_coefficient_count,
        nfilt=40,
        nfft=nfft,
        lowfreq=20.0,
        highfreq=4000.0,
        winfunc=hann,
        appendEnergy=False,
        preemph=0.0,
        ceplifter=0.0,
    )
    # Cut and transpose MFCC features
    mfcc_feat_py = mfcc_feat_py[:-1, :].T
    return mfcc_feat_py


def quantize_input(mfcc_feat_py):
    # Scaling
    quant_mfcc_feat = mfcc_feat_py / 0.8298503756523132
    # Clamping & rounding
    quant_mfcc_feat = np.where(quant_mfcc_feat > 127.0, 127.0, quant_mfcc_feat)
    quant_mfcc_feat = np.where(quant_mfcc_feat < -127.0, -127.0, quant_mfcc_feat)
    quant_mfcc_feat = np.round(quant_mfcc_feat)
    # flatten the array to one dimension
    quant_mfcc_feat = quant_mfcc_feat.astype(np.int8).reshape((1, quant_mfcc_feat.size))
    return quant_mfcc_feat
