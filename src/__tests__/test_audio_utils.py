import numpy as np
import pandas as pd

from src.audio_utils import CepstralNormalization, split_waveform


def test_split_waveform():

    waveform = np.array([1, 2, 3, 4, 5, 6 , 7, 8])
    window_size = 3

    expected_output = [np.array([1,2,3]), np.array([4,5,6])]
    actual_output = split_waveform(waveform, window_size)

    np.testing.assert_array_equal(expected_output, actual_output)
    

def test_cepstral_normalization():
    mfcc_channel_db = pd.DataFrame(
        columns=["mfcc_id", "channel_0", "channel_1"],
        data=[
            [1, np.array([10, 10]), np.array([100, 100])],
            [2, np.array([20, 20]), np.array([100, 100])],
            [3, np.array([30, 30]), np.array([100, 100])],
            ]
    ).set_index('mfcc_id')
    mfcc_ids = [1,2,3]

    cepstral = CepstralNormalization(number_spectral_coefficients=2)
    
    expected_mean = [20, 100]
    expected_std = [8.1649, 0]
    actual_mean, actual_std = cepstral.fit(mfcc_ids, mfcc_channel_db)

    np.testing.assert_almost_equal(expected_mean, actual_mean)
    np.testing.assert_almost_equal(expected_std, actual_std, decimal=2)

    cepstral.mfcc_mean = [10, 10]
    cepstral.mfcc_std = [10, 10]


    expected_mfcc_channel_db = pd.DataFrame(
        columns=["mfcc_id", "channel_0", "channel_1"],
        data=[
            [1, np.array([0.,0.]), np.array([9.,9.])],
            [2, np.array([1.,1.]), np.array([9.,9.])],
            [3, np.array([2.,2.]), np.array([9.,9.])],
            ]
    ).set_index('mfcc_id')
    actual_mfcc_channel_db = cepstral.transform(mfcc_ids, mfcc_channel_db)

    pd.testing.assert_frame_equal(expected_mfcc_channel_db, actual_mfcc_channel_db)
