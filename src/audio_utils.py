from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm


def split_waveform(waveform : npt.NDArray[np.float64], window_size : int) -> list[npt.NDArray[np.float64]]:
    """
    Takes an original waveform and reduce to chunks of smaller intervals using a sliding window
    
    waveform : nd.array
    
    window_size : int
        window size in seconds
    
    Returns
    -------
    splitted_waveform : list[nd.array]
        list of waveforms
    
    """
    splitted_waveform = []
    
    for i in range(0, len(waveform), window_size):
        split = waveform[i:i+window_size]
        if len(split) == window_size:
            splitted_waveform.append(split)

    return splitted_waveform

class CepstralNormalization:
    def __init__(self, number_spectral_coefficients : int):
        """
        Class for Cepstral Normalization. Cepstral normalization performs normalization
        by channels/coefficients
        """
        self.number_spectral_coefficients = number_spectral_coefficients
        self.mfcc_mean : list[float] = []
        self.mfcc_std : list[float] = []
    
    def fit(self, mfcc_ids : list[int], mfcc_channel_db : pd.DataFrame) -> Union[list[float], list[float]]:
        """
        Calculates the mean and standard deviation of each channel of the MFCCs
        
        Inputs
        ------
        mfcc_ids : list[int]
            list of the mfcc ids to use for fitting

        mfcc_channel_db : pd.DataFrame
            A dataframe of the following shape

                    channel_1 | channel 2 | channel 3 | ...
            mfcc_id

        Returns
        -------
        mfcc_mean : list[float]
            A list with the mean value of each channel
        
        mfcc_std : list[float]
            A list with the standard deviation value of each channel

        """
        mfcc_mean = []
        mfcc_std = []
        
        print('Normalizing...')
        for channel in tqdm(range(self.number_spectral_coefficients)):
            channel_data = mfcc_channel_db.loc[mfcc_ids][f"channel_{channel}"].values
            channel_data_list = []
            for x in channel_data:
                channel_data_list.extend(x.flatten().tolist())
            
            mfcc_mean.append(np.mean(channel_data_list))
            mfcc_std.append(np.std(channel_data_list))

        self.mfcc_mean, self.mfcc_std = mfcc_mean, mfcc_std
        
        return self.mfcc_mean, self.mfcc_std
    
    def transform(self, mfcc_ids : list[int], mfcc_channel_db : pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes MFCCs

        Inputs
        ------
        mfcc_ids : list[int]
            list of the mfcc ids to normalize

        mfcc_channel_db : pd.DataFrame
            A dataframe of the following shape

                    channel_1 | channel 2 | channel 3 | ...
            mfcc_id

        Returns
        -------
        mfcc_channel_db : pd.DataFrame
            normalized dataframe

        """                
        normalized_mfcc_channel_db = pd.DataFrame(data=mfcc_ids, columns=['mfcc_id']).set_index('mfcc_id')

        for channel in range(self.number_spectral_coefficients):
            channel_data = mfcc_channel_db.loc[mfcc_ids][f"channel_{channel}"].values
            channel_data = (channel_data - self.mfcc_mean[channel]) / self.mfcc_std[channel]
            normalized_mfcc_channel_db[f"channel_{channel}"] = channel_data
            
        return normalized_mfcc_channel_db

    def inverse_transform(self):
        #TODO
        raise NotImplementedError