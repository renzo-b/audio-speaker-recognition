import os

import pandas as pd
import torch
import torchaudio


def get_dataset(dataset_name : str, url: str | None = None, subset : str | None = None):
    """
    Retrieves a dataset and downloads it locally if needed.
        
    Inputs
    ------
    dataset_name : str
        Name of the dataset to download. Allowed dataset names: "librispeech", "voxceleb1identification"
    
    url : (str, optional)
        Optional argument for some datasets 
    
    subset : (str, optional)
        Optional argument for some datasets 
    
    Returns
    -------
    dataset :
        A torchaudio.dataset
    
    """
    _SAMPLE_DIR = "_assets"
    dataset_name = dataset_name.lower()
    dataset_path = os.path.join(_SAMPLE_DIR, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
        
    if dataset_name == "librispeech":
        url = url if url else "dev-clean"
        print(f"You're about to download librispeech with url {url}")
        dataset = torchaudio.datasets.LIBRISPEECH(dataset_path, download=True, url=url)
    
    elif dataset_name == "voxceleb1identification":
        subset = subset if subset else "train"
        print(f"You're about to download VoxCeleb with subset {subset}")
        dataset = torchaudio.datasets.VoxCeleb1Identification(dataset_path, download=True, subset=subset)

    else:
        raise ValueError(f"The dataset you passed '{dataset_name}' is not in the list of datasets")
    
    print("\n Done downloading")
    return dataset 


def read_pickled_dataset(dataset_name : str):
    """
    Inputs
    ------
    dataset_name : str
        Name of the dataset to download. 
    
    Returns
    -------
    speaker_mfcc_db : pd.DataFrame 
        df with shape (index_id, speaker_id, mfcc_id)
    
    mfcc_channel_db : pd.DataFrame
        df with shape (mfcc_id, channel_id)

    """
    if dataset_name == "librispeech-train-clean-100":
        speaker_mfcc_db = pd.read_pickle('speaker_mfcc_db_64000_16000_13_100.pkl')
        mfcc_channel_db = pd.read_pickle('mfcc_channel_db_64000_16000_13_100.pkl')
    
    elif dataset_name == "librispeech-train-other-500":
        speaker_mfcc_db = pd.read_pickle(f'speaker_mfcc_db_64000_16000_13_500.pkl')
        mfcc_channel_db = pd.read_pickle(f'mfcc_channel_db_64000_16000_13_500.pkl')

    elif dataset_name == "voxceleb1identification-4s":
        speaker_mfcc_db = pd.read_pickle(f'speaker_mfcc_db_64000_16000_13_voxceleb_4s.pkl')
        mfcc_channel_db = pd.read_pickle(f'mfcc_channel_db_64000_16000_13_voxceleb_4s.pkl')

    elif dataset_name == "voxceleb_with_sinisa":
        speaker_mfcc_db = pd.read_pickle(f'speaker_mfcc_voxceleb_with_sinisa.pkl')
        mfcc_channel_db = pd.read_pickle(f'mfcc_channel_db_voxceleb_with_sinisa.pkl')

    elif dataset_name == "librispeech-mixed":
        speaker_mfcc_db_500 = pd.read_pickle(f'speaker_mfcc_db_64000_16000_13_500.pkl')
        mfcc_channel_db_500 = pd.read_pickle(f'mfcc_channel_db_64000_16000_13_500.pkl')

        speaker_mfcc_db_100 = pd.read_pickle('speaker_mfcc_db_64000_16000_13_100.pkl')
        mfcc_channel_db_100= pd.read_pickle('mfcc_channel_db_64000_16000_13_100.pkl')

        speaker_mfcc_db_100["mfcc_id"] = speaker_mfcc_db_100["mfcc_id"] + 400000
        mfcc_channel_db_100.index = mfcc_channel_db_100.index + 400000

        mfcc_channel_db = pd.concat([mfcc_channel_db_500, mfcc_channel_db_100])
        speaker_mfcc_db = pd.concat([speaker_mfcc_db_500, speaker_mfcc_db_100])
   

    else:
        raise ValueError(f"The dataset you passed '{dataset_name}' is not in the list of datasets")

    print(f"\n Loaded {dataset_name}")

    return speaker_mfcc_db, mfcc_channel_db


