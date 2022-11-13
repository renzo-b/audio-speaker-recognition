import os

import torch
import torchaudio


def get_dataset(dataset_name : str, url: str | None = None, subset : str | None = None):
    """
    Retrieves a dataset and downloads it locally if needed.
        
    Inputs
    ------
    dataset_name : str
        Name of the dataset to download. Allowed dataset names: "librispeech", "librilightlimited", "tedlium"
    
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
    
    elif dataset_name == "librilightlimited":
        subset = subset if subset else "10min"
        print(f"You're about to download librispeech with subset {subset}")
        dataset = torchaudio.datasets.LibriLightLimited(dataset_path, download=True, subset=subset)
        
    elif dataset_name == "tedlium":
        subset = subset if subset else "train"
        print(f"You're about to download Tedlium with subset {subset}")
        dataset = torchaudio.datasets.TEDLIUM(dataset_path, download=True, subset=subset)
    
    else:
        raise ValueError(f"The dataset you passed '{dataset_name}' is not in the list of datasets")
    
    print("\n Done downloading")
    return dataset 