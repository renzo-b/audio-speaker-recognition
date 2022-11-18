import os
from random import randint

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from IPython.display import Audio, display


#Get a random video clip from dataset
def get_random_sample(dataset):
    """
    Takes a random sample in the dataset and return its waveform and sample rate
    """
    dataset_length = dataset.__len__()
    # get a random sample in the dataset
    sample_id = randint(0, dataset_length)
    (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id) = dataset.__getitem__(sample_id)
    
    return waveform, sample_rate
    
def plot_mfccs(waveform, sample_rate):
    wv_processed = waveform.numpy().flatten()
    mfccs = librosa.feature.mfcc(y=wv_processed, n_mfcc=13, sr=sample_rate)
    plt.figure(figsize=(15, 8))
    librosa.display.specshow(mfccs, x_axis="time", sr=sample_rate)
    #plt.colorbar(format="%+2.f")
    plt.show()
    
    
    
def play_audio(waveform : torch.Tensor, sample_rate : int) -> None:
    """
    Plays audio given a waveform and sample rate
    
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")
        
def plot_specgram(waveform : torch.Tensor, sample_rate : int , title : str ="Spectrogram", xlim=None) -> None:
    """
    Plots a spectogram of audio given a waveform and sample rate
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)
    
def plot_waveform(waveform : torch.Tensor, sample_rate : int, title : str ="Waveform", xlim : float | None =None, ylim : float | None=None) -> None:
    """
    Plots the waveform of an audio sample
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
    if num_channels > 1:
        axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
        axes[c].set_xlim(xlim)
    if ylim:
        axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)
    
def visualize_random_sample(dataset, play_waveform : bool = True, plot_wave : bool = False, plot_spectogram : bool = False) -> None:
    """
    Takes a random sample in the dataset and displays relevant metrics
    """
    dataset_length = dataset.__len__()
    # get a random sample in the dataset
    sample_id = randint(0, dataset_length)
    (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id) = dataset.__getitem__(sample_id)
    print('waveform: ', waveform)
    print('sample_rate: ', sample_rate)
    print('transcript: ', transcript)
    print('speaker_id: ', speaker_id)
    
    if play_waveform:
        play_audio(waveform, sample_rate)
        
    if plot_wave:
        plot_waveform(waveform, sample_rate)
    
    if plot_spectogram:
        plot_specgram(waveform, sample_rate, title=f"Sample {sample_id}, Speaker_id {speaker_id}")
        
def calculate_statistics(dataset) -> pd.DataFrame:
    """
    Calculates and plots statistics about the dataset
    
    Inputs
    ------
    dataset :
        The dataset to calculate statistics for
        
    verbose : bool
        If true, it will print the statistics
        
    Returns
    -------
    df_info : pd.DataFrame
        A dataframe with statistics about the dataset. Can be used to make your own statistics
    """
    print(f"Dataset has {dataset.__len__()} samples")
    print(f"Random samples has shape: {dataset.__getitem__(0)[0].shape}")
    
    waveform_lengths = []
    sample_rates = []
    transcript_lengths = []
    speaker_ids = []
    audio_durations = []    
    
    for i in range(dataset.__len__()):
        (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id) = dataset.__getitem__(i)
        waveform_lengths.append(waveform[0].shape[0])
        sample_rates.append(sample_rate)
        transcript_lengths.append(len(transcript))
        speaker_ids.append(speaker_id)
        
        num_channels, num_frames = waveform.shape
        duration = num_frames / sample_rate
        audio_durations.append(duration)
        
    dataset_stats = pd.DataFrame.from_dict(
    {
        "waveform_lengths" : waveform_lengths,
        "sample_rates" : sample_rates,
        "transcript_lengths" : transcript_lengths,
        "audio_durations" : audio_durations,
    }
    )
    
    print(f"Unique speaker ids: {len(np.unique(speaker_ids))}")
    display(dataset_stats.describe().loc[["min","std","min","max"]].round())
    print("\n")
    
    # plotting:
    fig, ax = plt.subplots(1, dataset_stats.shape[1], figsize=(12, 6))
    for i, col in enumerate(dataset_stats.columns):
        ax[i].boxplot(dataset_stats[col])
        ax[i].set_title(col)
    plt.tight_layout()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(dataset_stats["waveform_lengths"], dataset_stats["transcript_lengths"])
    plt.xlabel("waveform_lengths")
    plt.ylabel("transcript_lengths")
    
    return dataset_stats