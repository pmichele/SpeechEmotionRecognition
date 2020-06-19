import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
from glob import glob
import json
import os


class MultiFeaturesDataset(Dataset):
    """This dataset is responsible for feature extraction from audio samples.
        It computes and concatenates multiple features for each sample.
        The samples are loaded on the fly so that data loading is parallelized
        and optimized.
    """
    def __init__(self, data_path, label_map_path, mode):
        self.names = []
        self.labels = []
        with open(label_map_path) as label_map:
            self.label_map = json.load(label_map)
        if mode == "train" or mode == "test":
            data_by_class = glob(os.path.join(data_path, "*"))
            for group in data_by_class:
                _, label = os.path.split(group)
                cls = self.label_map[label]
                samples = glob(os.path.join(group, "*.wav"))
                self.names += samples
                self.labels += [cls] * len(samples)
        elif mode == "predict":
            from src.data_preparation.DataPreparation import files_to_labels
            # map from class index to string
            self.class_index_to_label_map = [None] * len(self.label_map)
            for label, cls in self.label_map.items():
                self.class_index_to_label_map[cls] = label
            self.names = glob(os.path.join(data_path, "*.wav"))
            self.labels = [self.label_map[label] for label in files_to_labels(self.names)]
        else:
            raise ValueError("mode {} is invalid".format(mode))

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return Iterator(self)

    def __getitem__(self, i):
        """Return audio features and a target label. The feature vector is of size 193
            and it consists of 5 different features averaged along time:
                1. MFCC
                2. Mel-spectrogram
                3. Chromagram
                4. Spectral Contrast
                5. Tonnetz representation
        """
        sample, sr = librosa.load(self.names[i], sr=None)
        mfcc = librosa.feature.mfcc(sample, sr, n_mfcc=60).mean(axis=1)
        melspectrogram = librosa.feature.melspectrogram(sample, sr, n_mels=60).mean(axis=1)
        chroma = librosa.feature.chroma_stft(sample, sr, n_chroma=60)
        spectral_contrast = librosa.feature.spectral_contrast(sample, sr, n_bands=6).mean(axis=1)
        tonnetz = librosa.feature.tonnetz(sample, sr, chroma=chroma).mean(axis=1)
        chroma = chroma.mean(axis=1)
        features = np.concatenate([mfcc, chroma, melspectrogram, spectral_contrast, tonnetz], axis=0)
        features = np.expand_dims(features, axis=0).astype(np.float32)
        return torch.from_numpy(features), self.labels[i]


class Iterator:
    """Iterator for the MultiFeaturesDataset"""
    def __init__(self, dataset):
        self._i = 0
        self._dataset = dataset

    def __next__(self):
        if self._i >= len(self._dataset):
            raise StopIteration
        self._i += 1
        return self._dataset[self._i - 1]


