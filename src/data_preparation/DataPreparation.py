"""This module produces a pre-processed and augmented dataset"""
from src import load_config
import librosa, librosa.display
import json
import soundfile
from sklearn.model_selection import train_test_split
import random
import numpy as np
from glob import glob
import os
import shutil

"""Load file-to-label map and parameters. Structure of the config:
        max_offset: the time in seconds of maximum shift during data augmentation
        input_data: the raw data to pre-process
        output_data: the output folder of the data preparation
        sr: the (expected) sampling rate
        file_map: the location of the sample filename to emotion label map.
        n_augm: the number of samples to generate from an original sample.
"""
config = load_config("/src/data_preparation/config.json")
max_offset = int(config.max_offset * config.sr)      # Convert max_offset from seconds to hertz

with open(config.file_map) as file_map:
    _filename_to_label_map = json.load(file_map)


def files_to_labels(files):
    """Convert a list of files to emotion labels"""
    labels = []
    for file in files:
        _, filename = os.path.split(file)
        code = filename[5]
        if code not in _filename_to_label_map:
            raise ValueError("Unexpected label in filename {f}".format(f=filename))
        labels.append(_filename_to_label_map[code])
    return labels


def augment_sample(sample, n_augm):
    """Produce a number of augmented samples (shifted and stretched versions
        of the original).
        Args:
            sample: the sample to augment.
            n_augmentations: how many random samples to generate
        Returns:
            a list of size [n_augmentations] containing augmented samples
    """
    augmented_samples = []
    for i in range(n_augm):
        eps = random.random()
        rate = 1
        if eps < 1 / 3:
            rate = 1.23
        elif eps < 2 / 3:
            rate = 0.81
        offset = random.randint(0, max_offset)
        stretched = librosa.effects.time_stretch(sample, rate)
        augmented_sample = np.pad(stretched, (offset, 0),
                                  mode='constant', constant_values=0)
        augmented_samples.append(augmented_sample)
    return augmented_samples


def write_audio(signal, dataset, label, file):
    """Write out a sample according to a tree structure: the 1st level is
        the sample label and the second level are the samples.
        Args:
            signal: the sample to write out
            dataset: the name of the sample set (i.e. train, test...)
            label: the sample label
            file: the sample name
        Returns:
            None
    """
    audio_folder = os.path.join(config.output_data, dataset, label)
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)
    soundfile.write(os.path.join(audio_folder, file), signal, config.sr)


def move_audio(file, label):
    """Move a sample according to a tree structure: the 1st level is
        the sample label and the second level are the samples.
        Args:
            label: the sample label
            file: the sample name
        Returns:
            None
    """
    parent, _ = os.path.split(file)
    destination = os.path.join(parent, label)
    if not os.path.exists(destination):
        os.makedirs(destination)
    shutil.move(file, destination)


def random_train_test_split():
    """Performs random split of the raw dataset into train and test"""
    files = glob(os.path.join(config.input_data, "*.wav"))
    files.sort()  # sort for replicability
    labels = files_to_labels(files)
    return train_test_split(
        files, labels, test_size=0.2, random_state=101)


def pre_process_dataset():
    """Write out the pre-processed dataset and augment samples for training"""
    x_train, x_test, y_train, y_test = random_train_test_split()
    for file, label in zip(x_train, y_train):
        signal, _sr = librosa.load(file, sr=None)
        assert _sr == config.sr
        _, name = os.path.split(file)
        for idx, augmented_sample in enumerate(augment_sample(signal, config.n_augm)):
            write_audio(augmented_sample, "train", label, "{i}_{n}".format(i=idx, n=name))
    for file, label in zip(x_test, y_test):
        signal, _sr = librosa.load(file, sr=None)
        _, name = os.path.split(file)
        assert _sr == config.sr
        write_audio(signal, "test", label, name)


if __name__ == '__main__':
    pre_process_dataset()
