#!/usr/env/bin python3

# THIS SCRIPT IS DERIVED FROM: https://github.com/google-research/chirp/tree/main/chirp/birb_sep_paper
import argparse
import tensorflow
import numpy as np
import os
import librosa

from scipy.io import wavfile

from chirp.birb_sep_paper import model_utils

tf = tensorflow.compat.v1

def separate(input_file, output_folder, model_path):

    separator = model_utils.load_separation_model(model_path)

    # Open the audio file with librosa
    arr, sr = librosa.load(input_file, sr=22050)

    # The sep_chunks will have shape [Audio duration / window_size, 4 Output channels, window_size*sample_rate (resampled at 22050Hz?)].
    sep_chunks, raw_chunks = model_utils.separate_windowed(arr, separator, hop_size_s=5.0, window_size_s=5)

    # Save the channels of sep_chunks independantly
    for channel_number in range(4):
        channel_data = sep_chunks[:, channel_number, :] 
        channel_data_1d = np.reshape(channel_data, -1)  # Reshape the sliced array to 1D for saving as wav file
        outname = os.path.join(output_folder, f'channel_{channel_number}.wav')
        wavfile.write(outname, sr, channel_data_1d)

if __name__== "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        help="Path to the file to analyze",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--output_folder",
        help="Path to folder that stores the output channels",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--model_path",
        help="Path to the model (can be the 4 or 8 channels, default is the 4 channel)",
        default="/app/bird_mixit_model_checkpoints/output_sources4/",
        required=False,
        type=str,
    )

    cli_args = parser.parse_args()

    separate(
        input_file=cli_args.input_file,
        output_folder=cli_args.output_folder,
        model_path=cli_args.model_path
    )

