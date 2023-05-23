import os
import wave
from sys import prefix
import numpy as np
import librosa
import values
import wave
import speech_recognition as sr
import soundfile as sf


cluster_size = 16
min_len = 100
max_len = -1

def read(dataset_path):
    folder_list = []
    audio_files = []
    for i in os.listdir(dataset_path):
        if i.endswith('_P'):
            folder_list.append(i)
            for j in os.listdir(os.path.join(dataset_path, i)):
                if 'wav' in j:
                    audio_files.append(os.path.join(dataset_path, i,j))
    return audio_files


audio_mel_features = []
audio_features = []
zip_directory = values.zip_directory
read(zip_directory)


def speech_rate(wav_path):
    audio, sample_rate = sf.read(wav_path)
    duration = librosa.get_duration(y=audio, sr=sample_rate)
    speech_rate = (len(audio) / sample_rate) / (duration / 60)
    return speech_rate

def calculate_freq(wav_path):

    audio, sample_rate = sf.read(wav_path)
    lpc_coeff = librosa.lpc(audio, order=8)
    roots = np.roots(lpc_coeff)
    angles = np.angle(roots)
    frequencies = np.arctan2(np.imag(angles), np.real(angles)) * (sample_rate / (2 * np.pi))
    bandwidths = -1 / 2 * (sample_rate / (2 * np.pi)) * np.log(np.abs(roots))
    return frequencies, bandwidths

def wavtovlad(wav_path):

    global cluster_size
    audio, sample_rate = sf.read(wav_path)

    melspec = librosa.feature.melspectrogram(y=audio, n_mels=80, sr = sample_rate).astype(np.float32).T
    melspec = np.log(np.maximum(1e-6, melspec))
    mel_feature_size = melspec.shape[1]
    mel_max_samples = melspec.shape[0]
    mel_output_dim = cluster_size * 16
    mel_features = (mel_feature_size, mel_max_samples, cluster_size, mel_output_dim)

    f0min = librosa.note_to_hz('C2')
    f0max = librosa.note_to_hz('C7')

    mfcc_melspec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=80)
    mfcc_coefficients = librosa.feature.mfcc(S=librosa.power_to_db(mfcc_melspec), n_mfcc=40)
    mfcc_val = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

    f0_val = librosa.yin(y=audio, fmin=f0min, fmax=f0max)
    f0_val, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=f0min, fmax=f0max)
    f0_mean = np.mean(f0_val)
    rms_val = librosa.feature.rms(y=audio)  # energy value
    amplitude = np.abs(librosa.stft(audio))
    local_minima = librosa.util.localmin(amplitude)
    shimmer_val = np.mean(amplitude[local_minima])
    formant_freqs, formant_bw = calculate_freq(wav_path)

    flux = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    centroid_val = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    bandwidth_val = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    contrast_val = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    flatness_val = librosa.feature.spectral_flatness(y=audio)
    rolloff_val = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)





