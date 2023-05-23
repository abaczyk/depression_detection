import os
import wave
from sys import prefix
import audioread
import numpy as np
import librosa
import values
import wave


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


def calculate_speech_rate(signal, sr, len_s, rms_val):
    silence_threshold = 0.01
    silence_segments = librosa.effects.split(signal, top_db=silence_threshold)

    # Filter out silence segments shorter than a certain duration (adjust as needed)
    min_silence_duration = 0.1
    silence_segments = [segment for segment in silence_segments if segment[1] - segment[0] >= min_silence_duration]

    total_silence_duration = sum(segment[1] - segment[0] for segment in silence_segments)
    speech_duration = len_s - total_silence_duration

    words = len(signal) / sr / 60
    speech_rate = words / speech_duration
    return speech_rate


def calculate_formant_freq_bandwidths(mfcc_val):
    delta_mfcc = librosa.feature.delta(mfcc_val)
    delta2_mfcc = librosa.feature.delta(mfcc_val, order=2)

    mfcc_mean = np.mean(mfcc_val, axis=1)
    delta_mfcc_mean = np.mean(delta_mfcc, axis=1)
    delta2_mfcc_mean = np.mean(delta2_mfcc, axis=1)

    formants = librosa.lpc(delta2_mfcc_mean, order=5)
    formant_freqs = np.abs(np.angle(np.roots(formants)))
    formant_bw = -np.log(np.abs(formants))

    return (formant_freqs[:3], formant_bw[1:4])


def wav2vlad(wave_data, sr, len_s):
    global cluster_size
    signal = wave_data
    melspec = librosa.feature.melspectrogram(y=signal, n_mels=80, sr=sr).astype(np.float32).T
    melspec = np.log(np.maximum(1e-6, melspec))
    mel_feature_size = melspec.shape[1]
    mel_max_samples = melspec.shape[0]
    mel_output_dim = cluster_size * 16
    mel_features = (mel_feature_size, mel_max_samples, cluster_size, mel_output_dim)

    f0min = librosa.note_to_hz('C2')
    f0max = librosa.note_to_hz('C7')

    mfcc_melspec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=80)
    mfcc_coefficients = librosa.feature.mfcc(S=librosa.power_to_db(mfcc_melspec), n_mfcc=40)
    mfcc_val = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    f0_val = librosa.yin(y=signal, fmin=f0min, fmax=f0max)
    f0_val, voiced_flag, _ = librosa.pyin(signal, fmin=f0min, fmax=f0max)
    f0_mean = np.mean(f0_val)
    rms_val = librosa.feature.rms(y=signal)  # energy value
    amplitude = np.abs(librosa.stft(signal))
    local_minima = librosa.util.localmin(amplitude)
    shimmer_val = np.mean(amplitude[local_minima])
    jitter_val = np.mean(np.abs(np.diff(f0_val[voiced_flag])))

    formant_freqs, formant_bw = calculate_formant_freq_bandwidths(mfcc_val)

    flux = librosa.onset.onset_strength(y=signal, sr=sr)
    centroid_val = librosa.feature.spectral_centroid(y=signal, sr=sr)
    bandwidth_val = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    contrast_val = librosa.feature.spectral_contrast(y=signal, sr=sr)
    flatness_val = librosa.feature.spectral_flatness(y=signal)
    rolloff_val = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    speech_rate = calculate_speech_rate(signal, sr, len_s, rms_val)

    return mel_features

def extract_features(number, audio_features, targets, path):
    global max_len, min_len
    if not os.path.exists(os.path.join(prefix, '{1}/t_{0}/positive_out.wav'.format(number, path))):
        return
    positive_file = wave.open(os.path.join(prefix, '{1}/t_{0}/positive_out.wav'.format(number, path)))
    sr1 = positive_file.getframerate()
    nframes1 = positive_file.getnframes()
    wave_data1 = np.frombuffer(positive_file.readframes(nframes1), dtype=np.short).astype(np.float64)
    len1 = nframes1 / sr1

    neutral_file = wave.open(os.path.join(prefix, '{1}/t_{0}/neutral_out.wav'.format(number, path)))
    sr2 = neutral_file.getframerate()
    nframes2 = neutral_file.getnframes()
    wave_data2 = np.frombuffer(neutral_file.readframes(nframes2), dtype=np.short).astype(np.float64)
    len2 = nframes2 / sr2

    negative_file = wave.open(os.path.join(prefix, '{1}/t_{0}/negative_out.wav'.format(number, path)))
    sr3 = negative_file.getframerate()
    nframes3 = negative_file.getnframes()
    wave_data3 = np.frombuffer(negative_file.readframes(nframes3), dtype=np.short).astype(np.float64)
    len3 = nframes3 / sr3

    for l in [len1, len2, len3]:
        if l > max_len:
            max_len = l
        if l < min_len:
            min_len = l

    with open(os.path.join(prefix, '{1}/t_{0}/new_label.txt'.format(number, path))) as fli:
        target = float(fli.readline())

    if wave_data1.shape[0] < 1:
        wave_data1 = np.array([1e-4] * sr1 * 5)
    if wave_data2.shape[0] < 1:
        wave_data2 = np.array([1e-4] * sr2 * 5)
    if wave_data3.shape[0] < 1:
        wave_data3 = np.array([1e-4] * sr3 * 5)

    mel_data1 = wav2vlad(wave_data1, sr1, len1)
    mel_data2 = wav2vlad(wave_data2, sr2, len2)
    mel_data3 = wav2vlad(wave_data3, sr3, len3)
    audio_mel_features.append([mel_data1, mel_data2, mel_data3])
    #audio_features.append([features_data1, features_data2, features_data3])

    targets.append(target)
audio_mel_features = []
audio_features = []
zip_directory = values.zip_directory
read(zip_directory)
sr = values.EXPERIMENT_DETAILS['SAMPLE_RATE']


'''

In audio_files we have pathes to files to operate in a simple way

'''




