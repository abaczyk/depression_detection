
import os
import numpy as np
import wave
import librosa





def calculate_speech_rate(signal, sr):
    duration = librosa.get_duration(y=signal, sr=sr)
    speech_rate = (len(signal) / sr) / (duration / 60)
    return speech_rate


def calculate_formant_freq_bandwidths(signal, sr):
    lpc_coeff = librosa.lpc(signal, order=8)
    roots = np.roots(lpc_coeff)
    angles = np.angle(roots)
    frequencies = np.arctan2(np.imag(angles), np.real(angles)) * (sr / (2 * np.pi))
    bandwidths = -1 / 2 * (sr / (2 * np.pi)) * np.log(np.abs(roots))
    return frequencies, bandwidths



def wav2vlad(wave_data, sr, len_s):
    global cluster_size
    signal = wave_data
    features = np.empty((0))

    # melspec = librosa.feature.melspectrogram(y=signal, n_mels=80, sr=sr).astype(np.float32).T
    # melspec = np.log(np.maximum(1e-6, melspec))
    # mel_feature_size = melspec.shape[1]
    # mel_max_samples = melspec.shape[0]
    # mel_output_dim = cluster_size * 16
    # mel_features = (mel_feature_size, mel_max_samples, cluster_size, mel_output_dim)

    # f0min = librosa.note_to_hz('C2')
    # f0max = librosa.note_to_hz('C7')
    # f0_val, voiced_flag, _ = librosa.pyin(signal, fmin=f0min, fmax=f0max)
    # f0_mean = np.mean(f0_val)
    # features = np.hstack((features, f0_val))
    # features = np.hstack((features, f0_mean))

    # mfcc_melspec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=80)
    # mfcc_coefficients = librosa.feature.mfcc(S=librosa.power_to_db(mfcc_melspec), n_mfcc=40)

    mfcc_val = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T, axis=0)
    features = np.hstack((features, mfcc_val))

    rms_val = np.mean(librosa.feature.rms(y=signal).T, axis=0)  # energy value
    features = np.hstack((features, rms_val))

    amplitude = np.abs(librosa.stft(signal))
    local_minima = librosa.util.localmin(amplitude)
    shimmer_val = np.mean(amplitude[local_minima])
    features = np.hstack((features, shimmer_val))

    formant_freqs, formant_bw = calculate_formant_freq_bandwidths(signal, sr)
    features = np.hstack((features, formant_freqs))

    # jitter_val = np.mean(np.abs(np.diff(f0_val[voiced_flag])))
    # features = np.hstack((features, jitter_val))

    speech_rate = calculate_speech_rate(signal, sr)
    features = np.hstack((features, speech_rate))

    # częstotliwość formantów i szerokość pasm (F1, F2, F3, B1, B2, B3)
    # formant_freqs, formant_bw = calculate_formant_freq_bandwidths(mfcc_val)
    # features = np.hstack((features, formant_freqs))
    # features = np.hstack((features, formant_bw))

    # flux = librosa.onset.onset_strength(y=signal, sr=sr)
    centroid_val = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr).T, axis=0)
    features = np.hstack((features, centroid_val))

    # bandwidth_val = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    # contrast_val = librosa.feature.spectral_contrast(y=signal, sr=sr)

    flatness_val = np.mean(librosa.feature.spectral_flatness(y=signal).T, axis=0)
    features = np.hstack((features, flatness_val))

    rolloff_val = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr).T, axis=0)
    features = np.hstack((features, rolloff_val))

    return features


def extract_features(number, audio_features, path, prefix):
    if not os.path.exists(os.path.join(prefix, '{1}/t_{0}/positive_out.wav'.format(number, path))):
        return
    print('Extracting features file: ', number)

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

    if wave_data1.shape[0] < 1:
        wave_data1 = np.array([1e-4] * sr1 * 5)
    if wave_data2.shape[0] < 1:
        wave_data2 = np.array([1e-4] * sr2 * 5)
    if wave_data3.shape[0] < 1:
        wave_data3 = np.array([1e-4] * sr3 * 5)


    features1 = wav2vlad(wave_data1, sr1, len1)
    features2 = wav2vlad(wave_data2, sr2, len2)
    features3 = wav2vlad(wave_data3, sr3, len3)
    audio_features.append([features1, features2, features3])






