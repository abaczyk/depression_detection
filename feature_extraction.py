import config
import numpy as np
import librosa
import audio_manipulations


cluster_size = 16


def get_speech_rate(audio, sample_rate):
    duration = librosa.get_duration(y=audio, sr=sample_rate)
    speech_rate = (len(audio) / sample_rate) / (duration / 60)
    return speech_rate


def calculate_freq(audio, sample_rate):
    lpc_coeff = librosa.lpc(audio, order=8)
    roots = np.roots(lpc_coeff)
    angles = np.angle(roots)
    frequencies = np.arctan2(np.imag(angles), np.real(angles)) * (sample_rate / (2 * np.pi))
    bandwidths = -1 / 2 * (sample_rate / (2 * np.pi)) * np.log(np.abs(roots))
    return frequencies, bandwidths


def get_features(wav_path):
    audio = audio_manipulations.remove_noise(wav_path)
    sample_rate = config.SAMPLE_RATE
    features = np.empty((0))
    # melspec = librosa.feature.melspectrogram(y=audio, n_mels=80, sr = sample_rate).astype(np.float32).T
    # melspec = np.log(np.maximum(1e-6, melspec))
    # mel_feature_size = melspec.shape[1]
    # mel_max_samples = melspec.shape[0]
    # mel_output_dim = cluster_size * 16
    # mel_features = (mel_feature_size, mel_max_samples, cluster_size, mel_output_dim)

    # f0min = librosa.note_to_hz('C2') #todo - jeśli odpalimy z f0, to strasznie długo to się wykonuje
    # f0max = librosa.note_to_hz('C7')
    # f0_val, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=f0min, fmax=f0max)
    # features = np.hstack((features, f0_val)
 
    # f0_mean = np.mean(f0_val)
    # features = np.hstack([features, f0_mean])
    
    # mfcc_melspec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=80)
    mfcc_val = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    features = np.hstack((features, mfcc_val))

    rms_val = np.mean(librosa.feature.rms(y=audio).T, axis=0)  # energy value
    features = np.hstack((features, rms_val))

    amplitude = np.abs(librosa.stft(audio))
    local_minima = librosa.util.localmin(amplitude)
    shimmer_val = np.mean(amplitude[local_minima])
    features = np.hstack((features, shimmer_val))

    formant_freqs, formant_bw = calculate_freq(audio, sample_rate)
    features = np.hstack((features, formant_freqs))
    
    speech_rate = get_speech_rate(audio, sample_rate)
    features = np.hstack((features, speech_rate))

    # flux = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    centroid_val = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate).T, axis=0)
    features = np.hstack((features, centroid_val))

    # bandwidth_val = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    # contrast_val = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    
    flatness_val = np.mean(librosa.feature.spectral_flatness(y=audio).T, axis=0)
    features = np.hstack((features, flatness_val))

    rolloff_val = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate).T, axis=0)
    features = np.hstack((features, rolloff_val))

    return features

