import config
import numpy as np
import librosa
import audio_manipulations
import wave
import os


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

def read_file(wav_path, sample_rate=config.SAMPLE_RATE):
    return librosa.load(str(wav_path), sr=sample_rate)

def get_features(audio_data, sample_rate=config.SAMPLE_RATE):
    audio = audio_manipulations.remove_noise(audio_data, sample_rate)
    features = np.empty((0))

    # f0min = librosa.note_to_hz('C2') #todo - jeśli odpalimy z f0, to strasznie długo to się wykonuje
    # f0max = librosa.note_to_hz('C7')
    # f0_val, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=f0min, fmax=f0max)
    # features = np.hstack((features, f0_val))
    # jitter_val = np.mean(np.abs(np.diff(f0_val[voiced_flag])))
    # features = np.hstack((features, jitter_val))
    
    # f0_mean = np.mean(f0_val)
    # features = np.hstack([features, f0_mean])
    
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

    centroid_val = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate).T, axis=0)
    features = np.hstack((features, centroid_val))
    
    flatness_val = np.mean(librosa.feature.spectral_flatness(y=audio).T, axis=0)
    features = np.hstack((features, flatness_val))

    rolloff_val = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate).T, axis=0)
    features = np.hstack((features, rolloff_val))

    return features

def extract_features_eatd(folder_name, audio_features, labels, prefix):
    if not os.path.exists(os.path.join(prefix, '{}/positive_out.wav'.format(folder_name))):
        return
    print('Extracting features file: ', folder_name)

    positive_file = wave.open(os.path.join(prefix, '{}/positive_out.wav'.format(folder_name)))
    sr1 = positive_file.getframerate()
    nframes1 = positive_file.getnframes()
    wave_data1 = np.frombuffer(positive_file.readframes(nframes1), dtype=np.short).astype(np.float64)

    neutral_file = wave.open(os.path.join(prefix, '{}/positive_out.wav'.format(folder_name)))
    sr2 = neutral_file.getframerate()
    nframes2 = neutral_file.getnframes()
    wave_data2 = np.frombuffer(neutral_file.readframes(nframes2), dtype=np.short).astype(np.float64)

    negative_file = wave.open(os.path.join(prefix, '{}/positive_out.wav'.format(folder_name)))
    sr3 = negative_file.getframerate()
    nframes3 = negative_file.getnframes()
    wave_data3 = np.frombuffer(negative_file.readframes(nframes3), dtype=np.short).astype(np.float64)
    
    with open(os.path.join(prefix, '{}/label.txt'.format(folder_name)), 'r') as label_file:
        raw_SDS_score = label_file.readline().strip()

    if wave_data1.shape[0] < 1:
        wave_data1 = np.array([1e-4] * sr1 * 5)
    if wave_data2.shape[0] < 1:
        wave_data2 = np.array([1e-4] * sr2 * 5)
    if wave_data3.shape[0] < 1:
        wave_data3 = np.array([1e-4] * sr3 * 5)

    features1 = get_features(wave_data1, sr1)
    features2 = get_features(wave_data2, sr2)
    features3 = get_features(wave_data3, sr3)
    audio_features.append([features1, features2, features3])
    
    is_depression = 1 if float(raw_SDS_score) > 42 else 0
    labels.append([folder_name, raw_SDS_score, is_depression])

