
import scipy
import values
import os
import pickle
import sys
import librosa
import numpy as np
import time
import soundfile as sf

'''

In this file we: remove audio segments from original file, process all transcriptions, 
calculate the maximum and minimum length of audio in the dataset, delete noises, remove 
words from transcriptions 

'''
'''

Function to remove audio segments from wav

'''
def remove_segments(data, timings, sr, mode=False):
    timings = np.array(timings, float)
    samples = timings * sr
    samples = np.array(samples, int)
    pointer = 0
    if mode:
        updated_audio = data[0:samples[0][1]]
    else:
        for i in samples:
            if pointer == 0:
                updated_audio = data[i[0]:i[1]]
                pointer += 1
            else:
                updated_audio = np.hstack((updated_audio, data[i[0]:i[1]]))

    return updated_audio


'''

Function to process file transcriptions

'''
def transcript_file_processing(transcript_paths, current_dir,
                               mode_for_bkgnd=False, remove_background=True):

    on_off_times = []
    interrupt = {373: [395, 428]}
    misaligned = {318: 34.319917,
                  321: 3.8379167,
                  341: 6.1892,
                  362: 16.8582}
    special_case = interrupt
    special_case_3 = misaligned
    for i in transcript_paths:
        trial = i.split('/')[-2]
        trial = int(trial.split('_')[0])
        with open(i, 'r') as file:
            data = file.readlines()
        ellies_first_intro = 0
        inter = []
        for j, values in enumerate(data):
            file_end = len(data) - 1
            if j == 0:
                pass
            else:
                temp = values.split()[0:3]
                if trial in special_case_3:
                    if len(temp) == 0:
                        time_start = time_end = 0
                    else:
                        time_start = float(temp[0]) + special_case_3[trial]
                        time_end = float(temp[1]) + special_case_3[trial]
                else:
                    if len(temp) == 0:
                        time_start = time_end = 0
                    else:
                        time_start = float(temp[0])
                        time_end = float(temp[1])
                if len(values) > 1:
                    sync = values.split()[-1]
                else:
                    sync = ''
                if sync == '[sync]' or sync == '[syncing]':
                    sync = True
                else:
                    sync = False
                if len(temp) > 0 and temp[-1] == ('Participant' or
                                                  'participant'):
                    if sync:
                        pass
                    else:
                        if trial in special_case:
                            inter_start = special_case[trial][0]
                            inter_end = special_case[trial][1]
                            if time_start < inter_start < time_end:
                                inter.append([time_start, inter_start - 0.01])
                            elif time_start < inter_end < time_end:
                                inter.append([inter_end + 0.01, time_end])
                            elif inter_start < time_start < inter_end:
                                pass
                            elif inter_start < time_end < inter_end:
                                pass
                            elif time_end < inter_start or time_start > inter_end:
                                inter.append(temp[0:2])
                        else:
                            if 0 < j:
                                prev_val = data[j-1].split()[0:3]
                                if len(prev_val) == 0:
                                    if j - 2 > 0:
                                        prev_val = data[j-2].split()[0:3]
                                    else:
                                        prev_val = ['', '', 'Ellie']
                                if j != file_end:
                                    next_val = data[j+1].split()[0:3]
                                    if len(next_val) == 0:
                                        if j+1 != file_end:
                                            next_val = data[j + 2].split()[0:3]
                                        else:
                                            next_val = ['', '', 'Ellie']
                                else:
                                    next_val = ['', '', 'Ellie']
                                if prev_val[-1] != ('Participant' or
                                                    'participant'):
                                    holding_start = time_start
                                elif prev_val[-1] == ('Participant' or
                                                      'participant'):
                                    pass
                                if next_val[-1] == ('Participant' or
                                                    'participant'):
                                    continue
                                elif next_val[-1] != ('Participant' or
                                                      'participant'):
                                    holding_stop = time_end
                                    inter.append([str(holding_start),
                                                  str(holding_stop)])
                            else:
                                inter.append([str(time_start), str(time_end)])
                elif not temp or temp[-1] == ('Ellie' or 'ellie') and not \
                        mode_for_bkgnd and not sync:
                    pass
                elif temp[-1] == ('Ellie' or 'ellie') and mode_for_bkgnd \
                        and not sync:
                    if ellies_first_intro == 0:
                        inter.append([0, str(time_start - 0.01)])
                        break
                elif temp[-1] == ('Ellie' or 'ellie') and sync:
                    if remove_background or mode_for_bkgnd:
                        pass
                    else:
                        inter.append([str(time_start), str(time_end)])
                        ellies_first_intro = 1
                else:
                    print('Error, Transcript file does not contain '
                          'expected values')
                    print(f"File: {i}, This is from temp: {temp[-1]}")
                    sys.exit()
        on_off_times.append(inter)

    with open(os.path.join(current_dir, 'on_off_times.pickle'), 'wb') as f:
        pickle.dump(on_off_times, f)

        return on_off_times


'''

Function to calculate the maximum and minimum length of audio in the dataset

'''
def max_min_values(current_directory, win_size, hop_size, audio_paths,
                   on_off_times, mode_for_background, librosa=None):
    max_value = 0
    min_value = 1e20
    output_data = np.zeros((len(audio_paths), 4))
    for iterator, filename in enumerate(audio_paths):
        audio_data, sample_rate = librosa.load(filename, sr=None)
        mod_audio = remove_segments(audio_data,
                                      on_off_times[iterator],
                                      sample_rate,
                                      mode_for_background)
        number_samples = int(mod_audio.shape[0])
        time_in_mins = number_samples / (sample_rate * 60)

        if sys.platform == 'win32':
            folder_name = filename.split('\\')[-2]
        else:
            folder_name = filename.split('/')[-2]

        path = os.path.join(current_directory, values.FEATURE_FOLDERS[0],
                            folder_name + '_audio_data.npy')
        np.save(path, mod_audio)
        folder_number = int(folder_name[0:3])

        if audio_data.ndim > 1:
            input('2 Channels were detected')

        if number_samples > max_value:
            max_value = number_samples
        if number_samples < min_value:
            min_value = number_samples

        output_data[iterator, :] = [sample_rate, number_samples,
                                    time_in_mins, folder_number]

    path = os.path.join(current_directory, 'meta_data')
    np.save(path, output_data)

    total_windows_in_file_max = (max_value + (hop_size * 2)) - win_size
    total_windows_in_file_max = (total_windows_in_file_max // hop_size) + 1
    total_windows_in_file_min = (min_value + (hop_size * 2)) - win_size
    total_windows_in_file_min = (total_windows_in_file_min // hop_size) + 1

    return max_value, min_value, sample_rate, total_windows_in_file_max, \
           total_windows_in_file_min, output_data


def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)

def removeNoise(
    audio_clip,
    noise_clip,
    n_grad_freq,
    n_grad_time,
    n_fft,
    win_length,
    hop_length,
    n_std_thresh,
    prop_decrease,
    verbose=False,
    visual=False,
):
    if verbose:
        start = time.time()
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    if verbose:
        start = time.time()
    if verbose:
        start = time.time()
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    if verbose:
        start = time.time()
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    print(noise_thresh, mask_gain_dB)
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    sig_mask = sig_stft_db < db_thresh
    if verbose:
        start = time.time()
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    if verbose:
        start = time.time()
    sig_stft_db_masked = (
            sig_stft_db * (1 - sig_mask)
            + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask)
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
            1j * sig_imag_masked
    )
    if verbose:
        start = time.time()
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)

    return recovered_signal


def remove_words_symbols(string):
    if isinstance(string, str):
        temp = string.split()
    else:
        temp = string[:]
    marked_for_removal = []
    for split_string in temp:
        words_to_remove = {0: 'xxx', 1: 'xxxx', 2: ' ', 3: '  ', 4: '   ', 5: '    ',
                           6: '     '}
        symbols_to_remove = ['<', '>', '[', ']']
        if split_string in words_to_remove.values():
            marked_for_removal.append(split_string)
        else:
            if any(symb in split_string for symb in
                   symbols_to_remove):
                marked_for_removal.append(split_string)
    for pointer in range(len(marked_for_removal)):
        string = string.replace(marked_for_removal[pointer], '')

    if string in words_to_remove.values():
        string = string.replace(string, '')

    return string

# def remove_noise_other_way(input_file):
     # audio_data, sample_rate = sf.read(input_file)
     # reduced_noise = nr.reduce_noise(audio_data, sample_rate)
