
import soundfile as sf
import noisereduce as nr
import os
import values
import feature_extraction

def read(dataset_path):
    transcript_paths = [
        os.path.join(dataset_path, i, j)
        for i in os.listdir(dataset_path)
        if i.endswith('_P')
        for j in os.listdir(os.path.join(dataset_path, i))
        if 'TRANSCRIPT.csv' in j
    ]
    return transcript_paths


def append_list(dataset_path):
    folder_list = []
    transcript_paths = []
    for i in os.listdir(dataset_path):
        if i.endswith('_P'):
            folder_list.append(i)
            for j in os.listdir(os.path.join(dataset_path, i)):
                if 'TRANSCRIPT' in j:
                        transcript_paths.append(os.path.join(dataset_path, i, j))

    return transcript_paths


def transcript_file_processing(current_dir,
                                   mode_for_bkgnd=False, remove_background=True):

        on_off_times = []
        interrupt = {373: [395, 428]}
        misaligned = {318: 34.319917,
                      321: 3.8379167,
                      341: 6.1892,
                      362: 16.8582}
        special_case = interrupt
        special_case_3 = misaligned
        for i in current_dir:
            trial = i.split('\\')[-1]
            print(trial)
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
                                    prev_val = data[j - 1].split()[0:3]
                                    if len(prev_val) == 0:
                                        if j - 2 > 0:
                                            prev_val = data[j - 2].split()[0:3]
                                        else:
                                            prev_val = ['', '', 'Ellie']
                                    if j != file_end:
                                        next_val = data[j + 1].split()[0:3]
                                        if len(next_val) == 0:
                                            if j + 1 != file_end:
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

        with open(os.path.join('C:/Users/admin/Desktop/My_preproces', 'on_off_times.pickle'), 'wb') as f:
            pickle.dump(on_off_times, f)

        return on_off_times

def remove_noise(input_dir):
     output_dir = '/not_noise/'
     audio_data, sample_rate = sf.read(input_dir)
     
     os.makedirs(output_dir, exist_ok=True)
     file_name = os.path.basename(input_dir)
     
     file_name_without_extension = os.path.splitext(file_name)[0]
     output_wav = file_name_without_extension + '_last_part.wav'
     
     volume_gain = 2.0
     reduced_noise = nr.reduce_noise(audio_data, sample_rate)
     volume_red = reduced_noise * volume_gain
     
     output_file = os.path.join(output_dir, output_wav)
     sf.write(output_file, volume_red , sample_rate)
        
zip_directory = values.zip_directory
input_dir = feature_extraction.read(zip_directory)

'''

e.g. remove_segments(audio_data,on_off_times[0],sample_rate, mode=False)
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


transcript_file_processing(read(values.zip_directory),mode_for_bkgnd=False, remove_background=True)
for i in input_dir:
    remove_noise(i)
