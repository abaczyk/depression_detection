
import soundfile as sf
import noisereduce as nr
import os
import values
import feature_extraction


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

for i in input_dir:
    remove_noise(i)
