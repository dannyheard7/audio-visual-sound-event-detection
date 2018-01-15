# Extract Video ids and start/end times from groundtruth csv file. Needed for evaluation set

from csv import reader

with open('evaluation_set.csv', 'w') as out, open('groundtruth_strong_label_evaluation_set.csv', 'r') as in_file:
    for row in reader(in_file):
        audio_file_name = row[0]
        split_dot = audio_file_name.split(".")[0]
        k = split_dot.rfind("_")
        video_id = split_dot[:k]


        wav_pos =  audio_file_name.rfind(".wav")
        last_underscore = audio_file_name.rfind("_") 
        end_time = audio_file_name[(last_underscore + 1):wav_pos].rstrip()

        audio_file_name_split = audio_file_name[:last_underscore]
        second_last_underscore = audio_file_name_split.rfind("_")

        start_time =  audio_file_name_split[(second_last_underscore + 1):last_underscore]

        classes = row[-1]
        
        print(video_id + ',' + start_time + ',' + end_time + ',' + classes, file=out)
