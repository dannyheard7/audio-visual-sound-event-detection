# Extract Video ids and start/end times from groundtruth csv file. Needed for evaluation set

from csv import reader

with open('evaluation_set.csv', 'w') as out, open('groundtruth_strong_label_evaluation_set.csv', 'r') as in_file:
    for row in reader(in_file):
        audio_file_name = row[0]
        split_dot = audio_file_name.split(".")[0]
        k = split_dot.rfind("_")
        video_id = split_dot[:k]

        audio_file_name_split = audio_file_name.split("_")

        start_time =  audio_file_name_split[1]
        k =  audio_file_name_split[2].rfind(".wav")
        end_time = audio_file_name_split[2][:k]
        print(video_id + ',' + start_time + ',' + end_time, file=out)
