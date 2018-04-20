audio_data_location = "/fastdata/aca14dh/DCASE-audio-data/"
audio_training_data_location = audio_data_location + "training/"
audio_testing_data_location = audio_data_location + "testing/"
audio_evaluatuib_data_location = audio_data_location + "testing/"
audio_file_extension = ".wav"


video_data_location = "/fastdata/aca14dh/DCASE-video-data/"
video_training_data_location = video_data_location + "training_set/"
video_testing_data_location = video_data_location + "testing_set/"
video_evaluation_data_location = video_data_location + "evaluation_set/"
video_file_extension = ".mp4"

video_training_frames_location = video_data_location + "frames/training/"
video_testing_frames_location = video_data_location + "frames/testing/"
video_evaluation_frames_location = video_data_location + "frames/evaluation/"
video_frames_extension = ".png"

training_data_csv_file = "metadata/training_set.csv"
testing_data_csv_file = "metadata/testing_set.csv"
evaluation_data_csv_file = "metadata/evaluation_set.csv"


num_components_to_keep = 32

labels = ['Train horn', 'Air horn, truck horn', 'Car alarm', 'Reversing beeps',
       'Bicycle', 'Skateboard', 'Ambulance (siren)',
       'Fire engine, fire truck (siren)', 'Civil defense siren',
       'Police car (siren)', 'Screaming', 'Car', 'Car passing by', 'Bus',
       'Truck', 'Motorcycle', 'Train']
lbs = labels
ids = ['/m/0284vy3', '/m/05x_td', '/m/02mfyn', '/m/02rhddq', '/m/0199g', 
       '/m/06_fw', '/m/012n7d', '/m/012ndj', '/m/0dgbq', '/m/04qvtq', 
       '/m/03qc9zr', '/m/0k4j', '/t/dd00134', '/m/01bjv', '/m/07r04', 
       '/m/04_sv', '/m/07jdr']

idx_to_id = {index: id for index, id in enumerate(ids)}
id_to_idx = {id: index for index, id in enumerate(ids)}
idx_to_lb = {index: lb for index, lb in enumerate(lbs)}
lb_to_idx = {lb: index for index, lb in enumerate(lbs)}
num_classes = len(lbs)

