#Code Contributor - Ankit Shah - ankit.tronix@gmail.com
import pafy
import time
import datetime
import itertools
import os
import sys
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

#Format audio - 16 bit Signed PCM audio sampled at 44.1kHz
def format_audio(input_audio_file,output_audio_file):
	temp_audio_file = output_audio_file.split('.wav')[0] + '_temp.wav'
	cmdstring = "ffmpeg -loglevel panic -i %s -ac 1 -ar 44100 %s" %(input_audio_file,temp_audio_file)
	os.system(cmdstring)
	cmdstring1 = "sox %s -G -b 16 -r 44100 %s" %(temp_audio_file,output_audio_file)
	os.system(cmdstring1)

	if os.path.exists(output_audio_file):
		cmdstring2 = "rm -rf %s" %(temp_audio_file)
		os.system(cmdstring2)
	else:
		print("Error converting audio file: " + temp_audio_file)

#Trim audio based on start time and duration of audio. 
def trim_audio(input_audio_file,output_audio_file,start_time,duration):
	#print input_audio_file
	#print output_audio_file
	cmdstring = "sox %s %s trim %s %s" %(input_audio_file,output_audio_file,start_time,duration)
	os.system(cmdstring)

def multi_run_wrapper(args):
   return download_audio_method(*args)

#Method to download audio - Downloads the best audio available for audio id, calls the formatting audio function and then segments the audio formatted based on start and end time. 
def download_audio_method(line,csv_file):
	print(csv_file)
	query_id = line.split(",")[0];
	start_seconds = line.split(",")[1];
	end_seconds = line.split(",")[2];
	audio_duration = float(end_seconds) - float(start_seconds)

	print("Query -> " + query_id)

	url = "https://www.youtube.com/watch?v=" + query_id
	try:
		video = pafy.new(query_id)
		bestaudio = video.getbestaudio()
		#.csv - split - to get the folder information. As path is also passed for the audio - creating the directory from the path where this audio script is present. THus using second split to get the folder name where output files shall be downloaded
		output_folder = sys.argv[1].split('.csv')[0].split("/")[-1] + "_" + csv_file.split('.csv')[0] +  "_" + "audio_downloaded"
		#print output_folder
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)
		
		path_to_download = output_folder + "/Y" + query_id + "." + bestaudio.extension
		#print path_to_download
		bestaudio.download(path_to_download)
		
		formatted_folder = sys.argv[1].split('.csv')[0].split("/")[-1] + "_" + csv_file.split('.csv')[0] + "_" + "audio_formatted_downloaded"
		
		if not os.path.exists(formatted_folder):
			os.makedirs(formatted_folder)
		
		path_to_formatted_audio = formatted_folder + "/Y" + query_id + ".wav"
		
		format_audio(path_to_download,path_to_formatted_audio)
		
		#Trimming code
		segmented_folder = sys.argv[1].split('.csv')[0].split("/")[-1] + "_" + csv_file.split('.csv')[0] +  "_" + "audio_formatted_and_segmented_downloads"
		if not os.path.exists(segmented_folder):
			os.makedirs(segmented_folder)
		
		path_to_segmented_audio = segmented_folder + "/Y" + query_id + '_' + start_seconds + '_' + end_seconds +  ".wav"
		trim_audio(path_to_formatted_audio,path_to_segmented_audio,start_seconds,audio_duration)

		if os.path.exists(path_to_segmented_audio):
			#Remove the original audio 
			delete_original_file_cmd = "rm -f {0}".format(path_to_download)
			os.system(delete_original_file_cmd)

			# Remove the formatted audio
			delete_original_file_cmd = "rm -f {0}".format(path_to_formatted_audio)
			os.system(delete_original_file_cmd)
		else:
			print("Error converting audio file: " + path_to_download)

		ex1 = ""
	except Exception as ex:
		ex1 = str(ex) + ',' + str(query_id)
		print("Error is ---> " + str(ex))
	return ex1


def download_audio(csv_file,timestamp):	
	segments_info = []
	
	with open(csv_file, "r") as segments_info_file:	
		for line in segments_info_file:
			segments_info.append((line, csv_file))

	cpu_count = multiprocessing.cpu_count()	

	P = multiprocessing.Pool(processes=cpu_count)
	exception = P.map(multi_run_wrapper, segments_info)

	for item in exception:
		if item:
			print((str(item)))

	P.close()
	P.join()

if __name__ == "__main__":
	if len(sys.argv) !=2:
		print('takes arg1 as csv file to downloaded')
	else:
		
		ts = time.time()
		timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')			
		download_audio(sys.argv[1],timestamp)
	
