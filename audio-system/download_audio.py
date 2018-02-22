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
def format_audio(input_audio_file, output_audio_file, start_time, duration):
	temp_audio_file = output_audio_file.split('.wav')[0] + '_temp.wav'
	
	cmdstring = "ffmpeg -loglevel panic -i %s -ac 1 -ar 44100 %s" %(input_audio_file,temp_audio_file)
	os.system(cmdstring)
	
	cmdstring1 = "sox %s -G -b 16 -r 44100 %s" %(temp_audio_file,output_audio_file)
	os.system(cmdstring1)

	cmdstring2 = "mv %s %s" %(output_audio_file, temp_audio_file)

	cmdstring = "sox %s %s trim %s %s" %(temp_audio_file, output_audio_file, start_time, duration)
	os.system(cmdstring)

	cmdstring2 = "rm -rf %s" %(temp_audio_file)
	os.system(cmdstring2)
	

def multi_run_wrapper(args):
   return download_audio_method(*args)

#Method to download audio - Downloads the best audio available for audio id, calls the formatting audio function and then segments the audio formatted based on start and end time. 
def download_audio_method(line,csv_file):
	query_id = line.split(",")[0];
	start_seconds = line.split(",")[1];
	end_seconds = line.split(",")[2];
	audio_duration = float(end_seconds) - float(start_seconds)
	#positive_labels = ','.join(line.split(",")[3:]);
	print("Query -> " + query_id)
	#print "start_time -> " + start_seconds
	#print "end_time -> " + end_seconds
	#print "positive_labels -> " + positive_labels
	url = "https://www.youtube.com/watch?v=" + query_id


	segmented_folder = sys.argv[1].split('.csv')[0].split("/")[-1] +  "_audio_formatted_and_segmented_downloads"

	if not os.path.exists(segmented_folder):
		os.makedirs(segmented_folder)

	path_to_segmented_audio = segmented_folder + "/Y" + query_id.rstrip() + '_' + start_seconds.rstrip() + '_' + end_seconds.rstrip() +  ".wav"

	if not os.path.isfile(path_to_segmented_audio):
		try:
			video = pafy.new(url)
			bestaudio = video.getbestaudio()

			#.csv - split - to get the folder information. As path is also passed for the video - creating the directory from the path where this video script is present. THus using second split to get the folder name where output files shall be downloaded
			output_folder = sys.argv[1].split('.csv')[0].split("/")[-1] + "_" + csv_file.split('.csv')[0] +  "_" + "audio_downloaded"
			
			#print output_folder
			if not os.path.exists(output_folder):
				os.makedirs(output_folder)
			
			path_to_download = output_folder + "/Y" + query_id + "." + bestaudio.extension
			bestaudio.download(path_to_download, quiet=True)			
				
			format_audio(path_to_download, path_to_segmented_audio, start_seconds, audio_duration)

			if os.path.isfile(path_to_segmented_audio):
				#Remove the original video 
				delete_original_file_cmd = "rm -f {0}".format(path_to_download)
				os.system(delete_original_file_cmd)
			else:
				print("Error converting file: " + path_to_download)

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
	
