import pafy
import time
import datetime
import itertools
import os
import sys
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

#Format video - Convert to H.264 and trim
def format_video(input_video_file, output_video_file, start_time, duration):
	cmdstring = "ffmpeg -loglevel panic -i {0} -ss {1} -t {2} {3}".format(input_video_file, start_time.split(".")[0], duration, output_video_file)
	print(cmdstring)
	os.system(cmdstring)

def multi_run_wrapper(args):
   return download_video_method(*args)

#Method to download video - Downloads the best video available for video id, calls the formatting & segmenting video function based on start and end time. 
def download_video_method(line,csv_file):
	query_id = line.split(",")[0];
	start_seconds = line.split(",")[1];
	end_seconds = line.split(",")[2];
	video_duration = float(end_seconds.split(".")[0]) - float(start_seconds.split(".")[0])
	#positive_labels = ','.join(line.split(",")[3:]);
	print("Query -> " + query_id)

	segmented_folder = sys.argv[1].split('.csv')[0].split("/")[-1] + "_" + csv_file.split('.csv')[0] +  "_" + "video_formatted_and_segmented_downloads"
		
	if not os.path.exists(segmented_folder):
		os.makedirs(segmented_folder)
	
	path_to_segmented_video = segmented_folder + "/Y" + query_id.rstrip() + '_' + start_seconds.rstrip() + '_' + end_seconds.rstrip() +  ".mp4"	

	if not os.path.isfile(path_to_segmented_video):
		try:
			video = pafy.new(query_id)
			bestvideo = video.getbestvideo() # Maybe some videos don't have streams without audio? so can try downloading with audio

			#.csv - split - to get the folder information. As path is also passed for the video - creating the directory from the path where this video script is present. THus using second split to get the folder name where output files shall be downloaded
			output_folder = sys.argv[1].split('.csv')[0].split("/")[-1] + "_" + csv_file.split('.csv')[0] +  "_" + "video_downloaded"
			
			#print output_folder
			if not os.path.exists(output_folder):
				os.makedirs(output_folder)
			
			path_to_download = output_folder + "/Y" + query_id + "." + bestvideo.extension
			bestvideo.download(path_to_download, quiet=True)			
				
			format_video(path_to_download, path_to_segmented_video, start_seconds, video_duration)

			if os.path.isfile(path_to_segmented_video):
				#Remove the original video 
				delete_original_file_cmd = "rm -f {0}".format(path_to_download)
				os.system(delete_original_file_cmd)
			else:
				print("Error converting file: " + path_to_download)

			ex1 = ""
		except Exception as ex:
			ex1 = str(ex) + ',' + str(query_id)
			print ("Error is ---> " + str(ex))  
		return ex1


def download_video(csv_file, timestamp):	
	segments_info = []
	
	with open(csv_file, "r") as segments_info_file:	
		for line in segments_info_file:
			segments_info.append((line, csv_file))

	cpu_count = multiprocessing.cpu_count()	

	P = multiprocessing.Pool(processes=cpu_count)
	exception = P.map(multi_run_wrapper, segments_info)

	for item in exception:
		if item:
			print(str(item))

	P.close()
	P.join()

if __name__ == "__main__":
	if len(sys.argv) !=2:
		print ('takes arg1 as csv file to downloaded')
	else:
		
		ts = time.time()
		timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')			
		download_video(sys.argv[1], timestamp)
	
