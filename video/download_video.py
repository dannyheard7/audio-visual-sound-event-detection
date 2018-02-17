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

#Format video - Convert to H.264 and trim
def format_video(input_video_file, output_video_file, start_time, duration):
	cmdstring = "ffmpeg -loglevel panic -i {0} -ss 00:00:{1} -t 00:00:{2} {3}".format(input_video_file, start_time, duration, output_video_file)
	print(cmdstring)
	os.system(cmdstring)

def multi_run_wrapper(args):
   return download_video_method(*args)

#Method to download video - Downloads the best video available for video id, calls the formatting & segmenting video function based on start and end time. 
def download_video_method(line,csv_file):
	query_id = line.split(",")[0];
	start_seconds = line.split(",")[1];
	end_seconds = line.split(",")[2];
	video_duration = float(end_seconds) - float(start_seconds)
	#positive_labels = ','.join(line.split(",")[3:]);
	print("Query -> " + query_id)
	
	url = "https://www.youtube.com/watch?v=" + query_id

	segmented_folder = sys.argv[1].split('.csv')[0].split("/")[-1] + "_" + csv_file.split('.csv')[0] +  "_" + "video_formatted_and_segmented_downloads"
		
	if not os.path.exists(segmented_folder):
		os.makedirs(segmented_folder)
	
	path_to_segmented_video = segmented_folder + "/Y" + query_id + '_' + start_seconds + '_' + end_seconds.rstrip() +  ".mp4"	

	if not os.path.isfile(path_to_segmented_video):
		try:
			video = pafy.new(url)
			bestvideo = video.getbestvideo()

			#.csv - split - to get the folder information. As path is also passed for the video - creating the directory from the path where this video script is present. THus using second split to get the folder name where output files shall be downloaded
			output_folder = sys.argv[1].split('.csv')[0].split("/")[-1] + "_" + csv_file.split('.csv')[0] +  "_" + "video_downloaded"
			
			#print output_folder
			if not os.path.exists(output_folder):
				os.makedirs(output_folder)
			
			path_to_download = output_folder + "/Y" + query_id + "." + bestvideo.extension
			bestvideo.download(path_to_download)			
				
			format_video(path_to_download, path_to_segmented_video, start_seconds, video_duration)

			#Remove the original video 
			delete_original_file_cmd = "rm -f {0}".format(path_to_download)
			os.system(delete_original_file_cmd)

			ex1 = ""
		except Exception as ex:
			ex1 = str(ex) + ',' + str(query_id)
			print ("Error is ---> " + str(ex))
		return ex1

#Download video - Reads 3 lines of input csv file at a time and passes them to multi_run wrapper which calls download_video_method to download the file based on id.
#Multiprocessing module spawns 3 process in parallel which runs download_video_method. Multiprocessing, thus allows downloading process to happen in 40 percent of the time approximately to downloading sequentially - processing line by line of input csv file. 
def download_video(csv_file, timestamp):	
	error_log = 'error' + timestamp + '.log'
	
	with open(csv_file, "r") as segments_info_file:	
		with open(error_log, "a") as fo:
			
			for line in tqdm(segments_info_file):
				cpu_count = multiprocessing.cpu_count()
				lines_list = []
				line = (line, csv_file)
				lines_list.append(line)

				for cpu in range(0, cpu_count - 2):
					try:
						next_line = segments_info_file.next()
						next_line = (next_line, csv_file)
						lines_list.append(next_line)
					except:
						print ("end of file")
				
				#print lines_list
				P = multiprocessing.Pool(processes=cpu_count)
				exception = P.map(multi_run_wrapper, lines_list)
				
				for item in exception:
					if item:
						line = fo.writelines(str(item) +  '\n')

				P.close()
				P.join()
		fo.close()

if __name__ == "__main__":
	if len(sys.argv) !=2:
		print ('takes arg1 as csv file to downloaded')
	else:
		
		ts = time.time()
		timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')			
		download_video(sys.argv[1], timestamp)
	
