from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
import os

def timestamp_to_seconds(timestamp_str):
    hours, minutes, seconds = timestamp_str.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
#  
# --> 
timestamp_from = "00:35:51.829"
timestamp_till = "00:35:55.778"

#timestamp_till = 
timestamp_in_seconds_input = timestamp_to_seconds(timestamp_from)
timestamp_in_seconds_output = timestamp_to_seconds(timestamp_till)


video_file_dir = r'D:\BSL_project\bsl_dataset\BOBSL\bobsl\videos\5207824799325488412.mp4' 
output_dir = r'D:\BSL_project\bsl_dataset\manual-script\sorry\sorry_92.mp4'

ffmpeg_extract_subclip(video_file_dir, timestamp_in_seconds_input, timestamp_in_seconds_output, targetname=output_dir)