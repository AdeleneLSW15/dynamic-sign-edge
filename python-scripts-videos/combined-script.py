import re
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# Global variable to keep track of the ID
clip_id = 0

def timestamp_to_seconds(timestamp_str):
    hours, minutes, seconds = timestamp_str.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

def find_keyword_and_extract_clips(vtt_file_path, keyword, video_file_path, output_dir_base, start_clip_id):
    global clip_id  # Reference the global variable
    clip_id = start_clip_id  # Start clip ID from the passed argument

    timestamp_ranges = []
    with open(vtt_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    cues = re.split(r'\n\n+', content)
    
    for cue in cues:
        if re.search(keyword, cue, re.IGNORECASE):
            timestamp_match = re.search(r'^(\d{2}:\d{2}:\d{2}.\d{3}) --> (\d{2}:\d{2}:\d{2}.\d{3})', cue, re.MULTILINE)
            if timestamp_match:
                start_time = timestamp_match.group(1)
                end_time = timestamp_match.group(2)
                timestamp_ranges.append((start_time, end_time))

    if not timestamp_ranges:
        print(f"Keyword '{keyword}' not found in {vtt_file_path}.")
    else:
        for (start, end) in timestamp_ranges:
            buffer_seconds = 3  # Adjusted to directly add seconds
            start_seconds = timestamp_to_seconds(start)
            end_seconds = timestamp_to_seconds(end) + buffer_seconds
            output_file_name = f"{keyword}_{clip_id}.mp4"
            output_file_path = os.path.join(output_dir_base, output_file_name)
            ffmpeg_extract_subclip(video_file_path, start_seconds, end_seconds, targetname=output_file_path)
            print(f"Extracted clip {clip_id}: '{output_file_path}'")
            clip_id += 1  # Increment the clip ID for the next file

def process_videos_from_folder(video_dir, subtitle_dir, keyword, output_dir_base):
    start_clip_id = 0  # Initialize start clip ID
    for video_file in os.listdir(video_dir):
        if video_file.endswith(".mp4"):
            video_file_path = os.path.join(video_dir, video_file)
            video_base_name = os.path.splitext(video_file)[0]
            vtt_file_name = f"{video_base_name}.vtt"
            vtt_file_path = os.path.join(subtitle_dir, vtt_file_name)
            
            if os.path.exists(vtt_file_path):
                find_keyword_and_extract_clips(vtt_file_path, keyword, video_file_path, output_dir_base, start_clip_id)
                # Update start_clip_id to the next available ID after processing each video
                start_clip_id = clip_id

# Example usage
video_dir = r'D:\BSL_project\bsl_dataset\BOBSL\bobsl\videos'
subtitle_dir = r'D:\BSL_project\bsl_dataset\BOBSL\subtitles\subtitles\audio-aligned'
keyword = 'think'  # Example keyword to search for in subtitles
output_dir_base = r'D:\BSL_project\bsl_dataset\scripting_result\think'

# Ensure the output directory exists
if not os.path.exists(output_dir_base):
    os.makedirs(output_dir_base)

process_videos_from_folder(video_dir, subtitle_dir, keyword, output_dir_base)