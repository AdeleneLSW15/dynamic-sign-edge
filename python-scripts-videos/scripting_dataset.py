import re
import os

def append_keyword_timestamps_to_file(vtt_file_path, keyword, output_file_path, video_id):
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
                timestamp_range_formatted = f"{start_time} --> {end_time}"
                timestamp_ranges.append(timestamp_range_formatted)

    with open(output_file_path, 'a', encoding='utf-8') as file:  # 'a' mode for appending to the file
        file.write(f"Video ID: {video_id}\n")
        if not timestamp_ranges:
            file.write("Keyword not found\n\n")
        else:
            for timestamp_range in timestamp_ranges:
                file.write(f"{timestamp_range}\n")
            file.write("\n")  # Add extra newline for spacing between sections

def process_videos_and_subtitles(video_dir, subtitle_dir, keyword, output_file_path):
    for video_file in os.listdir(video_dir):
        if video_file.endswith(".mp4"):
            video_base_name = os.path.splitext(video_file)[0]
            vtt_file_name = f"{video_base_name}.vtt"
            vtt_file_path = os.path.join(subtitle_dir, vtt_file_name)

            if os.path.exists(vtt_file_path):
                append_keyword_timestamps_to_file(vtt_file_path, keyword, output_file_path, video_base_name)
            else:
                print(f"Subtitle file {vtt_file_name} not found for video {video_file}.")

# Example usage
video_dir = r'D:\BSL_project\bsl_dataset\BOBSL\bobsl\videos'
subtitle_dir = r'D:\BSL_project\bsl_dataset\BOBSL\subtitles\subtitles\audio-aligned'
keyword = 'how are you'
output_file_path = r'D:\BSL_project\bsl_dataset\BOBSL\manual-script\timestamps_how_are_you_all_videos.txt'

# Clear the output file before appending new content
if os.path.exists(output_file_path):
    os.remove(output_file_path)

process_videos_and_subtitles(video_dir, subtitle_dir, keyword, output_file_path)
