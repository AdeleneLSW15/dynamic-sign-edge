import re
import os


def find_keyword_in_vtt(file_path, keyword, output_file_path):
    timestamp_ranges = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    cues = re.split(r'\n\n+', content)
    
    for cue in cues:
        if re.search(keyword, cue, re.IGNORECASE):
            timestamp_match = re.search(r'^(\d{2}:\d{2}:\d{2}.\d{3}) --> (\d{2}:\d{2}:\d{2}.\d{3})', cue, re.MULTILINE)
            if timestamp_match:
                start_time = timestamp_match.group(1)
                end_time = timestamp_match.group(2)
                timestamp_range_formatted = f"timestamp_from = {start_time}\ntimestamp_till = {end_time}\n"
                timestamp_ranges.append(timestamp_range_formatted)

    # Check if any keyword matches were found
    if not timestamp_ranges:
        # Keyword not found, write message to output file
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write('keyword not found')
        print(f"Keyword '{keyword}' not found in {file_path}.")
    else:
        # Write the timestamp ranges to the output file
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for timestamp_range in timestamp_ranges:
                file.write(timestamp_range + '\n')  # Additional newline for separation between entries
        print(f"Timestamp ranges where '{keyword}' is found have been written to {output_file_path}")

# Example usage
vtt_file_path = r'D:\BSL_project\bsl_dataset\BOBSL\subtitles\subtitles\audio-aligned\5085344787448740525.vtt'
keyword = 'sorry'
output_file_path = r'D:\BSL_project\bsl_dataset\BOBSL\manual-script\timestamps_sorry_5085344787448740525_alternate.txt'
find_keyword_in_vtt(vtt_file_path, keyword, output_file_path)