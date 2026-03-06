#!/usr/bin/env python3
import json

# Load the JSON file
with open('/Users/hivamoh/cs234Proj/CS234-Project-RL/video_rankings3.json', 'r') as f:
    data = json.load(f)

# Count the number of groups
num_groups = len(data)

# Count the total number of videos across all groups
total_videos = 0
for group_name, group_data in data.items():
    if 'rankings' in group_data:
        total_videos += len(group_data['rankings'])

print(f"Number of groups: {num_groups}")
print(f"Total number of videos: {total_videos}")
print(f"\nBreakdown by group:")
for group_name, group_data in data.items():
    if 'rankings' in group_data:
        num_videos = len(group_data['rankings'])
        print(f"  {group_name}: {num_videos} videos")
