from temporal_flickering_ratio import *
import os 

# load video

video_path = "/shared/john/AR_dawood/AR/videos/260609_sf_from_basewan50_continue_4k/iter_8000/without_recache/"
video_paths = os.listdir(video_path)
full_video_paths = [os.path.join(video_path, video) for video in video_paths]

import ipdb;ipdb.set_trace()

all_results, video_results = compute_temporal_flickering_ratio_video_paths(full_video_paths)
temporal_flickering_results = {
	'overall': all_results,
	'total_videos': len(full_video_paths),
	'video_results': video_results
}

import json 
json.dump(temporal_flickering_results, open("sample_flickering_results.json", "w"), indent=4)
