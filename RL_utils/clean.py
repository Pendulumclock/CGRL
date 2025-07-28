import json
import os

raw_path = '/mnt/vepfs/Djinhan/FlightGPT/RL_utils/citynav_train_seen.json'
output_path = '/mnt/vepfs/Djinhan/FlightGPT/RL_utils/citynav_rl_data.json'

with open(raw_path, 'r') as f:
    data = json.load(f)

result = []
episode_id = ['', 0, 0]

imagesize_dir = {}
for step_data in data:
    if step_data['episode_id'][0] == episode_id[0] and step_data['episode_id'][1] == episode_id[1]:
        continue
    elif step_data['landmark_bbox'] == []:
        continue
    elif len(step_data['landmark_bbox']) > 2:
        continue
    
    episode_id = step_data['episode_id']
    

    cur_image_size = str(step_data['image_size'])
    
    if cur_image_size in imagesize_dir:
        imagesize_dir[cur_image_size] += 1
    else:
        imagesize_dir[cur_image_size] = 1
    
    if cur_image_size == '[4001, 4001]' and imagesize_dir[cur_image_size] >= 1400:
        continue
    

    
    result.append(step_data)

with open(output_path, 'w') as f:
    json.dump(result, f, indent=4)