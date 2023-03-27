import sys
from pathlib import Path

root_path = Path(__file__).parent.resolve()
sys.path.append(str(root_path))

from utils.visualizations import encode_video_from_folder

def main():
    observation_files = root_path.glob('observations/*')

    for observation_file in observation_files:
        rgb_dir = observation_file / 'rgb'
        depth_dir = observation_file / 'depth'
        semantic_dir = observation_file / 'semantic'
        
        rgb_save_dir = observation_file / 'rgb.mp4'
        depth_save_dir = observation_file / 'depth.mp4'
        semantic_save_dir = observation_file / 'semantic.mp4'
        
        
        encode_video_from_folder(str(rgb_dir), fps= 10, save_dir=rgb_save_dir)
        encode_video_from_folder(str(depth_dir), fps= 10, save_dir=depth_save_dir)
        encode_video_from_folder(str(semantic_dir), fps= 10, save_dir=semantic_save_dir)

if __name__ == '__main__':
    main()