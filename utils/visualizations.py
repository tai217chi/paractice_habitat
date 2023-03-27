from pathlib import Path

from habitat.utils.visualizations.utils import images_to_video
from habitat_sim.utils import viz_utils as vut

#====================================================================================================
# encoding video from rgb image observations
#====================================================================================================
def encode_video_from_rgb_image(rgb_obserbations: list, search_num: int, scene_id: str="") -> None:
    """_description_
    カラー画像のリストを受け取り、mp4ファイルに変換する。

    Args:
        rgb_obserbations (list): _description_
        search_num (int): _description_
    """
    if len(scene_id) == 0:
        dir_name = Path(__file__).parent.parent.resolve() / "observations" / "rgb"
        
    else: 
        dir_name = Path(__file__).parent.parent.resolve() / "observations" / scene_id / "rgb"
    
    if not dir_name.exists():
        dir_name.mkdir()
        
    images_to_video(rgb_obserbations, str(dir_name), f"continuous_nav_{search_num}")
    
#====================================================================================================
# encoding video from semantic mask observations
#====================================================================================================
def encode_video_from_semantic_image(semantic_observations: list, search_num: int, scene_id: str="") -> None:
    """_description_
    セマンティックセグメンテーションの画像のリストを受け取り、mp4ファイルに変換する。

    Args:
        semantic_obserbations (list): _description_
        search_num (int): _description_
    """
    
    if len(scene_id) == 0:
        dir_name = Path(__file__).parent.parent.resolve() / "observations" / "semantic"
        
    else: 
        dir_name = Path(__file__).parent.parent.resolve() / "observations" / scene_id / "semantic"
    
    
    if not dir_name.exists():
        dir_name.mkdir() 
        
    images_to_video(semantic_observations, str(dir_name), f"continuous_nav_{search_num}")
    
#====================================================================================================
# encoding video from depth image observations
#====================================================================================================
def encode_video_from_depth_image(depth_observations: list, search_num: int, scene_id: str="") -> None:
    """_description_
    深度画像のリストを受け取り、mp4ファイルに変換する。

    Args:
        depth_observations (list): _description_
        search_num (int): _description_
    """
    
    if len(scene_id) == 0:
        dir_name = Path(__file__).parent.parent.resolve() / "observations" / "depth"
        
    else: 
        dir_name = Path(__file__).parent.parent.resolve() / "observations" / scene_id / "depth"
    
    if not dir_name.exists():
        dir_name.mkdir() 
        
    images_to_video(depth_observations, str(dir_name), f"continuous_nav_{search_num}")
    
#====================================================================================================
# encoding video from all types image observations
#====================================================================================================
def encode_video_from_all_kind_image(observations: list, search_num: int, scene_id: str="", visualize_rgb: bool=True, visualize_semantic: bool=True, visualize_depth: bool=True):
    
    """_description_
    全種類の画像のリストを受け取り、mp4ファイルに変換する。

    Args:
        obserbations (list): _description_
        search_num (int): _description_
    """
    
    output_dir = Path(__file__).parent.parent.resolve() / "observations"
    
    if len(scene_id) != 0:
        output_dir = Path(__file__).parent.parent.resolve() / "observations" / scene_id
        
    if not output_dir.exists():
        output_dir.mkdir()
        
    rgb_dir = output_dir / "rgb"
    semantic_dir = output_dir / "semantic"
    depth_dir = output_dir / "depth"
    
    if not rgb_dir.exists():
        rgb_dir.mkdir()
        
    if not depth_dir.exists():
        depth_dir.mkdir()
        
    if not semantic_dir.exists():
        semantic_dir.mkdir()
    
    if visualize_rgb:
        vut.make_video(
                observations=observations,
                primary_obs="color_sensor",
                primary_obs_type="color",
                video_file=str(rgb_dir / f"continuous_nav_{search_num}"),
                fps=10,
                open_vid=False,
            )
    
    if visualize_semantic:
        vut.make_video(
            observations=observations,
            primary_obs="semantic_sensor",
            primary_obs_type="semantic",
            video_file=str(semantic_dir / f"continuous_nav_{search_num}"),
            fps=10,
            open_vid=False,
        )
        
    if visualize_depth:
        vut.make_video(observations=observations, 
                        primary_obs="depth_sensor", 
                        primary_obs_type="depth", 
                        video_file=str(depth_dir / f"continuouts_nav_{search_num}"), 
                        fps=10, 
                        open_vid=False
        )
    