""" this code contains python class for configuration of habitat simulator """

from pathlib import Path

class MatterportConfig:
    
    ## configuraton for observation image resolution ##
    width = 320
    height = 240
    
    ## specify dataset ##
    _dataset_path = Path(__file__).parents[1] / "data"
    scene = _dataset_path / "scene_datasets" / "mp3d" / "29hnd4uzFmX" / "29hnd4uzFmX.glb"
    mp3d_scene_dataset = _dataset_path / "scene_datasets" / "mp3d" / "mp3d.scene_dataset_config.json"
    
    ## configuration for observation kind ##
    rgb_sensor = True
    depth_sensor = True
    semantic_sensor = True 
    
    ## configuration for parameters related to embodied agent ##
    default_agent = 0
    sensor_height = 1.5
    
    ## other configuration ##
    seed = 1
    enable_physics = False
    
class GibbsonConfig:
    
    pass