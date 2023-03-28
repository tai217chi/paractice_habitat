""" this code contains python class for configuration of habitat simulator """

from pathlib import Path

class MatterportConfig:
    
    ## configuraton for observation image resolution ##
    # NOTE：低解像度時の品質劣化が激しい
    width = 1080
    height = 720
    
    ## specify dataset ##
    _dataset_path = Path(__file__).parents[1] / "data"
    scene = _dataset_path / "scene_datasets" / "mp3d" / "2azQ1b91cZZ" / "2azQ1b91cZZ.glb"
    scene_dataset_config = _dataset_path / "scene_datasets" / "mp3d.scene_dataset_config.json"
    
    ## configuration for observation kind ##
    rgb_sensor = True
    depth_sensor = True
    semantic_sensor = True 
    
    ## configuration for parameters related to embodied agent ##
    default_agent = 0
    sensor_height = 0.8
    
    ## other configuration ##
    seed = 1
    enable_physics = False
    
class HM3DConfig:
    
    dataset_kind = "val"
    
    ## configuration for observation images (RGB, instance mask, Depth) ##
    width = 1280
    height = 960
    
    ## specify dataset ##
    _dataset_path = Path(__file__).parents[1] / "data" 
    scene = _dataset_path / "scene_datasets" / "hm3d" / f"{dataset_kind}" /"00800-TEEsavR23oF" / "TEEsavR23oF.basis.glb"
    scene_dataset_config = _dataset_path / "scene_datasets" / "hm3d_annotated_val_basis.scene_dataset_config.json"
    
    ## configuration for observation kind ##
    rgb_sensor = True 
    depth_sensor = True 
    semantic_sensor = True 
    
    ## configuration for parameters related to embodied agent ##
    default_agent = 0 
    sensor_height = 1.0
    
    ## other configuration ##
    seed = 1
    enable_physics = False

class ReplicaConfig:
    
    ## configuration for observation images (RGB, instance mask, Depth) ##
    width = 640
    height = 480
    
    ## specify dataset ##
    _dataset_path = Path(__file__).parents[1] / "data" 
    scene = _dataset_path / "scene_datasets" / "replica_v1" / "apartment_1" / "habitat" /  "mesh_semantic.ply"
    scene_dataset_config = _dataset_path / "scene_datasets" / "replica_v1.scene_dataset_config.json"
    
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
    