import sys
from pathlib import Path 

import habitat_sim

root_dir = Path(__file__).parent.parent.resolve()
print(root_dir)
sys.path.append(str(root_dir))

from config.simulator_configure import MatterportConfig

def make_sim_config(config_class: MatterportConfig) -> habitat_sim.Configuration:
    
    """
    Method for create simulator environment
    
    Args: 
        config_class(__MatterportConfig__) : python class for setting simulator encironment
        
    Return:

    """
    
    settings = {
        "width": config_class.width, 
        "height": config_class.height, 
        "scene": config_class.scene, 
        "scene_dataset": config_class.scene_dataset_config, 
        "default_agent": config_class.default_agent, 
        "sensor_height": config_class.sensor_height, 
        "color_sensor": config_class.rgb_sensor, 
        "depth_sensor": config_class.depth_sensor, 
        "semantic_sensor": config_class.semantic_sensor, 
        "seed": config_class.seed, 
        "enable_physics": False
    }
    
    ## simulator backend ##
    sim_cfg = habitat_sim.SimulatorConfiguration() 
    sim_cfg.gpu_device_id = 0 
    sim_cfg.scene_id = str(config_class.scene)
    sim_cfg.scene_dataset_config_file = str(config_class.scene_dataset_config)
    sim_cfg.enable_physics = False
    sim_cfg.use_semantic_textures = True
    
    # NOTE: all sensors must have the same resolution 
    # NOTE: refer https://aihabitat.org/docs/habitat-sim/habitat_sim.sensor.CameraSensorSpec.html about the detail of CameraSensorSpec
    sensor_specs = [] 
    
    if config_class.rgb_sensor:
        color_sensor_spec = habitat_sim.CameraSensorSpec() 
        color_sensor_spec.uuid = "color_sensor" 
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR 
        color_sensor_spec.resolution = [config_class.height, config_class.width]
        color_sensor_spec.hfov = 74
        color_sensor_spec.position = [0.0, config_class.sensor_height, 0.0]
        color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_spec)
        
    if settings["depth_sensor"]:
        depth_sensor_spec = habitat_sim.CameraSensorSpec() 
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [config_class.height, config_class.width]
        depth_sensor_spec.hfov = 74
        depth_sensor_spec.position = [0.0, config_class.sensor_height, 0.0]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(depth_sensor_spec)
    
    if settings["semantic_sensor"]:
        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [config_class.height, config_class.width]
        semantic_sensor_spec.hfov = 74
        semantic_sensor_spec.position = [0.0, config_class.sensor_height, 0.0]
        semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(semantic_sensor_spec)
    
    ## specify the amount of displacement in a forward action and the turn angle ##
    # NOTE: refer https://aihabitat.org/docs/habitat-sim/habitat_sim.agent.AgentConfiguration.html about the detail of AgentConfiguration class
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.05)), 
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)), # amount [degree]
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)), # amount [degree]
    }
    agent_cfg.radius = 0.1 
    agent_cfg.height = 1.0 
    
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def recompute_navmesh(sim: habitat_sim.Simulator, cell_size: float=0.05, cell_height: float=0.2) -> bool:
    """
    ボクセルの解像度を変更するために、NavMeshを再計算する

    Args:
        sim (habitat_sim.Simulator): _description_
        cell_size (float, optional): _description_. Defaults to 0.05.
        cell_height (float, optional): _description_. Defaults to 0.2.

    Returns:
        bool: _description_
    """    
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.cell_size = cell_size
    navmesh_settings.cell_height = cell_height
    
    recompute_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings, include_static_objects=True)
    
    return recompute_success 
    