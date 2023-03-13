import sys
from pathlib import Path 

import habitat_sim

root_dir = Path(__file__).parent.parent.resolve()
print(root_dir)
sys.path.append(str(root_dir))

from config.simulator_configure import MatterportConfig

def make_sim_config(matterport_config: MatterportConfig) -> habitat_sim.Configuration:
    
    """
    Method for create simulator environment
    
    Args: 
        matterport_config(__MatterportConfig__) : python class for setting simulator encironment
        
    Return:

    """
    
    settings = {
        "width": matterport_config.width, 
        "height": matterport_config.height, 
        "scene": matterport_config.scene, 
        "scene_dataset": matterport_config.mp3d_scene_dataset, 
        "default_agent": matterport_config.default_agent, 
        "sensor_height": matterport_config.sensor_height, 
        "color_sensor": matterport_config.rgb_sensor, 
        "depth_sensor": matterport_config.depth_sensor, 
        "semantic_sensor": matterport_config.semantic_sensor, 
        "seed": matterport_config.seed, 
        "enable_physics": False
    }
    
    ## simulator backend ##
    sim_cfg = habitat_sim.SimulatorConfiguration() 
    sim_cfg.gpu_device_id = 0 
    sim_cfg.scene_id = str(matterport_config.scene)
    sim_cfg.scene_dataset_config_file = str(matterport_config.mp3d_scene_dataset)
    sim_cfg.enable_physics = False
    
    # NOTE: all sensors must have the same resolution 
    # NOTE: refer https://aihabitat.org/docs/habitat-sim/habitat_sim.sensor.CameraSensorSpec.html about the detail of CameraSensorSpec
    sensor_specs = [] 
    
    if matterport_config.rgb_sensor:
        color_sensor_spec = habitat_sim.CameraSensorSpec() 
        color_sensor_spec.uuid = "color_sensor" 
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR 
        color_sensor_spec.resolution = [matterport_config.height, matterport_config.width]
        color_sensor_spec.position = [0.0, matterport_config.sensor_height, 0.0]
        color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_spec)
        
    if settings["depth_sensor"]:
        depth_sensor_spec = habitat_sim.CameraSensorSpec() 
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [matterport_config.height, matterport_config.width]
        depth_sensor_spec.position = [0.0, matterport_config.sensor_height, 0.0]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(depth_sensor_spec)
    
    if settings["semantic_sensor"]:
        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [matterport_config.height, matterport_config.width]
        semantic_sensor_spec.position = [0.0, matterport_config.sensor_height, 0.0]
        semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(semantic_sensor_spec)
    
    ## specify the amount of displacement in a forward action and the turn angle ##
    # NOTE: refer https://aihabitat.org/docs/habitat-sim/habitat_sim.agent.AgentConfiguration.html about the detail of AgentConfiguration class
    # TODO: Add the amount of agent displacement to the configuration file.
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