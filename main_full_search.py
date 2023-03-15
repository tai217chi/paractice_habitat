import time
import sys
from pathlib import Path

import habitat_sim

parent_dir = Path(__file__).parent.resolve()
sys.path.append(str(parent_dir))

from config.simulator_configure import MatterportConfig
from utils.simulator_configuration import make_sim_config
from utils.map_utilities import generate_search_points, draw_all_points_on_map
from utils.visualizations import encode_video_from_rgb_image
from scripts.navigation_to_goal_point import NavigateGoalPoint

def main():
    
    ## create simulator ##
    simulator_configuration = make_sim_config(MatterportConfig)
    sim = habitat_sim.Simulator(simulator_configuration)
    
    ## generate search points ##
    points = generate_search_points(sim)
    
    ## generate agent ##
    agent = sim.initialize_agent(0)
    agent_state = habitat_sim.AgentState()
    agent.set_state(agent_state)
    
    draw_all_points_on_map(sim, points)
    
    ## full search ##
    
    ## 探索候補点のインデックスを定義 ##
    current_point = 0
    
    while current_point < len(points) :
    
        observations = NavigateGoalPoint.navigation(sim, points[current_point])
        
        current_point += 1
        time.sleep(2.0)
        
        encode_video_from_rgb_image(observations, current_point)
        
if __name__ == "__main__" :
    main()
        
        