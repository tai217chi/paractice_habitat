
import time
import sys
import queue
from pathlib import Path

import numpy as np

import habitat_sim

parent_dir = Path(__file__).parent.resolve()
sys.path.append(str(parent_dir))

from config.simulator_configure import HM3DConfig
from utils.simulator_configuration import make_sim_config
from utils.map_utilities import generate_search_points, draw_all_points_on_map
from utils.visualizations import encode_video_from_rgb_image
from scripts.navigation_to_goal_point import NavigateGoalPoint

def main():
    
    ## create simulator ##
    simulator_configuration = make_sim_config(HM3DConfig)
    sim = habitat_sim.Simulator(simulator_configuration)
    dataset_id = str(HM3DConfig.scene.name).split('.')[0]
    
    ## generate search points ##
    points = generate_search_points(sim)
    
    ## generate agent ##
    agent = sim.initialize_agent(0)
    agent_state = habitat_sim.AgentState()
    agent.set_state(agent_state)
    
    draw_all_points_on_map(sim, points, dataset_id=dataset_id)
    
    ## full search ##
    
    ## 探索候補点のインデックスを定義 ##
    current_point = 0
    visited = set()
    while current_point < len(points) :
        # 最も近いゴール地点を決定
        closest_goal = None
        min_distance = float("inf")
        current_agent_position = sim.agents[0].get_state().position
        for goal in points:
            dist = np.linalg.norm(goal - current_agent_position)
            if dist < min_distance and tuple(goal.tolist()) not in visited:
                closest_goal = goal
                min_distance = dist

        visited.add(tuple(closest_goal.tolist()))
        sim = NavigateGoalPoint.navigation(sim, closest_goal, is_save_obs=True, 
                                     current_point=current_point, dataset_id=dataset_id)
        
        current_point += 1
        time.sleep(2.0)
        
if __name__ == "__main__" :
    main()
        
        