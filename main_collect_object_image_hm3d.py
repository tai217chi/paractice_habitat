""" This source code is used to test the object observation on habitat simulator """

import time
import sys
from pathlib import Path

import numpy as np

import habitat_sim
from habitat_sim.nav import ShortestPath, GreedyGeodesicFollower

parent_dir = Path(__file__).parent.resolve()
sys.path.append(str(parent_dir))

from config.simulator_configure import HM3DConfig
from utils.simulator_configuration import make_sim_config, recompute_navmesh
from utils.map_utilities import generate_search_points, draw_all_points_on_map
from utils.visualizations import encode_video_from_rgb_image, encode_video_from_all_kind_image
from utils.label_utilities import isntance_to_semantic, id_to_name
from scripts.navigation_to_goal_point import NavigateGoalPoint

def main():
    
    ## create simulator ##
    simulator_configuration = make_sim_config(HM3DConfig)
    sim = habitat_sim.Simulator(simulator_configuration)
    recompute_success = recompute_navmesh(sim, cell_size=0.03, cell_height=0.03)
    if not recompute_success :
        assert False, "Failed to recompute navmesh"
    
    dataset_id = str(HM3DConfig.scene.name).split('.')[0]
    
    ## generate search points ##
    points = generate_search_points(sim)
    
    ## generate agent ##
    agent = sim.initialize_agent(0)
    agent_state = habitat_sim.AgentState()
    agent.set_state(agent_state)
    
    instanceID_to_SemanticID = isntance_to_semantic(sim.semantic_scene) # インスタンスIDからセマンティックIDを取得するための辞書
    semanticID_to_name = id_to_name(sim.semantic_scene) # セマンティックIDからオブジェクト名を取得するための辞書
    
    ## 探索用のクラスをインスタンス化 ##
    observer = NavigateGoalPoint(instanceID_to_SemanticID, semanticID_to_name)
    
    draw_all_points_on_map(sim, points)
    
    ## full search ##
    
    ## 探索候補点のインデックスを定義 ##
    current_point = 0
    
    ## 物体を保存するディレクトリを保存する ##
    obj_save_dir = Path(__file__).parent.resolve() / "observations" / dataset_id /"objects"
    
    if not obj_save_dir.exists():
        obj_save_dir.mkdir(parents=True)
    
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
        sim = observer.navigation_with_object_obs(sim, closest_goal, obj_save_dir=obj_save_dir)
        
        current_point += 1
        
if __name__ == "__main__" :
    main()
        
        