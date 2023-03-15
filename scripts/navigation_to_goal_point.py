
import numpy as np

import habitat_sim
from habitat_sim.nav import GreedyGeodesicFollower, ShortestPath

class NavigateGoalPoint :
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def navigation(sim:habitat_sim.Simulator, search_point: np.ndarray) -> list:
        
        ## 経路計画に使用するクラスをインスタンス化 ##
        path_finder = ShortestPath()
        
        ## 経路追従を行うクラスをインスタンス化 ##
        follower = GreedyGeodesicFollower(pathfinder=sim.pathfinder, agent=sim.get_agent(0), goal_radius=0.75, forward_key="move_forward", left_key="turn_left", right_key="turn_right")
        observations = [] 
        follower.reset() 
        
        ## エージェントの現在地を取得 ##
        current_pose = sim.get_agent(0).get_state().position
        
        ## navmeshを使用して、探索候補点までナビゲーション可能か調べる ##
        path_finder.requested_start = current_pose
        path_finder.requested_end = search_point
        found_path = sim.pathfinder.find_path(path_finder)
        
        if found_path:
            try:
                action_list = follower.find_path(search_point)
            except habitat_sim.errors.GreedyFollowerError:
                action_list = [None]
                
            while True: 
                next_action = action_list[0]
                action_list = action_list[1:]
                
                if next_action is None or len(action_list) == 0:
                    break 
                
                sim.step(next_action)
                
                observations.append(sim.get_sensor_observations()["color_sensor"])
                    
        return observations