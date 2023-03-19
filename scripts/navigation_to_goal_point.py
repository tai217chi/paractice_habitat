import time

import cv2
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
                
                observations.append(sim.get_sensor_observations())
                    
        return observations
    
    @classmethod 
    def navigation_with_object_obs(cls, sim: habitat_sim.Simulator, search_point: np.ndarray, obj_save_dir: str) -> None :
        
        """
        環境内を移動し物体の観測情報を集めるための関数
        
        """        
        
        ## 経路計画に使用するクラスをインスタンス化 ##
        path_finder = ShortestPath()
        
        ## 経路追従を行うクラスをインスタンス化 ##
        follower = GreedyGeodesicFollower(pathfinder=sim.pathfinder, agent=sim.get_agent(0), goal_radius=0.75, forward_key="move_forward", left_key="turn_left", right_key="turn_right")
        follower.reset() 
        
        observations = []
        
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
                
            current_ation = 0
            while True: 
                next_action = action_list[0]
                action_list = action_list[1:]
                
                if next_action is None or len(action_list) == 0:
                    break 
                
                sim.step(next_action)
                
                observation = sim.get_sensor_observations()
                bbox_image = cls._object_obs(observation["color_sensor"], observation["semantic_sensor"], obj_save_dir + f"{current_ation}_")
                
                observations.append(bbox_image)
                current_ation += 1
                
        return observations
                

    @classmethod
    def _object_obs(cls, rgb_view: np.ndarray, semantic_view: np.ndarray, save_dir: str) -> np.ndarray :
        
        """
        instance maskに従いカラー画像から物体の画像を取得するための関数
        
        """
        
        bboxes = cls._mask_to_bbox(semantic_view)
        
        ## デバッグ用。BBoxをrgb画像に可視化 ##
        bbox_image = cls._bbox_plot(rgb_view, bboxes)
        
        for index, bbox in enumerate(bboxes) :
            
            ## 切り出す領域が画像サイズよりも大きい場合、黒色で補間する ##
            if bbox[2] > rgb_view.shape[1]  or bbox[3] > rgb_view.shape[0] :
                if bbox[2] > rgb_view.shape[1] :
                    rgb_view_copy = cv2.copyMakeBorder(rgb_view, 0, 0, 0, bbox[2] - rgb_view.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))

                if bbox[3] > rgb_view.shape[0] :
                    rgb_view_copy = cv2.copyMakeBorder(rgb_view, 0, bbox[3] - rgb_view.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                
                ## RoIを切り出す ##
                crop_image = cv2.cvtColor(rgb_view_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]], cv2.COLOR_BGR2RGB)
            
                ## 切り出した領域を保存する ##
                cv2.imwrite(save_dir + f"{index}.png", crop_image)
                
            else:
                if len(rgb_view.copy()[bbox[1]:bbox[3], bbox[0]:bbox[2]]) != 0:
                    crop_image = cv2.cvtColor(rgb_view.copy()[bbox[1]:bbox[3], bbox[0]:bbox[2]], cv2.COLOR_BGR2RGB)
                    cv2.imwrite(save_dir + f"{index}.png", crop_image)
            
            
        return bbox_image
        

    @staticmethod
    def _mask_to_bbox(instance_mask: np.ndarray) -> np.ndarray :
        
        """
        instance maskをBounding Boxに変換するための関数
        
        """
        
        object_index = np.unique(instance_mask)
        
        bboxes  = []
        for i in object_index :
            y, x = np.where(instance_mask == i)
            bbox = [np.min(x), np.min(y), np.max(x), np.max(y)]
            
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            ## バウンディングボックスが正方形になるように固定 ##
            if width > height:
                bbox[3] = bbox[1] + width
            else:
                bbox[2] = bbox[0] + height
                
            bboxes.append(bbox)
            
        return np.array(bboxes)
                

    @staticmethod
    def _bbox_plot(rgb_view: np.ndarray, bboxes: np.ndarray) -> np.ndarray :
        
        """
        Bounding Boxをrgb画像に可視化するための関数
        
        """
        plot_image = rgb_view.copy()
        for bbox in bboxes :
            plot_image = cv2.rectangle(plot_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
        return plot_image