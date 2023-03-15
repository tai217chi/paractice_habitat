from pathlib import Path

import numpy as np

import habitat_sim
from habitat.utils.visualizations import maps

import matplotlib.pyplot as plt

#===============================================================================
# 探索候補点を地図上に生成する関数
#===============================================================================
def generate_search_points(sim: habitat_sim.Simulator) -> np.ndarray:
        
    bounds = sim.pathfinder.get_bounds() 
    grid_size = 2.0 
    grid_x = np.arange(bounds[0][0], bounds[1][0], grid_size)
    grid_y = np.arange(bounds[0][2], bounds[1][2], grid_size)
    xx, yy = np.meshgrid(grid_x, grid_y)
    points = np.column_stack((xx.ravel(), np.zeros_like(xx).ravel(), yy.ravel()))
    
    ## 占有されている点を除外 ##
    for i in range(len(points)) :
        is_occupied = sim.pathfinder.is_navigable(points[i])
        if not is_occupied:
            points[i] = np.nan 
            
    points = points[~np.isnan(points).any(axis=1)] # NaNを除外する
    
    return points


#===============================================================================
# 軌跡を描画した地図を保存する関数
#===============================================================================
def _save_robot_trajectory(top_down_map, key_points=None, num_search = 0) -> None:
    
    save_dir = Path(__file__).parent.resolve() / "map"
    
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(top_down_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.savefig(str(save_dir / f"/map_{num_search}"))
    
#===============================================================================
# すべての探索候補点を2次元の地図上に描画する関数
#===============================================================================
def draw_all_points_on_map(sim: habitat_sim.Simulator, search_points: np.ndarray, meters_per_pixel: float=0.02) -> None:
    
    save_dir = Path(__file__).parent.parent.resolve() / "map"
    
    height = sim.pathfinder.get_bounds()[0][1] # 地図の高さを取得
    
    ## 2次元の地図を生成 ##
    top_down_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
    print(top_down_map.shape)
    
    grid_dimension = (top_down_map.shape[0], top_down_map.shape[1])
    
    ## 探索候補点の座標系を変換 ##
    points_map_index = _convert_world_coordinate_to_map_index(sim.pathfinder, search_points, meters_per_pixel)
    
    ## 探索候補点を地図上に描画 ##
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off") 
    plt.imshow(top_down_map)
    if points_map_index is not None: 
        for point_map_index in points_map_index:
            plt.scatter(point_map_index[0], point_map_index[1], marker="o", s=30, c="black", alpha=0.8)
            
    plt.savefig(str(save_dir / "all_points"))
    
#===============================================================================
# 世界座標系の探索候補点を地図のインデックスに変換する関数
#===============================================================================
def _convert_world_coordinate_to_map_index(pathfinder, points_world_coordinate, meters_per_pixel) -> list:
    
    points_map_index = [] 
    bounds = pathfinder.get_bounds()
    for point_world_coordinate in points_world_coordinate:
        # convert world coordinate x, z to mapindex x, y
        px = (point_world_coordinate[0] - bounds[0][0]) / meters_per_pixel
        py = (point_world_coordinate[2] - bounds[0][2]) / meters_per_pixel
        points_map_index.append(np.array([px, py]))
        
    return points_map_index