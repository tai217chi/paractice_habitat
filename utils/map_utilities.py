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
    grid_size = 2.0 #! 2 [m] ごとに探索候補点を作成
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
def _draw_all_points_on_map(sim: habitat_sim.Simulator, search_points: np.ndarray) -> None:
    
    save_dir = Path(__file__).parent.resolve() / "map"
    
    ## 2次元の地図を生成 ##
    top_down_map = maps.get_topdown_map(
        sim,
        map_resolution=0.1,
        include_agent_map=True,
        include_collision_map=True,
        include_semantic_map=True,
    )
    
    ## 探索候補点を地図上に描画 ##
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off") 
    plt.imshow(top_down_map)
    if search_points is not None: 
        for point in search_points:
            plt.scatter(point[0], point[1], marker="o", s=10, alpha=0.8)
            
    plt.savefig(str(save_dir / "all_points"))