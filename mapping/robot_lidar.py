"""
Author: Adriel Martins

A simulation of a robot using LiDAR. You know the position of the robot.
The LiDAR will work as a sensor, and will give you the distance to the closest
obstacle in a given direction. We will only have 4 directions: 0, 90, 180, 270.
The robot in the simulation will follow a prespecified path.
"""

import random
import numpy as np
import pandas as pd

class Map:
    def __init__(self, labyrinth:np.array) -> None:
        self.labyrinth=labyrinth
    
    def generate_labyrinth(self, N:int):
        if N < 5:
            print("Labyrinth size should be at least 5x5.")
            return None

        labyrinth = [[1] * N for _ in range(N)]

        def is_valid(x, y):
            return x > 0 and x < N - 1 and y > 0 and y < N - 1

        def has_unvisited_neighbors(x, y):
            neighbors = [(x + 2, y), (x - 2, y), (x, y + 2), (x, y - 2)]
            for nx, ny in neighbors:
                if is_valid(nx, ny) and labyrinth[ny][nx] == 1:
                    return True
            return False

        def random_unvisited_neighbor(x, y):
            neighbors = [(x + 2, y), (x - 2, y), (x, y + 2), (x, y - 2)]
            random.shuffle(neighbors)
            for nx, ny in neighbors:
                if is_valid(nx, ny) and labyrinth[ny][nx] == 1:
                    return nx, ny
            return None

        x, y = random.randrange(1, N, 2), random.randrange(1, N, 2)
        labyrinth[y][x] = 0

        while has_unvisited_neighbors(x, y):
            nx, ny = random_unvisited_neighbor(x, y)
            dx = (nx - x) // 2
            dy = (ny - y) // 2
            labyrinth[y + dy][x + dx] = 0
            labyrinth[ny][nx] = 0
            x, y = nx, ny

        return self.labyrinth

    def print_labyrinth(self):
        for row in self.labyrinth.tolist():
            print(" ".join(["#" if cell == 1 else " " for cell in row]))

    
class LiDARobot:
    def __init__(self) -> None:
        self.lidar_directions = [0, 90, 180, 270]
        self.log = []

    def go_to(self, x: int, y: int, map:Map) -> None:
        if map.labyrinth[x,y] == 1:
            print(f"There is a wall on ({x},{y}).")
            raise Exception("There is a wall on the path.")
        self.x, self.y = x, y
    
    def sensor(self, map: Map) -> None:
        manhattan_distance_in_each_direction = []
        for lidar_direction in self.lidar_directions:
            if lidar_direction == 0:
                pseudo_vision_field = map.labyrinth[self.x, self.y+1:].tolist()
                manhattan_distance = pseudo_vision_field.index(1) + 1
                manhattan_distance_in_each_direction.append(manhattan_distance)
            elif lidar_direction == 90:
                pseudo_vision_field = map.labyrinth[:self.x:, self.y].tolist()
                # revert list to get the vision field in the right order
                pseudo_vision_field = pseudo_vision_field[::-1]
                manhattan_distance = pseudo_vision_field.index(1) + 1
                manhattan_distance_in_each_direction.append(manhattan_distance)
            elif lidar_direction == 180:
                pseudo_vision_field = map.labyrinth[self.x, :self.y].tolist()
                # revert list to get the vision field in the right order
                pseudo_vision_field = pseudo_vision_field[::-1]
                manhattan_distance = pseudo_vision_field.index(1) + 1
                manhattan_distance_in_each_direction.append(manhattan_distance)
            elif lidar_direction == 270:
                pseudo_vision_field = map.labyrinth[self.x+1:, self.y].tolist()
                manhattan_distance = pseudo_vision_field.index(1) + 1
                manhattan_distance_in_each_direction.append(manhattan_distance)
        
        self.log.append({"position": (self.x, self.y), "lidar": manhattan_distance_in_each_direction})
    
    def print_map(self) -> None:
        # construct point map of all the walls perceived by the robot
        wall_map = []
        position_map = []
        for log in self.log:
            x, y = log["position"]
            lidar = log["lidar"]
            wall_positions=[]
            for index_lidar_direction, lidar_manhantan_distance in enumerate(lidar):
                if index_lidar_direction == 0:
                    wall_positions.append((x, y+lidar_manhantan_distance))
                elif index_lidar_direction == 1: #90 angle
                    wall_positions.append((x-lidar_manhantan_distance, y))
                elif index_lidar_direction == 2: #180 angle
                    wall_positions.append((x, y-lidar_manhantan_distance))
                elif index_lidar_direction == 3: #270 angle
                    wall_positions.append((x+lidar_manhantan_distance, y))
            wall_map.extend(wall_positions)
            position_map.append((x,y))
        
        # print the wall map
        print("Wall map")
        N=10
        map_matrix = np.zeros((N,N))
        for (x,y) in wall_map:
            map_matrix[x,y] = 1
        for row in map_matrix.tolist():
            print(" ".join(["#" if cell == 1 else " " for cell in row]))

        # print the position map
        print("Position map")
        N=10
        map_matrix = np.zeros((N,N))
        for (x,y) in position_map:
            map_matrix[x,y] = 1
        for row in map_matrix.tolist():
            print(" ".join(["#" if cell == 1 else " " for cell in row]))

            

def main(path_to_follow: list, robot:LiDARobot, map:Map) -> None:
    for (x,y) in path_to_follow:
        print(f"Going to ({x},{y})")
        robot.go_to(x,y, map)
        print(f"Scanning...")
        robot.sensor(map)
    robot.print_map()

if __name__ == "__main__":
    print("Visualizing the map...")
    labyrinth = pd.read_csv("mapping/labyrinth.csv")
    map = Map(labyrinth=labyrinth.values)
    map.print_labyrinth()

    path_to_follow=[
        (1,1),
        (2,1),
        (3,1),
        (3,2),
        (3,3)
    ]
    robot = LiDARobot()

    main(path_to_follow, robot, map)

