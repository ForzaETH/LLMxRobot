import json
import os

with open('centerline_car.json', 'r') as file:
    robot_states = json.load(file)

for i in range(len(robot_states)):
    print(robot_states[i])
    print("\n")