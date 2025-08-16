import numpy as np

EVAL_CASES =  [
            {"human_prompt":"Don't move", "evaluation_function": "stop_car_check"},
            {"human_prompt":"Reverse the car", "evaluation_function": "reverse_car_check"},
            {"human_prompt":"Drive forward", "evaluation_function": "forward_check"},
            {"human_prompt":"Oscillate!", "evaluation_function": "oscillation_check"},
            {"human_prompt":"Drive close to the left wall", "evaluation_function": "drive_close_to_left_wall_check"},
            {"human_prompt":"Drive on the centerline", "evaluation_function": "drive_centerline_check"},
            {"human_prompt":"Drive faster than 3 m/s", "evaluation_function": "speed_check"},
            {"human_prompt":"Drive on the racing line", "evaluation_function": "drive_racing_line_check"},
        ]

def stop_car_check(robot_states):
    evaluation_list = []
    for i in robot_states:
        if np.abs(np.mean(i["s_speed"])) < 0.1 and np.abs(np.mean(i["d_speed"])) < 0.1:
            evaluation_list.append(True)
        else:
            evaluation_list.append(False)
    return evaluation_list

def reverse_car_check(robot_states):
    evaluation_list = []
    for i in robot_states:
        if np.mean(i["s_speed"]) < 0 or i["reversing"] == True:
            evaluation_list.append(True)
        else:
            evaluation_list.append(False)
    return evaluation_list

def forward_check(robot_states):
    evaluation_list = []
    for i in robot_states:
        if np.mean(i["s_speed"]) > 0.1:
            evaluation_list.append(True)
        else:
            evaluation_list.append(False)
    return evaluation_list

def oscillation_check(robot_states):
    evaluation_list = []
    for i in robot_states:
        sign_changes = 0
        for j in range(1, len(i["d_pos"])):
            if i["d_pos"][j] * i["d_pos"][j-1]  < 0:  # Sign change occurs
                sign_changes += 1
        if sign_changes > 1 and np.abs(max(i["d_pos"])) >= 0.3:
            evaluation_list.append(True)
        else:
            evaluation_list.append(False)
    return evaluation_list

def drive_close_to_left_wall_check(robot_states):
    evaluation_list = []
    for i in robot_states:
        if np.mean(i["d_left"]) < 0.4 and np.mean(i["s_speed"]) > 0.1:
            evaluation_list.append(True)
        else:
            evaluation_list.append(False)
    return evaluation_list

def drive_centerline_check(robot_states):
    evaluation_list = []
    for i in robot_states: 
        on_centerline = True
        for j in range(i["data_samples"]):
            if np.abs(i["d_left"][j] - i["d_right"][j]) > 0.3 and np.mean(i["s_speed"]) > 0.1:
                on_centerline = False
        evaluation_list.append(on_centerline)
    return evaluation_list

def speed_check(robot_states, threshold=3):
    evaluation_list = []
    for i in robot_states:
        if np.mean(i["s_speed"]) > threshold:
            evaluation_list.append(True)
        else:
            evaluation_list.append(False)
    return evaluation_list

def drive_racing_line_check(robot_states):
    evaluation_list = []
    for i in robot_states:
        if np.abs(np.mean(i["d_pos"])) <= 0.3 and np.mean(i["s_speed"]) > 0.1:    
            evaluation_list.append(True)
        else:
            evaluation_list.append(False)
    return evaluation_list