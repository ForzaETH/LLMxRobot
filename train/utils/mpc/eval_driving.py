import numpy as np
from llm_mpc import RaceLLMMPC

DRIVING_CASES =  [
    #MAP: circle (params taken from git hash: 0563a6426a856689ced171a12623a6993f0af8e8)
    # Baseline params are taken from train on circle
    {"human_prompt":"Drive on the centerline", "evaluation_function": "center_rmse", "baseline_rmse": 1.23},
    {"human_prompt":"Drive on the racing line", "evaluation_function": "raceline_rmse", "baseline_rmse": 0.11},
    {"human_prompt":"Reverse the car", "evaluation_function": "reverse_rmse", "baseline_rmse": 4.93},
    {"human_prompt":"Drive at 1.83 m/s as closely as possible", "evaluation_function": "refvel_rmse", "baseline_rmse": 2.1},
    {"human_prompt":"Drive smoothly", "evaluation_function": "smooth_rmse", "baseline_rmse": 0.38},
]

# TODO: this way it is just sampling 1k samples, but in every part of the track can be different which is sucky
SAMPLE_NUMBER = 1000

# RMSE calculations
def center_rmse(race_llm: RaceLLMMPC):
    # Get data
    data_raw = race_llm._echo_topic(topic="/car_state/odom_frenet", topic_type='nav_msgs/Odometry', number=SAMPLE_NUMBER)
    data = race_llm._filter_odom(odom=data_raw)
    raceline = race_llm.raceline
    
    s_poses = data['s_pos']
    d_poses = data['d_pos']
    s_raceline = raceline['s']
    # print(f"data len: {len(s_poses)}, raceline len: {len(s_raceline)}")
    rmse = 0.0
    for i in range(len(s_poses)):
        raceline_idx = min(range(len(s_raceline)), key=lambda j: abs(s_raceline[j]-s_poses[i])) # find closest raceline index
        d_center = raceline['d_left'][raceline_idx] - (raceline['d_left'][raceline_idx] + raceline['d_right'][raceline_idx]) / 2 # center of the track
        rmse += (d_poses[i] - d_center)**2 # sum of squared errors
    rmse = (rmse / len(s_poses))**0.5 # root mean squared error
    return rmse

def raceline_rmse(race_llm: RaceLLMMPC):
    # Get data
    data_raw = race_llm._echo_topic(topic="/car_state/odom_frenet", topic_type='nav_msgs/Odometry', number=SAMPLE_NUMBER)
    data = race_llm._filter_odom(odom=data_raw)
    
    s_poses = data['s_pos']
    d_poses = data['d_pos']
    # print(f"data len: {len(s_poses)}, d_pos len: {len(d_poses)}")
    rmse = 0.0
    for i in range(len(s_poses)):
        rmse += d_poses[i]**2
    rmse = (rmse / len(s_poses))**0.5
    return rmse

def reverse_rmse(race_llm: RaceLLMMPC, target_speed: float=-1.0):
    # Get data
    data_raw = race_llm._echo_topic(topic="/car_state/odom_frenet", topic_type='nav_msgs/Odometry', number=SAMPLE_NUMBER)
    data = race_llm._filter_odom(odom=data_raw)
    
    s_poses = data['s_pos']
    s_speeds = data['s_speed']
    # print(f"data len: {len(s_poses)}, s speed len: {len(s_speeds)}")
    rmse = 0.0
    for i in range(len(s_poses)):
        rmse += (s_speeds[i] - target_speed)**2
    rmse = (rmse / len(s_poses))**0.5
    return rmse

def refvel_rmse(race_llm: RaceLLMMPC, target_speed: float=1.83):
    print(f"Refval RMSE against target speed: {target_speed}")
    # Get data
    data_raw = race_llm._echo_topic(topic="/car_state/odom_frenet", topic_type='nav_msgs/Odometry', number=SAMPLE_NUMBER)
    data = race_llm._filter_odom(odom=data_raw)
    
    s_speeds = data['s_speed']
    # print(f"s speeds len: {len(s_speeds)}")
    rmse = 0.0
    for i in range(len(s_speeds)):
        rmse += (s_speeds[i] - target_speed)**2
    rmse = (rmse / len(s_speeds))**0.5
    return rmse

def smooth_rmse(race_llm: RaceLLMMPC):
    # Get data
    data_raw = race_llm._echo_topic(topic="/vesc/sensors/imu/raw", topic_type='sensor_msgs/Imu', number=SAMPLE_NUMBER)
    data = race_llm._filter_imu(imu=data_raw)
    
    ax = data['ax']
    ay = data['ay']
    # print(f"ax len: {len(ax)}, ay len: {len(ay)}")
    rmse = 0.0
    for i in range(len(ax)):
        rmse += ax[i]**2 + ay[i]**2
    rmse = (rmse / len(ax))**0.5
    return rmse