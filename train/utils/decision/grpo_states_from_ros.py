import os, json, time
import argparse
import numpy as np
from dotenv import load_dotenv, find_dotenv
from llm_mpc import RaceLLMMPC

load_dotenv(find_dotenv())
OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def get_data(race_llm, filename, target_hz=10, data_time=60.0):
    while True:
        #Init data if necessary
        if race_llm.raceline is None or race_llm.odom_hz is None or race_llm.reasoning_hints is None:
            raceline_raw = race_llm._echo_topic(topic="/global_waypoints", topic_type='f110_msgs/WpntArray', number=1)
            race_llm.raceline = race_llm._filter_raceline(raceline=raceline_raw)
            race_llm.odom_hz = race_llm._get_topic_hz(topic="/car_state/odom_frenet", topic_type='nav_msgs/Odometry')
            race_llm.reasoning_hints = race_llm.load_reasoning_hints()
            time.sleep(0.5)
        else:
            break

    # Sample data for data_time [s] duration and downsample it to data_samples
    echo_nb = int(data_time * race_llm.odom_hz)
    data_raw = race_llm._echo_topic(topic="/car_state/odom_frenet", topic_type='nav_msgs/Odometry', number=echo_nb, timeout=data_time)
    odom_data = race_llm._filter_odom(odom=data_raw)
    data_samples = target_hz * data_time
    odom_data = {
        key: value[::max(1, int(np.ceil(len(value)/data_samples)))] 
        for key, value in odom_data.items()
    }
    d_left, d_right = race_llm._dist_to_boundaries(data=odom_data, raceline=race_llm.raceline)
    
    state = {"s_pos": odom_data["s_pos"],
            "d_pos": odom_data["d_pos"],
            "s_speed": odom_data["s_speed"],
            "d_speed": odom_data["d_speed"],
            "d_left": d_left, 
            "d_right": d_right} 
    
    # dump the state as json to file
    with open(filename, 'x') as f:  # 'x' mode ensures file does not exist
        json.dump(state, f, indent=4)
    
    print(f"Data saved to file: {filename}")               


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='State the filename to save the data to.')
    parser.add_argument('--filename', type=str, help='The filename to save the data to. Has to be a json.')
    parser.add_argument('--hostip', type=str, default='192.168.192.107', help='The host IP for the ROS connection.')
    parser.add_argument('--target_hz', type=int, default=10, help='The target frequency to downsample the data to.')
    parser.add_argument('--data_time', type=float, default=60.0, help='The duration of the data collection in seconds.')
    args = parser.parse_args()
    
    assert args.filename is not None, "Please provide a filename to save the data to."
    assert args.filename.endswith('.json'), "Please provide a filename ending with '.json'."
    
    race_llm = RaceLLMMPC(hf_token=HF_API_TOKEN, openai_token=OPENAI_API_TOKEN, model="gpt-4o", host_ip=args.hostip)
    full_filename = os.path.join('train/dataset/grpo_decision/raw_states', args.filename)
    if os.path.exists(full_filename):
        raise FileExistsError(f"File '{full_filename}' already exists. Choose a different name.")
    print(f"Saving data to file: {full_filename}")
    get_data(race_llm=race_llm, filename=full_filename, target_hz=args.target_hz, data_time=args.data_time)
