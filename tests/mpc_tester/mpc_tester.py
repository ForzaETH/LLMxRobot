import os, time, ast, argparse
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from langchain_experimental.llm_bash.bash import BashProcess
from llm_mpc import RaceLLMMPC, MODEL_OPTIONS
from inference.inf_pipeline import CHAT_TEMPLATE_OPTIONS, RaceLLMPipeline
from train.utils.mpc.eval_cases import EVAL_DRIVING_CASES
import numpy as np
import matplotlib.pyplot as plt

TEST_OPTIONS = ['center', 'reverse', 'refvel', 'smooth']
    
class Tester:
    def __init__(self, openai_token, model_type, model_dir, quant, run_name, step, host_ip, ros=None):
        self.racechat: RaceLLMMPC = RaceLLMMPC(openai_token=openai_token, 
                                               model=model_type, 
                                               model_dir=model_dir,
                                               quant=quant,
                                               ros=ros,
                                               host_ip=host_ip,)
        
        self.model_name = model_type
        self.test_samples = 5000
        self.timeout = 120
        self.run_name = run_name
        self.current_step = step

        #create eval dir
        self.eval_dir = os.path.join('tests/mpc_tester/eval', run_name)
        os.makedirs(self.eval_dir, exist_ok=True)

        # Target speed for reverse test case
        self.rev_target_speed = -1.0 # [m/s]
        self.target_speed = 1.25 # [m/s]
        
        # bash settings
        self.bash = BashProcess(
            strip_newlines = False,
            return_err_output = False,
            persistent = False,
        )
        self.default_mpc_params = {
            "qv": 10.0,
            "qn": 20.0,
            "qalpha": 7.0,
            "qac": 0.01,
            "qddelta": 0.1,
            "alat_max": 10.0,
            "a_min": -10.0,
            "a_max": 10.0,
            "v_min": 1.0,
            "v_max": 12.0,
            "track_safety_margin": 0.3
        }

    def test(self, test_case: str, prompt: str, memory_nb: int = 5, ref_val: float = None):
        raceline_raw = self.racechat._echo_topic(topic="/global_waypoints", topic_type="f110_msgs/WpntArray", number=1)
        raceline = self.racechat._filter_raceline(raceline=raceline_raw)
        command = None
        llm_expl = None
        llm_inf_latency = None
        data = None
        rmse = 0.6969
        print('Reseting MPC params')
        self.reset_mpc_params()
        
        print(f"STARTING TEST CASE: {test_case}, Memory Number: {memory_nb}")
        start_time = time.time()
        command, llm_expl, mem_sources, _, _ = self.racechat.race_mpc_interact(scenario=prompt, memory_nb=memory_nb)
        llm_inf_latency = time.time() - start_time
        if test_case == "center":
            if command:
                self.racechat._reset_car()
                data_raw = self.racechat._echo_topic_over_one_lap(topic="/car_state/odom_frenet", topic_type='nav_msgs/Odometry',  timeout=self.timeout)
                data = self.racechat._filter_odom(odom=data_raw)
                rmse = self.center_rmse(data=data, raceline=raceline)
            else:
                # Fails to generate command, return high RMSE
                rmse = 69.0
        elif test_case == "raceline":
            if command:
                self.racechat._reset_car()
                data_raw = self.racechat._echo_topic_over_one_lap(topic="/car_state/odom_frenet", topic_type='nav_msgs/Odometry',  timeout=self.timeout)
                data = self.racechat._filter_odom(odom=data_raw)
                rmse = self.raceline_rmse(data=data)
            else:
                # Fails to generate command, return high RMSE
                rmse = 69.0
        elif test_case == "reverse":
            if command:
                self.racechat._reset_car()
                data_raw = self.racechat._echo_topic(topic="/car_state/odom_frenet", topic_type='nav_msgs/Odometry',  number= self.test_samples, timeout=self.timeout)
                data = self.racechat._filter_odom(odom=data_raw)
                rmse = self.reverse_rmse(data=data, target_speed=self.rev_target_speed)
            else:
                # Fails to generate command, return high RMSE
                rmse = 69.0
        elif test_case == "refvel":
            if command:
                self.racechat._reset_car()
                data_raw = self.racechat._echo_topic_over_one_lap(topic="/car_state/odom_frenet", topic_type='nav_msgs/Odometry', timeout=self.timeout)
                data = self.racechat._filter_odom(odom=data_raw)
                rmse = self.refvel_rmse(data=data, target_speed=ref_val)
            else:
                # Fails to generate command, return high RMSE
                rmse = 69.0
        elif test_case == "smooth":
            if command:
                self.racechat._reset_car()
                data_raw = self.racechat._echo_topic_over_one_lap(topic="/vesc/sensors/imu/raw", topic_type='sensor_msgs/Imu', timeout=self.timeout)
                data = self.racechat._filter_imu(imu=data_raw) # returns dict with keys: 'ax', 'ay'
                rmse = self.smooth_rmse(data=data)
            else:
                # Fails to generate command, return high RMSE
                rmse = 69.0
        else:
            raise ValueError("Invalid test case")

        return {'mem_nb': memory_nb, 'llm_cmd': command, 'llm_expl': llm_expl, 'llm_inf_latency': llm_inf_latency, 'rmse': rmse, 'case': test_case, 'mem_sources': mem_sources}
      
    def default_test(self, test_case: str, ref_val: float = None):
        raceline_raw = self.racechat._echo_topic(topic="/global_waypoints", topic_type="f110_msgs/WpntArray", number=1)
        raceline = self.racechat._filter_raceline(raceline=raceline_raw)
        data = None
        rmse = None
        # print('Reseting MPC params')
        self.reset_mpc_params()
        print(f"GETTING DEFAULTS FOR TEST CASE: {test_case}")
        if test_case == "center":
            self.racechat._reset_car()
            data_raw = self.racechat._echo_topic_over_one_lap(topic="/car_state/odom_frenet", topic_type='nav_msgs/Odometry',timeout=self.timeout)
            data = self.racechat._filter_odom(odom=data_raw)
            rmse = self.center_rmse(data=data, raceline=raceline)
        elif test_case == "raceline":
            self.racechat._reset_car()
            data_raw = self.racechat._echo_topic_over_one_lap(topic="/car_state/odom_frenet", topic_type='nav_msgs/Odometry', timeout=self.timeout)
            data = self.racechat._filter_odom(odom=data_raw)
            rmse = self.raceline_rmse(data=data)
        elif test_case == "reverse":
            self.racechat._reset_car()
            data_raw = self.racechat._echo_topic_over_one_lap(topic="/car_state/odom_frenet", topic_type='nav_msgs/Odometry', number= self.test_samples, timeout=self.timeout)
            data = self.racechat._filter_odom(odom=data_raw)
            rmse = self.reverse_rmse(data=data, target_speed=self.rev_target_speed)
        elif test_case == "refvel":
            self.racechat._reset_car()
            data_raw = self.racechat._echo_topic_over_one_lap(topic="/car_state/odom_frenet", topic_type='nav_msgs/Odometry', timeout=self.timeout)
            data = self.racechat._filter_odom(odom=data_raw)
            rmse = self.refvel_rmse(data=data, target_speed=ref_val)
        elif test_case == "smooth":
            self.racechat._reset_car()
            data_raw = self.racechat._echo_topic_over_one_lap(topic="/vesc/sensors/imu/raw", topic_type='sensor_msgs/Imu', timeout=self.timeout)
            data = self.racechat._filter_imu(imu=data_raw)
            rmse = self.smooth_rmse(data=data)
        else:
            raise ValueError("Invalid test case")
        
        return rmse
    
    def run_tests(self, num_tests: int, num_memories: int):
        all_reports = {}
        for test_case in EVAL_DRIVING_CASES.keys():
            all_reports[test_case] = {"subcases": {}}
            default_results = np.ones(len(EVAL_DRIVING_CASES[test_case]))
            per_case_rmses = np.zeros(len(EVAL_DRIVING_CASES[test_case]))
            if num_tests is not None:
                print(f"Running {num_tests} tests for {test_case} test case")
                default_results = self.default_test(test_case=test_case)
                result = self.run_single_test(test_case=test_case, num_tests=num_tests)
            else:
                for index, prompt in enumerate(EVAL_DRIVING_CASES[test_case]):
                    all_reports[test_case][str(index)] = {}
                    print(f"Running SINGLE {num_memories} memory test for prompt number {index+1} of the {test_case} test case")
                    if test_case == "refvel":
                        ref_val = prompt['ref_val']
                    else:
                        ref_val = None
                    default_results[index] = self.default_test(test_case=test_case, ref_val=ref_val)
                    per_case_result = self.test(test_case=test_case, prompt=prompt['human_prompt'], memory_nb=num_memories, ref_val=ref_val)
                    per_case_rmses[index] = per_case_result['rmse']
                    all_reports[test_case]["subcases"][str(index)] = {"result": per_case_result}
                    all_reports[test_case]["subcases"][str(index)]["default_rmse"] = default_results[index]
                    all_reports[test_case]["subcases"][str(index)]["logfile"] = self.generate_md_report(per_case_result, default_results[index], test_case, index)
            
            per_case_avg_rmse = np.mean(np.delete(per_case_rmses,np.where(per_case_rmses==69.0)))
            per_case_std_rmse = np.std(np.delete(per_case_rmses,np.where(per_case_rmses==69.0)))
            per_case_avg_default_rmse = np.mean(default_results)
            per_case_std_default_rmse = np.std(default_results)
            all_reports[test_case]["avg_rmse"] = per_case_avg_rmse
            all_reports[test_case]["std_rmse"] = per_case_std_rmse
            all_reports[test_case]["avg_default_rmse"] = per_case_avg_default_rmse
            all_reports[test_case]["std_default_rmse"] = per_case_std_default_rmse
            all_reports[test_case]["avg_rmse_improvement"] = (per_case_avg_default_rmse - per_case_avg_rmse) / per_case_avg_default_rmse
        return all_reports
    
    def run_single_test(self, test_case: str, num_tests: int):
        results = []
        for i in range(num_tests):
            result = self.test(test_case=test_case, memory_nb=i)
            results.append(result)
        return results

    # Plotting functions
    def plot_results(self, results, default_rmse=69.0):
        print("Plotting results:", results)
        # Dump the results to a file
        with open(os.path.join(self.eval_dir, 'results.txt'), 'w') as f:
            for result in results:
                f.write(str(result) + '\n')
        memory_nbs = [result['mem_nb'] for result in results]
        rmses = [result['rmse'] for result in results]
        latencies = [result['llm_inf_latency'] for result in results]
        param_adjustments = [result['llm_cmd'] for result in results]

        # Extract and normalize parameter adjustments
        param_adjustments_flat = []
        for adjustment in param_adjustments:
            if adjustment is not None:
                for param, value in adjustment.items():
                    default_value = self.default_mpc_params[param]
                    normalized_value = value / default_value if default_value != 0 else 0
                    param_adjustments_flat.append((param, normalized_value))

        if param_adjustments_flat:
            params, normalized_values = zip(*param_adjustments_flat)
        else:
            params, normalized_values = [], []

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

        # Bar plot for RMSE
        color = 'tab:red'
        ax1 = axs[0]
        ax1.set_xlabel('Number of Memories')
        ax1.set_ylabel('RMSE', color=color)
        ax1.grid(axis='y', linestyle='--', alpha=0.6)
        ax1.bar(memory_nbs, rmses, color=color, alpha=0.6, width=0.4, align='center')
        ax1.set_xticks(memory_nbs)  # Set x-axis ticks to discrete integer values
        ax1.axhline(y=default_rmse, color='r', linestyle='--', label='Default RMSE')

        # Histogram for parameter adjustments
        ax2 = axs[1]
        color = 'tab:blue'
        ax2.set_ylabel('Normalized Parameter Adjustments', color=color)
        ax2.grid(axis='y', linestyle='--', alpha=0.6)
        if params:
            unique_params = sorted(set(params))  # Get sorted unique parameters
            param_indices = [unique_params.index(param) for param in params]
            ax2.hist(param_indices, weights=normalized_values, bins=len(unique_params), edgecolor='black', alpha=0.6)
            ax2.set_xticks(range(len(unique_params)))
            ax2.set_xticklabels(unique_params, rotation=90)
        ax2.tick_params(axis='y', labelcolor=color)

        # Bar plot for latency
        ax3 = axs[2]
        color = 'tab:green'
        ax3.set_xlabel('Number of Memories')
        ax3.set_ylabel('LLM Inference Latency (s)', color=color)
        ax3.grid(axis='y', linestyle='--', alpha=0.6)
        ax3.bar(memory_nbs, latencies, color=color, alpha=0.3, width=0.4, align='center')
        ax3.set_xticks(memory_nbs)  # Set x-axis ticks to discrete integer values

        test_case = results[0]['case']
        fig.suptitle(f'{test_case} results')
        plot_path = os.path.join(self.eval_dir, f"{test_case}_evaluation_results.png")
        plt.savefig(plot_path)
        #plt.show()
        return plot_path

    # RMSE calculation & Utils
    def center_rmse(self, data: dict, raceline: dict):
        s_poses = data['s_pos']
        d_poses = data['d_pos']
        s_raceline = raceline['s']
        rmse = 0.0
        for i in range(len(s_poses)):
            raceline_idx = min(range(len(s_raceline)), key=lambda j: abs(s_raceline[j]-s_poses[i])) # find closest raceline index
            d_center = raceline['d_left'][raceline_idx] - (raceline['d_left'][raceline_idx] + raceline['d_right'][raceline_idx]) / 2 # center of the track
            rmse += (d_poses[i] - d_center)**2 # sum of squared errors
        if len(s_poses) == 0:
            return 69.69 # Evade zero division and give high error for not finishing the test
        rmse = (rmse / len(s_poses))**0.5 # root mean squared error
        return rmse

    def raceline_rmse(self, data: dict):
        s_poses = data['s_pos']
        d_poses = data['d_pos']
        rmse = 0.0
        for i in range(len(s_poses)):
            rmse += d_poses[i]**2
        if len(s_poses) == 0:
            return 69.69 # Evade zero division and give high error for not finishing the test
        rmse = (rmse / len(s_poses))**0.5
        return rmse

    def reverse_rmse(self, data: dict, target_speed: float):
        s_poses = data['s_pos']
        s_speeds = data['s_speed']
        rmse = 0.0
        for i in range(len(s_poses)):
            rmse += (s_speeds[i] - target_speed)**2
        if len(s_poses) == 0:
            return 69.69 # Evade zero division and give high error for not finishing the test
        rmse = (rmse / len(s_poses))**0.5
        return rmse
    
    def refvel_rmse(self, data: dict, target_speed: float):
        s_speeds = data['s_speed']
        rmse = 0.0
        for i in range(len(s_speeds)):
            rmse += (s_speeds[i] - target_speed)**2
        if len(s_speeds) == 0:
            return 69.69 # Evade zero division and give high error for not finishing the test
        rmse = (rmse / len(s_speeds))**0.5
        return rmse

    def smooth_rmse(self, data: dict):
        ax = data['ax']
        ay = data['ay']
        rmse = 0.0
        for i in range(len(ax)):
            rmse += ax[i]**2 + ay[i]**2
        if len(ax) == 0:
            return 69.69 # Evade zero division and give high error for not finishing the test
        rmse = (rmse / len(ax))**0.5
        return rmse

    def reset_mpc_params(self):        
        for param, value in self.default_mpc_params.items():
            self.racechat._set_ros_param(param, value, supress_print=True)
    
    # Utility functions
    def plot_from_txt(self, path: str):
        results = []
        with open(path, 'r') as f:
            for line in f:
                try:
                    results.append(ast.literal_eval(line.strip()))
                except (SyntaxError, ValueError) as e:
                    print(f"Skipping malformed line: {line.strip()}")
                    continue

        plot_path = self.plot_results(results)
        print("USING DUMMY DEFAULT RMSE VALUE FOR REPORTING PURPOSES")
        self.generate_md_report(results, plot_path, 69)

    def generate_md_report(self, results, default_rmse, test_case, index):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.eval_dir, f"step-{self.current_step}/{test_case}_{index}_evaluation_report_{timestamp}.md")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(f"#{self.model_name} {test_case.capitalize()} Evaluation Report\n\n")
            f.write(f"#### Default RMSE: {default_rmse:.3f}[m]\n")
            f.write("## Test Results\n")
            f.write(f"### Memory Number: {results['mem_nb']}\n")
            f.write(f"- **RMSE [m]**: {results['rmse']}\n")
            f.write(f"- **LLM Inference Latency [s]**: {results['llm_inf_latency']}\n")
            f.write(f"- **LLM Explanation**: {results['llm_expl']}\n")
            f.write(f"- **Generated Command**: {results['llm_cmd']}\n")
            f.write(f"- **Memory Sources**: {results['mem_sources']}\n\n")

        print(f"Markdown report generated at: {report_path}")
        return report_path


# Wrapper for running eval during RL-training
class TrainingTester(Tester):
    def __init__(self,
                 openai_token, 
                 custom_model,
                 shared_ros,  
                 run_name, 
                 step,
                 custom_tokenizer,
                 custom_chat_template,
                 host_ip,
                 ):
        super().__init__(openai_token=openai_token, 
                         model_type='training',
                         model_dir=None, 
                         quant=False,
                         run_name=run_name,
                         step=step,
                         host_ip=host_ip,
                         ros=shared_ros) 
        # Overwrite the LLM with the custom model
        training_llm = RaceLLMPipeline(chat_template=custom_chat_template, model=custom_model, tokenizer=custom_tokenizer, load_in_4bit=True)
        self.racechat.llm = training_llm
        self.racechat.custom = True
        self.racechat.use_openai = False
        print(f"Testing in TrainingTester using custom model: {custom_model} with tokenizer: {custom_tokenizer} and chat template: {custom_chat_template}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Tester with a specified model.')
    parser.add_argument('--model', choices=MODEL_OPTIONS, help='The model to use for the Tester.')    
    parser.add_argument('--model_dir', type=str, default=None, help='Path to the model directory (if using a custom model) can also point to HF.')
    parser.add_argument('--quant', action='store_true', help='Use quantization for the custom model, needs to be downloaded in models/ folder and dir towards it.')
    parser.add_argument('--chat_template', choices=CHAT_TEMPLATE_OPTIONS, default=CHAT_TEMPLATE_OPTIONS[-1], type=str, help='The chat template to use for the Tester.')
    parser.add_argument('--host_ip', type=str, default='192.168.192.107', help='ROS IP address for the SIM.')
    args = parser.parse_args()

    load_dotenv(find_dotenv())
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")
        
    run_name = os.path.basename(args.model_dir) if args.model_dir else args.model
    run_name = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Model options: 'gpt-4o', 'custom'
    tester = Tester(openai_token=OPENAI_API_TOKEN,
                    model_type=args.model,
                    model_dir=args.model_dir,
                    quant=args.quant,
                    run_name=run_name,
                    step=0,
                    host_ip=args.host_ip) 

    # Run tests and plot results
    tester.run_tests(num_tests=None, num_memories=5)
    
    # Example command: 
    # python3 -m tests.mpc_tester.mpc_tester --model custom --model_dir nibauman/RobotxLLM_Qwen7B_SFT --host_ip 192.168.192.75 --chat_template qwen-2.5
    
    