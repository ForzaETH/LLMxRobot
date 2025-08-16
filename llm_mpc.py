from dotenv import load_dotenv, find_dotenv
from typing import List
import numpy as np
import os, time, ast, re, argparse, math
from scipy.spatial.transform import Rotation as R
import roslibpy
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

MODEL_OPTIONS = ['gpt-4o', 'custom', 'training']

class RaceLLMMPC():
    def __init__(self, 
                 openai_token, 
                 model,
                 model_dir=None,
                 quant=False,
                 ros=None,
                 no_ROS=False,  
                 host_ip='192.168.192.105'):
        
        # MPC PARAMS
        self.DEFAULT_MPC_PARAMS = {
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
        
        # ROS stuff
        if no_ROS:
            #We only want the access members of the class
            self.ros = None
            pass
        else:
            # ROS hook
            if ros:
                self.ros = ros
            else:
                self.ros = roslibpy.Ros(host=host_ip, port=9090)
                self.ros.run()
            self.mpc_param_namespace = 'mpc_param_tuner'
            if not self.ros.is_connected:
                raise ValueError("ROS connection failed. Please check the ROS master URI.")
            #Check that the llm_mpc_controller is active and get the current mpc params
            proceed_bool, diagnosis = self._check_llm_mpc_node()
            while not proceed_bool:
                print(diagnosis)
                time.sleep(1)
                proceed_bool, diagnosis = self._check_llm_mpc_node()

            # Dynamic Reconfigure client
            self.dyn_client = roslibpy.Service(self.ros, f'/{self.mpc_param_namespace}/set_parameters', 'dynamic_reconfigure/Reconfigure')
            self.raceline, self.odom_hz = self.init_race_data()
            
        # LLM stuff
        self.openai_token = openai_token
        self.quant = quant
        self.llm, self.custom, self.use_openai = self.init_llm(model=model, model_dir=model_dir, openai_token=openai_token)

        # MPC RAG
        self.base_memory, self.vector_index = self.load_memory(openai_token=self.openai_token)
        # Decision RAG
        self.decision_index = self.load_decision_mem(openai_api_key=self.openai_token)

    def load_memory(self, openai_token) -> tuple[str, VectorstoreIndexCreator]:
        # Base Memory
        base_mem_dir = os.path.join('./', 'prompts/mpc_base_memory.txt')
        print(f'Loading base memory from {base_mem_dir}...')
        with open(base_mem_dir, 'r') as f:
            base_mem = f.read()

        loaders = []
        # Get Memories for the RAG
        memories_dir = os.path.join('./', 'prompts/mpc_memory.txt')
        print(f'Loading memories from {memories_dir}...')
        memories_loader = TextLoader(memories_dir)
        loaders.append(memories_loader)

        # Create a VectorstoreIndex from the collected loaders
        splitter = CharacterTextSplitter(separator='#', keep_separator=False, chunk_overlap=20, chunk_size=100)
        index = VectorstoreIndexCreator(embedding=OpenAIEmbeddings(api_key=openai_token), text_splitter=splitter).from_loaders(loaders)
        return base_mem, index

    def load_decision_mem(self, openai_api_key) -> VectorstoreIndexCreator:
        # Get Memories for the RAG
        memories_dir = 'prompts/RAG_memory.txt'
        print(f'Loading Decision RAG from {memories_dir}...')
        memories_loader = TextLoader(file_path=memories_dir)

        # Create a VectorstoreIndex from the collected loaders
        splitter = CharacterTextSplitter(separator='#', keep_separator=False, chunk_overlap=20, chunk_size=100)
        index = VectorstoreIndexCreator(embedding=OpenAIEmbeddings(api_key=openai_api_key), text_splitter=splitter).from_loaders([memories_loader])
        return index
    
    def race_mpc_interact(self, scenario, memory_nb : int=0, prompt_only: bool=False) -> str:
        '''
        Tunes Dynamic Reconfigure parameters of the MPC controller based on the scenario
        '''
        # Query the index and pass the result to the command chain for processing
        RAG_query = f"""
        Task: {scenario}\n
        """
        #Perform RAG manually
        start_time = time.time()
        #Retrieve docs from the RAG
        rag_sources: List[Document] = self.vector_index.vectorstore.search(query=RAG_query, search_type='similarity', k=memory_nb) if memory_nb > 0 else []
        rag_sources = [{'meta': doc.metadata, 'content': doc.page_content} for doc in rag_sources]
        LLM_query = f"""
        You are an AI assistant helping to tune the parameters of an MPC controller for an autonomous racing car. Below is the context and the task:

        ## Context
        1. **Scenario**: {scenario}
        2. **Base Memory and Cost Formulation**: {self.base_memory}

        ## Task
        Adapt the tuneable parameters of the MPC so that the car achieves the following: **{scenario}**.

        ## Caution
        - Do not invent new parameters! Strictly adhere to the parameters provided in the base memory.

        ## Constraints
        - Use the min and max ranges of the parameters provided in the base memory.
        - Only consider relevant RAG information and quote it briefly in your explanation.

        ## RAG Information
        - !!! Not all memories are relevant to the task. Select the most relevant ones. !!!
        - **Memory Entries**:
        {rag_sources}

        ## Expected Output Format
        Always strictly return your answers in the following format (no other dicts in your response and no comments in the params part!):
        new_mpc_params = {{
            'param1': new_value1,
            'param2': new_value2,
            ...
        }}
        """

        # Returns only prompt, is used to create distillation data
        if prompt_only:
            return LLM_query
        else:
            #Invoke the LLM
            llm = self.llm
            if self.custom:
                llm_out, input_tokens, output_tokens = llm(LLM_query)
            else:
                llm_out = llm.invoke(LLM_query)
            llm_out = llm_out.content if self.use_openai else llm_out
            extracted_command, llm_expl = self._sanitize_tune_output(llm_out)

            # Set the new MPC parameters
            extracted_command = extracted_command if extracted_command is not None else {}
            try:
                for p, v in self.DEFAULT_MPC_PARAMS.items():
                    if p in extracted_command.keys():
                        self._set_ros_param(p, extracted_command[p])
                    else:
                        self._set_ros_param(p, v)
            except Exception as e:
                print(f"Failed to get valid command from LLM: {llm_out}, {str(e)}")
                extracted_command = None

            return extracted_command, llm_expl, rag_sources, LLM_query, llm_out

    def race_reasoning(self, human_prompt: str, data_time: float=2.0, data_samples: int=5, prompt_only: bool=False, k: int = 5) -> str:
        # Sample data for data_time [s] duration and downsample it to data_samples
        echo_nb = int(data_time * self.odom_hz)
        data_raw = self._echo_topic(topic="/car_state/odom_frenet", topic_type='nav_msgs/Odometry', number=echo_nb, timeout=2.0)
        odom_data = self._filter_odom(odom=data_raw)
        odom_data = {
            key: value[::max(1, int(np.ceil(len(value)/data_samples)))] 
            for key, value in odom_data.items()
        }
        d_left, d_right = self._dist_to_boundaries(data=odom_data, raceline=self.raceline)
        reversing_bool = np.mean(odom_data['s_speed']) < -0.1
        crashed_bool, facing_wall = self._crash_detection(d_left=d_left, d_right=d_right, odom=odom_data, raceline=self.raceline)

        rag_sources = self.decision_index.vectorstore.search(query=human_prompt, search_type='similarity', k=k) if k > 0 else []
        rag_sources = [{'meta': doc.metadata, 'content': doc.page_content} for doc in rag_sources]
        hints = ''
        for hint in rag_sources:
            hints += hint['content'] + "\n"

        prompt = f"""
        You are an AI embodied on an autonomous racing car. The human wants to: {human_prompt} \n
        The car is currently driving on a track, data available is in the Frenet Corrdinate frame, units are in meters and meters per second. 
        The racing line is a minimal curvature trajectory which is not the centerline.
        The data has been sampled for {data_time} seconds in {data_samples} samples.
        - The car's position along the racing line is given by the s-coordinate: {odom_data['s_pos']}\n
        - The car's lateral deviation from the racing line is given by the d-coordinate: {odom_data['d_pos']}\n
        - The car's speed along the racing line is given by the s-speed: {odom_data['s_speed']}\n
        - The car's speed perpendicular to the racing line is given by the d-speed: {odom_data['d_speed']}\n
        - The distance to the left wall is: {d_left}\n
        - The distance to the right wall is: {d_right}\n
        - Bool if the car is reversing: {reversing_bool}\n
        - Bool if the car has crashed: {crashed_bool}\n

        Here are some hints to help you reason about the car's current state: {hints}

        If the car has crashed, then you should instruct it to reverse! \n

        Check if the car is doing what the human wants. Choose one of the following actions to command the car: \n
        - "Continue behavior" \n
        - "Change behavior": <Brief Instruction>
        Output template: \n
        "Action": <Action> \n
        """
        # Returns only prompt, is used to create distillation data
        if prompt_only:
            return prompt
        
        else:
            #Invoke the LLM
            llm = self.llm
            if self.custom:
                llm_out, input_tokens, output_tokens = llm(prompt)
            else:
                llm_out = llm.invoke(prompt)
            llm_out = llm_out.content if self.use_openai else llm_out

            state_str = f"""d_coordinate: {odom_data['d_pos']}, s_speed: {odom_data['s_speed']}, d_speed: {odom_data['d_speed']}, d_left: {d_left}, d_right: {d_right}, crashed: {crashed_bool}, facing_wall: {facing_wall}"""
            return llm_out, state_str

    def _sanitize_tune_output(self, text: str):
        #print("Full text:", text)

        # Initialize variables to hold the dictionary and the explanation
        command_dict = None
        explanation_text = text

        # Find the start of the dictionary using '{' which marks the beginning of the parameter settings
        try:
            dict_start_idxs = [m.start() for m in re.finditer(r'new_mpc_params\s*=\s*{', text)]
            dict_end_idxs = [m.end() for m in re.finditer(r'}', text)]

            # Ensure start and end indices are properly paired
            for start_idx in dict_start_idxs:
                for end_idx in dict_end_idxs:
                    if end_idx > start_idx:
                        dict_text = text[start_idx:end_idx]
                        try:
                            # Extract the dictionary text and safely evaluate it
                            dict_str = dict_text.split('=', 1)[1].strip()
                            command_dict = ast.literal_eval(dict_str)
                            # Check if the command_dict contains at least one key-value pair
                            if isinstance(command_dict, dict) and len(command_dict) > 0:
                                # print("Valid dictionary found:", command_dict)
                                return command_dict, explanation_text
                            else:
                                print("Invalid dictionary format.")
                                return None, explanation_text
                        except (ValueError, SyntaxError):
                            continue
                else:
                    # Continue to the next start index if no valid end index was found
                    continue
                # Break if a valid dictionary was found and parsed
                break
        except Exception as e:
            print(f"Dictionary or explanation format error within text: {text}, {type(text)}")

        return command_dict, explanation_text

    def init_llm(self, model: str, model_dir:str, openai_token: str) -> tuple:
        use_openai = False
        custom = False
        llm = None
        if model not in MODEL_OPTIONS:
            raise ValueError(f"Model {model} not supported. Please use one of {MODEL_OPTIONS}")
        if model == 'gpt-4o':
            # Use OpenAI LLM
            use_openai = True
            llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_token)
        elif model == 'custom':
            # Use a custom model that runs locally so you need CUDA!
            if self.quant:
                from inference.inf_gguf import RaceLLMGGGUF
                # Find gguf file in the model_dir
                gguf_name = [f for f in os.listdir(model_dir) if f.endswith('.gguf')][0]
                print(f"Using gguf file: {gguf_name}")
                llm = RaceLLMGGGUF(model_dir=model_dir, gguf_name=gguf_name, max_tokens=256)
            else:
                from inference.inf_pipeline import RaceLLMPipeline
                llm = RaceLLMPipeline(model_dir=model_dir, chat_template='qwen-2.5', load_in_4bit=True)
            custom = True
        elif model == 'training':
            # passing for RobotxR1 style training
            print("Not setting a model because we are training and using race_llm just as a vessel to interact with ROS and utils.")
            pass
        else:
            raise ValueError(f"Something went wrong with the model selection: {model}")
        return llm, custom, use_openai
        
    ################ROS UTILS################
    def init_mpc_dynconf(self, mpc_param_namespace:str):
        mpc_param_namespace = 'mpc_param_tuner'
        if not self.ros.is_connected:
            raise ValueError("ROS connection failed. Please check the ROS master URI.")
        #Check that the llm_mpc_controller is active and get the current mpc params
        proceed_bool, diagnosis = self._check_llm_mpc_node(mpc_param_namespace=mpc_param_namespace)
        while not proceed_bool:
            print(diagnosis)
            time.sleep(1)
            proceed_bool, diagnosis = self._check_llm_mpc_node(mpc_param_namespace=mpc_param_namespace)

        # Dynamic Reconfigure client
        dyn_client = roslibpy.Service(self.ros, f'/{mpc_param_namespace}/set_parameters', 'dynamic_reconfigure/Reconfigure')
        return dyn_client
    
    def init_race_data(self):
        #Init data
        raceline_raw = self._echo_topic(topic="/global_waypoints", topic_type='f110_msgs/WpntArray', number=1)
        raceline = self._filter_raceline(raceline=raceline_raw)
        odom_hz = self._get_topic_hz(topic="/car_state/odom_frenet", topic_type='nav_msgs/Odometry')
        return raceline, odom_hz
    
    def _check_llm_mpc_node(self, mpc_param_namespace: str = 'mpc_param_tuner') -> tuple[bool, str]:
        try:
            ros_nodes = self._get_ros_nodes()
            llm_node_active = False
            for node in ros_nodes:
                if mpc_param_namespace in node:
                    llm_node_active = True
                    break
            if not llm_node_active:
                llm_mpc_controller_not_active_str = f"The {mpc_param_namespace} Node is not active. Please start the llm_mpc_controller first."
                print(llm_mpc_controller_not_active_str)
                return False, llm_mpc_controller_not_active_str
            #Get the mpc params from dynamic reconfigure
            current_mpc_params = self.get_current_mpc_params(mpc_param_namespace=mpc_param_namespace)
            return True, current_mpc_params
        except Exception as e:
            ros_not_active_str = "Failed to get ROS nodes or failed to get current mpc params:"
            print(ros_not_active_str, str(e))
            return False, ros_not_active_str

    def get_current_mpc_params(self, mpc_param_namespace: str) -> dict:
        try:
            current_params = {}
            for key in self.DEFAULT_MPC_PARAMS.keys():
                param_value = self.ros.get_param(f'/{mpc_param_namespace}/{key}')
                current_params[key] = param_value
            return current_params
        except Exception as e:
            print("Failed to get current mpc params:", str(e))
            return None

    def _get_ros_nodes(self):
        try:
            nodes_service = roslibpy.Service(self.ros, '/rosapi/nodes', 'rosapi/Nodes')
            request = roslibpy.ServiceRequest()
            result = nodes_service.call(request)
            return result['nodes']
        except Exception as e:
            print("Failed to get ROS nodes:", str(e))
            return []

    def _set_ros_param(self, param, value, supress_print=False):
        try:
            # Only set the parameter if it is in the list of MPC parameters
            if param not in self.DEFAULT_MPC_PARAMS.keys():
                # print(f"Parameter {param} not in the list of MPC parameters. Skipping...")
                return
            else:
                params = {
                    'config': {
                        'doubles': [{'name': param, 'value': value}]
                    }
                }
                request = roslibpy.ServiceRequest(params)
                self.dyn_client.call(request, self._on_param_set)
                # if not supress_print:
                    # print(f"Set {param} to {value}")
        except Exception as e:
            print("Failed to set ROS parameter:", str(e))

    def _on_param_set(self, response):
        #print(f"Parameter set response: {response}")
        pass
    
    def _echo_topic_over_one_lap(self, topic: str, topic_type: str, number: int = 1000, timeout: float = 60.0) -> List[str]:
        data = []
        lap_data = []
        lap_listener = roslibpy.Topic(self.ros,'/lap_data','f110_msgs/LapData')
        listener = roslibpy.Topic(self.ros, topic, topic_type)
        
        def lap_callback(message):
            lap_data.append(message['lap_count'])
            print("Lap data received")
            if len(lap_data) > 1:
                lap_listener.unsubscribe()
    
        def callback(message):
            if len(lap_data) == 1:
                data.append(message)
            if len(lap_data) > 1:
                listener.unsubscribe()

        listener.subscribe(callback)
        lap_listener.subscribe(lap_callback)
        start_time = time.time()
        while len(lap_data) < 2 and time.time() - start_time < timeout:
            time.sleep(0.1)

        try:
            listener.unsubscribe()
        except KeyError:
            pass
        try:
            lap_listener.unsubscribe()
        except KeyError:
            pass
        return data

    def _echo_topic(self, topic: str, topic_type: str, number: int = 1, timeout: float = 5.0) -> List[str]:
        data = []
        listener = roslibpy.Topic(self.ros, topic, topic_type)

        def callback(message):
            data.append(message)
            if len(data) >= number:
                listener.unsubscribe()

        listener.subscribe(callback)
        start_time = time.time()
        while len(data) < number and time.time() - start_time < timeout:
            time.sleep(0.1)
        try:
            listener.unsubscribe()
        except KeyError:
            pass
        # print("+++++++++++++++++++++++++++++")
        # print(f"Received {len(data)} messages from topic {topic}")
        # print("+++++++++++++++++++++++++++++")
        return data
    
    def _filter_ackermann(self, ackermann: List[str]) -> dict:
        steering_angles = []
        speeds = []
        for i, item in enumerate(ackermann):
            steering_angle = round(float(item['drive']['steering_angle']), 3)
            speeds_value = round(float(item['drive']['speed']), 3)
            steering_angles.append(steering_angle)
            speeds.append(speeds_value)
        return {'steering_angle': steering_angles, 'speed': speeds}
    
    def _filter_odom(self, odom: List[str], frenet: bool = True) -> dict:
        x_poses = []
        y_poses = []
        x_speeds = []
        y_speeds = []
        thetas = []
        for i, item in enumerate(odom):
            x_pos = round(float(item['pose']['pose']['position']['x']), 3)
            y_pos = round(float(item['pose']['pose']['position']['y']), 3)
            x_poses.append(x_pos)
            y_poses.append(y_pos)

            x = float(item['pose']['pose']['orientation']['x'])
            y = float(item['pose']['pose']['orientation']['y'])
            z = float(item['pose']['pose']['orientation']['z'])
            w = float(item['pose']['pose']['orientation']['w'])
            r = R.from_quat([x, y, z, w])
            thetas.append(r.as_euler('zyx', degrees=False)[0])
        
            x_speed = round(float(item['twist']['twist']['linear']['x']), 3)
            y_speed = round(float(item['twist']['twist']['linear']['y']), 3)
            x_speeds.append(x_speed)
            y_speeds.append(y_speed)

        if frenet:
            return {'s_pos': x_poses, 'd_pos': y_poses, 's_speed': x_speeds, 'd_speed': y_speeds, 'theta': thetas}
        else:
            return {'x_pos': x_poses, 'y_pos': y_poses, 'x_speeds': x_speeds, 'y_speeds': y_speeds, 'thetas': thetas}

    def _filter_imu(self, imu: List[str]) -> dict:
        axs = []
        ays = []
        for i, item in enumerate(imu):
            ax = round(float(item['linear_acceleration']['x']), 3)
            ay = round(float(item['linear_acceleration']['y']), 3)
            axs.append(ax)
            ays.append(ay)
        return {'ax': axs, 'ay': ays}

    def _filter_raceline(self, raceline: List[str]) -> dict:
        ss = []
        xs = []
        ys = []
        d_lefts = []
        d_rights = []
        psis = []
        raceline = raceline[0]
        for _, item in enumerate(raceline['wpnts']):
            x = float(item['x_m'])
            xs.append(x)

            y = float(item['y_m'])
            ys.append(y)

            s = float(item['s_m'])
            ss.append(s)

            d_left = float(item['d_left'])
            d_lefts.append(d_left)

            d_right = float(item['d_right'])
            d_rights.append(d_right)

            psi = float(item['psi_rad'])
            psis.append(psi)

        return {'s': ss, 'x': xs, 'y': ys, 'd_left': d_lefts, 'd_right': d_rights, 'psi': psis}
    
    def _dist_to_boundaries(self, data: dict, raceline: dict) -> List[float]:
        s_poses = data['s_pos']
        d_poses = data['d_pos']
        s_raceline = raceline['s']
        d_left = []
        d_right = []
        for i in range(len(s_poses)):
            raceline_idx = min(range(len(s_raceline)), key=lambda j: abs(s_raceline[j]-s_poses[i])) # find closest raceline index
            d_l = round(abs(raceline['d_left'][raceline_idx] - d_poses[i]), 3)
            d_r = round(abs(raceline['d_right'][raceline_idx] + d_poses[i]), 3)
            d_left.append(d_l)
            d_right.append(d_r)
        return d_left, d_right
    
    def _get_topic_hz(self, topic: str, topic_type: str, timeout: float=2.0) -> float:
        # Get the rate of a topic
        data_raw = self._echo_topic(topic=topic, topic_type=topic_type, number=200, timeout=timeout)
        stamps = []
        for _, item in enumerate(data_raw):
            secs = int(item['header']['stamp']['secs'])
            nsecs = int(item['header']['stamp']['nsecs'])
            stamps.append(secs + nsecs/1e9)
        hz = 1.0 / np.mean(np.diff(stamps)) 
        print(f"Topic {topic} has a rate of {hz} Hz")
        return hz
    
    def _crash_detection(self, d_left: List[float], d_right: List[float], odom: dict, raceline: dict):
        too_close = []
        for i in range(len(d_left)):
            if d_left[i] < 0.38 or d_right[i] < 0.38:
                too_close.append(True)
            else:
                too_close.append(False)
        too_close = sum(too_close) / len(too_close) > 0.5
        
        car_s = odom['s_pos'][-1]
        idx = min(range(len(raceline['s'])), key=lambda i: abs(raceline['s'][i]-car_s))
        raceline_heading = raceline['psi'][idx]
        car_heading = odom['theta'][-1]

        wall_heading = raceline_heading + np.pi/2 if odom['d_pos'][-1] > 0 else raceline_heading - np.pi/2

        # scalar product of the car's heading and the raceline heading to check if the car is facing the wall
        dot_prod = np.dot([np.cos(car_heading), np.sin(car_heading)], [np.cos(wall_heading), np.sin(wall_heading)])
        facing_wall = -5*np.pi/12 < np.arccos(dot_prod) < 5*np.pi/12
        
        crashed = True if too_close and facing_wall else False

        if crashed:
            print("Car has crashed!")

        return crashed, facing_wall
    
    def _crash_detection_via_sim(self):
        data = self._echo_topic(topic="/wall_collision", topic_type='std_msgs/Bool', number=10)
        crashed = True if any([d['data'] for d in data]) else False
        return crashed
    
    def _mpc_crash_detection(self, echo_nb: int=200, timeout: float=2.0):
        # Check if MPC has crapped its pants
        drive_data_raw = self._echo_topic(topic="/drive", topic_type='ackermann_msgs/AckermannDriveStamped', number=echo_nb, timeout=timeout)
        drive_data = self._filter_ackermann(drive_data_raw)
        no_steer = True if len(drive_data_raw) == 0 else False
        
        if no_steer:
            print("No steering data received! MPC is not running.")
            return True
        
        # Check if steer and speed is the same
        steer_std = np.std(drive_data['steering_angle'])
        speed_std = np.std(drive_data['speed']) 
        same_drive = True if (steer_std < 1e-4 and speed_std < 1e-4) else False
        
        if same_drive:
            print(f"MPC is not running! Steering and speed are the same. Steering std: {steer_std}, Speed std: {speed_std}")
            return True
        else:
            return False
    
    def _reset_car(self):
        # Reset the car to the starting position
        initialpose_topic = roslibpy.Topic(self.ros, '/initialpose', 'geometry_msgs/PoseWithCovarianceStamped')
        
        # Get first point on the racing line
        x = self.raceline['x'][0]
        y = self.raceline['y'][0]
        psi = self.raceline['psi'][0]
        # Convert yaw (psi) to quaternion
        qz = math.sin(psi / 2.0)
        qw = math.cos(psi / 2.0)
        
        # Create a reset message (you may need to adjust these values based on your actual starting position)
        reset_msg = roslibpy.Message({
            'header': {
                'stamp': {'secs': 0, 'nsecs': 0},
                'frame_id': 'map'
            },
            'pose': {
                'pose': {
                    'position': {'x': x, 'y': y, 'z': 0.0},
                    'orientation': {'x': 0.0, 'y': 0.0, 'z': qz, 'w': qw}
                },
                'covariance': [0.0]*36
            }
        })
        
        # Publish the reset message
        initialpose_topic.publish(reset_msg)
        print("Published reset command to /initialpose")
    
    def _reset_mpc_params(self):
        # Reset the MPC parameters to the default values
        for param, value in self.DEFAULT_MPC_PARAMS.items():
            self._set_ros_param(param, value, supress_print=True)
        # print("Reset MPC parameters to default values")
    
if __name__ == "__main__":
    load_dotenv(find_dotenv())
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")

    parser = argparse.ArgumentParser(description='Run with a specified model.')
    parser.add_argument('--model', choices=MODEL_OPTIONS, help='The model to use for the LLMMPC.')
    parser.add_argument('--model_dir', type=str, default="nibauman/RobotxLLM_Qwen7B_SFT", help='The model directory for the custom LLM.')
    parser.add_argument('--quant', action='store_true', help='Use quantization for the custom model, needs to be downloaded in models/ folder and dir towards it.')
    parser.add_argument('--prompt', nargs='?', default='I want you to drive forwards at 2m/s', help='The scenario prompt for the LLM MPC.')
    parser.add_argument('--hostip', type=str, default='192.168.192.107', help='The host IP for the ROS connection.')
    parser.add_argument('--mpconly', action='store_true', help='Only run the MPCxLLM.')
    args = parser.parse_args()
    
    # Model options: 'gpt-4o', 'custom'
    race_llm = RaceLLMMPC(openai_token=OPENAI_API_TOKEN,
                          model=args.model,
                          host_ip=args.hostip,
                          model_dir=args.model_dir,
                          quant=args.quant,)

    if args.mpconly:
        while True:
            prompt = input("Enter prompt: ")
            start_time = time.time()
            extracted_command , _, _, _, _ = race_llm.race_mpc_interact(scenario=prompt, memory_nb=2)
            print(f"MPCxLLM commands: {extracted_command}")
            print(f"⏱ Total Time: {time.time() - start_time:.2f}s\n")
    else:
        while True :
            start_time = time.time()
            llm_action_prompt, state = race_llm.race_reasoning(human_prompt=args.prompt, data_time=1.0, data_samples=5)
            reason_time = time.time() - start_time
            print("State: ", state)
            print(f"LLM Action Prompt: {llm_action_prompt}")
            print(f"Reasoning time: {reason_time}")
            if "Adhering to Human: True" in llm_action_prompt:
                print("No change necessary.")
            else:
                race_llm.race_mpc_interact(scenario=llm_action_prompt, memory_nb=2)
                mpc_time = time.time() - start_time - reason_time
                print(f"MPC time: {mpc_time}")
            total_time = time.time() - start_time
            print(f"⏱ Total time: {total_time}")
            time.sleep(0.05)

    # Example usage:
    # python3 llm_mpc.py --model custom --model_dir nibauman/RobotxLLM_Qwen7B_SFT --hostip 192.168.192.75 --prompt "Drive forwards"