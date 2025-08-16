# Params for f-map
EVAL_DRIVING_CASES = {
    "center": [
    {"human_prompt":"Stay directly on the middle of the track", "evaluation_function": "center_rmse", "baseline_rmse": 0.727},
    {"human_prompt":"Follow the track by staying aligned with the middle of the track", "evaluation_function": "center_rmse", "baseline_rmse": 0.727},
    {"human_prompt":"Drive away as far as possible from the walls", "evaluation_function": "center_rmse", "baseline_rmse": 0.727},
    {"human_prompt":"Ensure that the distance to the left and right wall remain the same", "evaluation_function": "center_rmse", "baseline_rmse": 0.727},
    {"human_prompt":"Drive on the centerline", "evaluation_function": "center_rmse", "baseline_rmse": 0.727}],

    # "raceline": [
    # {"human_prompt":"Keep driving on the racing line", "evaluation_function": "raceline_rmse", "baseline_rmse": 0.145},
    # {"human_prompt":"Follow the racing line", "evaluation_function": "raceline_rmse", "baseline_rmse": 0.145},
    # {"human_prompt":"Drive along the racing line.", "evaluation_function": "raceline_rmse", "baseline_rmse": 0.145},
    # {"human_prompt":"Hold your path on the racing line", "evaluation_function": "raceline_rmse", "baseline_rmse": 0.145},
    # {"human_prompt":"Stick to racing line", "evaluation_function": "raceline_rmse", "baseline_rmse": 0.145}],

    "refvel": [
    {"human_prompt":"Set the driving speed to 3.5 m/s", "evaluation_function": "refvel_rmse", "ref_val": 3.5, "baseline_rmse": 1.160},
    {"human_prompt":"Target a driving speed of 2.2 meters per second", "evaluation_function": "refvel_rmse", "ref_val": 2.2, "baseline_rmse": 1.375},
    {"human_prompt":"Move at a constant speed of 1.25 m/s", "evaluation_function": "refvel_rmse", "ref_val": 1.25, "baseline_rmse": 2.151},
    {"human_prompt":"Travel at 2.9 meters per second", "evaluation_function": "refvel_rmse", "ref_val": 2.9, "baseline_rmse": 1.01},
    {"human_prompt":"Adjsut the speed to exactly 4.5 m/s", "evaluation_function": "refvel_rmse", "ref_val": 4.5, "baseline_rmse":  1.654}],
    
    "reverse": [
    {"human_prompt":"Slowly back the vehicle up", "evaluation_function": "reverse_rmse", "baseline_rmse": 4.29},
    {"human_prompt":"Reverse the vehicle", "evaluation_function": "reverse_rmse", "baseline_rmse": 4.29},
    {"human_prompt":"Switch to reverse and drive backwards", "evaluation_function": "reverse_rmse", "baseline_rmse": 4.29},
    {"human_prompt":"Retreat by reversing the car", "evaluation_function": "reverse_rmse", "baseline_rmse": 4.29},
    {"human_prompt":"Drive the car backwards", "evaluation_function": "reverse_rmse", "baseline_rmse": 4.29}],

    "smooth": [
    {"human_prompt":"Drive smoothly", "evaluation_function": "smooth_rmse", "baseline_rmse": 1.563},
    {"human_prompt":"Reduce lateral acceleration", "evaluation_function": "smooth_rmse", "baseline_rmse": 1.563},
    {"human_prompt":"Drive in a fluid and controlled manner", "evaluation_function": "smooth_rmse", "baseline_rmse": 1.563},
    {"human_prompt":"Maintain low jerk and high smoothness in driving behavior", "evaluation_function": "smooth_rmse", "baseline_rmse": 1.563},
    {"human_prompt":"Focus on smooth driving", "evaluation_function": "smooth_rmse", "baseline_rmse": 1.563}]
}
