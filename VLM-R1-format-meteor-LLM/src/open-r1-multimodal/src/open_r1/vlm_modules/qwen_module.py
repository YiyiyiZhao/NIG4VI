from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
import requests
import ast
from open_r1.vlm_modules.vlm_module import VLMBaseModule
import pdb
import re
import nltk
nltk.download('wordnet')
from nltk.translate.meteor_score import meteor_score
import time

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        model_cls = Qwen2_5_VLForConditionalGeneration
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return ['max_pixels', 'min_pixels']
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "nig":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in string format."
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_nig(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    @staticmethod
    def fallback_parsing(message):
        template = "\nAssume you are a blind person. Analyze the instruction through these steps:\n1. Determine if movement parameters exist:\n    - Extract direction: Convert any directional information (e.g., left, right, east, west) into the \"X o'clock\" format (e.g., \"2 o'clock\" for a slight right turn, \"9 o'clock\" for a left turn).\n    - Extract distance: Ensure the distance includes a numerical value and a unit (e.g., meters, steps).\n2. Check for danger alerts:\n    - Identify if the instruction includes detailed warnings about hazards (e.g., specific obstacles at specific directions or distances). If hazards are mentioned but lack detail, consider the alert as non-detailed.\n3. If both direction and distance are missing, or if the instruction is unclear or ambiguous, return None for the movement parameters.\n\nOutput Format:\n{\n  \"move\": {\n    \"direction\": \"X o'clock\",  // Replace \"X\" with the appropriate value (e.g., \"2 o'clock\").\n    \"distance\": \"Y meters/steps\"  // Replace \"Y\" with the numerical value and unit.\n  },\n  \"detailed_hazard_alert\": true/false  // Set to `true` if detailed hazard warnings are present, otherwise `false`.\n}\nor, if the instruction is invalid or incomplete:\n{\n  \"move\": None,\n  \"detailed_hazard_alert\": true/false  // Set to `true` if any hazard warnings are present, even if incomplete.\n}\n\nNow, solve the following task: "
        try:
            content = message["content"].replace(template, "")
            # Look for direction patterns
            direction_match = re.search(r'([0-9]+)\s*o\'?clock', content)
            direction = direction_match.group(0) if direction_match else None

            # Look for distance patterns
            distance_match = re.search(r'([0-9]+\s*(?:meters?|steps?|m))', content)
            distance = distance_match.group(0) if distance_match else None

            # Look for hazard mentions
            detailed_hazard = re.search(r'(obstacle|hazard|danger|warning|caution)', content, re.IGNORECASE) is not None

            return {
                'move': {
                    'direction': direction,
                    'distance': distance
                },
                'detailed_hazard_alert': detailed_hazard
            }
        except:
            return {'move': {'direction': None, 'distance': None}, 'detailed_hazard_alert': False}

    @staticmethod
    def call_inference_server(messages_list):
        url = "http://localhost:6006/infer"
        payload = {"messages_list": messages_list}
        response = requests.post(url, json=payload, timeout=20)
        data = response.json()
        if 'response' not in data:
            time.sleep(5)
            response = requests.post(url, json=payload, timeout=120)
            data=response.json()
            if 'response' not in data:
                time.sleep(5)
                response = requests.post(url, json=payload, timeout=120)
                data = response.json()
                if 'response' not in data:
                    time.sleep(5)
                    response = requests.post(url, json=payload, timeout=120)
                    data = response.json()

        if 'response' in data:
            response = data['response']
            result_dict_list = []
            for resp_i in response:
                if resp_i:
                    try:
                        result_dict = ast.literal_eval(resp_i.replace("0 o\'clock","12 o\'clock"))
                    except:
                        result_dict = Qwen2VLModule.fallback_parsing({"content":resp_i})
                else:
                    result_dict = Qwen2VLModule.fallback_parsing({"content":resp_i})
                result_dict_list.append(result_dict)
        else:
            result_dict_list = []
            for ii in range(len(messages_list)):
                result_dict = Qwen2VLModule.fallback_parsing(messages_list[ii])
                result_dict_list.append(result_dict)
        return result_dict_list

    @staticmethod
    def caption_reward_nig(completions, solution, **kwargs):
        template = "\nAssume you are a blind person. Analyze the instruction through these steps:\n1. Determine if movement parameters exist:\n    - Extract direction: Convert any directional information (e.g., left, right, east, west) into the \"X o'clock\" format (e.g., \"2 o'clock\" for a slight right turn, \"9 o'clock\" for a left turn).\n    - Extract distance: Ensure the distance includes a numerical value and a unit (e.g., meters, steps).\n2. Check for danger alerts:\n    - Identify if the instruction includes detailed warnings about hazards (e.g., specific obstacles at specific directions or distances). If hazards are mentioned but lack detail, consider the alert as non-detailed.\n3. If both direction and distance are missing, or if the instruction is unclear or ambiguous, return None for the movement parameters.\n\nOutput Format:\n{\n  \"move\": {\n    \"direction\": \"X o'clock\",  // Replace \"X\" with the appropriate value (e.g., \"2 o'clock\").\n    \"distance\": \"Y meters/steps\"  // Replace \"Y\" with the numerical value and unit.\n  },\n  \"detailed_hazard_alert\": true/false  // Set to `true` if detailed hazard warnings are present, otherwise `false`.\n}\nor, if the instruction is invalid or incomplete:\n{\n  \"move\": None,\n  \"detailed_hazard_alert\": true/false  // Set to `true` if any hazard warnings are present, even if incomplete.\n}\n\nNow, solve the following task: "
        messages_list=[]
        for completion in completions:
            content=completion[0]["content"]
            try:
                pred_answer_tmp = re.findall(r"<answer>([\s\S]*?)</answer>", content)
                pred_answer = pred_answer_tmp[0]
                messages = [{"role": "user", "content": template + pred_answer}]
            except:
                messages = [{"role": "user", "content": template + content}]
            messages_list.append(messages)
        try:
            pred_result_dict_list= Qwen2VLModule.call_inference_server(messages_list)
            status_code=200
        except Exception as e:
            pred_result_dict_list = []
            for ii in range(len(messages_list)):
                pred_result_dict_list.append(Qwen2VLModule.fallback_parsing(messages_list[ii]))
            status_code=500
        rewards = []

        for content, sol in zip(pred_result_dict_list, solution):
            reward, move_reward_1, move_reward_2, alert_reward = 0.0, 0.0, 0.0, 0.0
            pred_dict=content
            sol_dict=sol
            if status_code == 500:
                rewards.append(0.65) #avoid the influence by network lost
                time.sleep(7) # for network connecting
                continue

            if sol_dict and 'move' in sol_dict and 'direction' in sol_dict['move'] and sol_dict['move']['direction'] == "0 o\'clock":
                sol_dict['move']['direction'] = "12 o\'clock"

            if pred_dict is not None and 'move' in pred_dict and pred_dict['move'] is not None and 'direction' in pred_dict['move'] and pred_dict['move']['direction'] == "0 o'clock":
                pred_dict['move']['direction'] = "12 o\'clock"

            if ('move' in pred_dict) and ('move' in sol_dict) and (pred_dict['move'] is not None) and (sol_dict['move'] is not None) and ('direction' in pred_dict['move']) and ('direction' in sol_dict['move']) and (pred_dict['move']['direction'] == sol_dict['move']['direction']):
                move_reward_1 = 0.4

            if ('move' in pred_dict) and ('move' in sol_dict) and (pred_dict['move'] is not None) and (sol_dict['move'] is not None) and ('distance' in pred_dict['move']) and ('distance' in sol_dict['move']) and (pred_dict['move']['distance'] == sol_dict['move']['distance']):
                move_reward_2 = 0.4

            if ('detailed_hazard_alert' in pred_dict) and ('detailed_hazard_alert' in sol_dict) and (pred_dict['detailed_hazard_alert'] == sol_dict['detailed_hazard_alert']):
                alert_reward = 0.3

            print('\n', "**************","pred_dict", pred_dict, '\n',"**************","sol_dict", sol_dict)
            reward = move_reward_1+move_reward_2 + alert_reward
            rewards.append(reward)
        print("status_code", status_code,'\n')
        return rewards


    @staticmethod
    def cal_meteor_scores(reference_list, hypothesis_list):
        scores = []
        for reference, hypothesis in zip(reference_list, hypothesis_list):
            reference_tokens = reference.split()
            hypothesis_tokens = hypothesis.split()
            score = meteor_score([reference_tokens], hypothesis_tokens)
            scores.append(score)
        return scores

    @staticmethod
    def meteor_reward_nig(completions, solution_str, **kwargs):
        content_list = []
        for completion in completions:
            content = completion[0]["content"]
            pred_answer_tmp = re.findall(r"<answer>([\s\S]*?)</answer>", content)
            if len(pred_answer_tmp) > 0:
                pred_answer = pred_answer_tmp[0]
            else:
                pred_answer = ''
            content_list.append(pred_answer)
        rewards = Qwen2VLModule.cal_meteor_scores(content_list, solution_str)
        return rewards