import os
import shutil
import numpy as np
import imageio
from typing import List
import os
import re
import pprint
import time
from pathlib import Path
os.environ['IMAGE_MAX_TOKEN_NUM'] = '256'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '64'
os.environ['FPS_MAX_FRAMES'] = '10'

import json
from swift.llm import PtEngine, InferRequest, RequestConfig
from .qwen_cache import ContentAddressedQwenCache, append_jsonl, hash_path

class Qwen3VLModel:
    
    def __init__(self, 
        adapter_path: str,
        base_model_path: str = "Qwen/Qwen3-VL-4B-Instruct",
        subgoal_type: str = "simple_subgoal", 
        cache_dir: str | None = None,
    ):
        self.model_name = "qwenvl"
        self.subgoal_type = subgoal_type
        self.image_size = (256, 256)
        self.adapter_path = adapter_path
        self.cache = (
            ContentAddressedQwenCache(cache_dir, adapter_path, base_model_path)
            if cache_dir is not None
            else None
        )
        self.generation_config = {"max_tokens": 128, "temperature": 0}
        self.prompt_version = "robomme-qwenvl-official-v1"
        
        assert subgoal_type in ["simple_subgoal", "grounded_subgoal"]
        
        # Load appropriate prompt dictionary
        if subgoal_type == "simple_subgoal":
            self.system_prompt = "You are a helpful assistant to help guide the robot to complete the task by predicting a sequence of language subgoals"
        elif subgoal_type == "grounded_subgoal":
            self.system_prompt = "You are a helpful assistant to help guide the robot to complete the task by predicting a sequence of grounded language subgoals"
        else:
            raise ValueError(f"Invalid subgoal type: {subgoal_type}")
        
        print(f"Loading Qwen3-VL-4B-Instruct model Adapter from {adapter_path}")
        self.engine = PtEngine(
            model_id_or_path=base_model_path,
            adapters=[adapter_path],
            # The cluster has no CUDA compiler for the optional flash-attn
            # extension. RoboMME documents SDPA as the supported fallback.
            attn_impl='sdpa'
        )
        
    def _parse_box_patterns(self, subgoal: str, replacement: str = "scaled_coords", return_bbox: bool = False):
        """
        Parse box patterns from subgoal and replace them.
        
        Args:
            subgoal: The subgoal string containing box patterns
            replacement: Either "scaled_coords" to replace with <x, y> or "bbox" to replace with <bbox>
            return_bbox: If True, also return the list of bbox coordinates
        
        Returns:
            If return_bbox is False: modified subgoal string
            If return_bbox is True: tuple of (modified subgoal string, bbox list)
        """
        matches = re.findall(r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>', subgoal)
        
        qwen3_vl_image_size = (1000, 1000)
        
        if len(matches) == 0:
            if return_bbox:
                return subgoal, []
            return subgoal
        
        # Extract bbox coordinates (scaled)
        bbox = [[int(float(match[0])/qwen3_vl_image_size[1]*self.image_size[1]), 
                 int(float(match[1])/qwen3_vl_image_size[0]*self.image_size[0])] for match in matches]
        
        # Replace based on replacement type
        if replacement == "scaled_coords":
            response = re.sub(
                r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>',
                lambda m: f'<{int(int(m.group(1)) * self.image_size[1] / qwen3_vl_image_size[1])}, {int(int(m.group(2)) * self.image_size[0] / qwen3_vl_image_size[0])}>',
                subgoal
            )
        elif replacement == "bbox":
            response = re.sub(
                r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>',
                '<bbox>',
                subgoal
            )
        else:
            raise ValueError(f"Invalid replacement type: {replacement}")
        
        if return_bbox:
            return response, bbox
        return response
    
    def _parse_subgoal_for_vla(self, subgoal: str) -> str:
        """Parse subgoal and replace box patterns with scaled coordinates for VLA."""
        return self._parse_box_patterns(subgoal, replacement="scaled_coords", return_bbox=False)
    
    def _parse_grounded_subgoal(self, subgoal) -> tuple:
        """Preprocess grounded subgoal by replacing box patterns with <bbox> and extracting bbox coordinates."""
        return self._parse_box_patterns(subgoal, replacement="bbox", return_bbox=True)
    
    def start_new_episode(self, save_dir: str, video_query: List[np.ndarray]| None, task_goal: str = None) -> dict:
        self.save_dir = save_dir
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        ep_name = os.path.basename(save_dir)
        self.save_json_path = os.path.join(os.path.dirname(save_dir), f"{ep_name}_QwenVL_log.jsonl")

        
        if video_query is not None and len(video_query) > 0:
            imageio.mimsave(os.path.join(self.save_dir, f"step_0_video.mp4"), video_query, fps=30)
            self.video_path = os.path.join(self.save_dir, f"step_0_video.mp4")
        else:
            self.video_path = None
        self.task_goal = task_goal
        self.conversation_history = []
        self.total_images = []
        self.subgoals = []
        self.history_simple_subgoals = []
        self.history_grounded_subgoals = []
        self.history_grounded_bboxes = []
        self.last_response = None
        self.call_log_path = os.path.join(os.path.dirname(save_dir), f"{ep_name}_QwenVL_calls.jsonl")
        self.video_hash = hash_path(Path(self.video_path)) if self.video_path else None
     
    def _wrap_history_subgoals(self, subgoals) -> str:
        return "; ".join([f"{i+1}. {subgoal}" for i, subgoal in enumerate(subgoals)])
    
    def _parse_grounded_subgoal(self, subgoal) -> tuple:
        bbox = []
        # seatch the pattern "at <y, x>"
        matches = re.findall(r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>', subgoal)
        if matches:
            bbox = [[int(float(match[0])/1000*self.image_size[1]), int(float(match[1])/1000*self.image_size[0])] for match in matches]
        else:
            bbox = []        
        response = re.sub(
            r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>',
            '<bbox>',
            subgoal
        )
        
        return response, bbox
    
    def update_history_subgoals(self, subgoal: str):
        if self.subgoal_type == "simple_subgoal":
            if self.history_simple_subgoals:
                if self.history_simple_subgoals[-1] != subgoal:
                    self.history_simple_subgoals.append(subgoal)
            else:
                self.history_simple_subgoals.append(subgoal)
        else:
            assistant_prompt, bbox = self._parse_grounded_subgoal(subgoal)
            if self.history_grounded_subgoals:
                if self.history_grounded_subgoals[-1] != assistant_prompt:
                    self.history_grounded_subgoals.append(assistant_prompt)
                    self.history_grounded_bboxes.extend(bbox)
            else:
                self.history_grounded_subgoals.append(assistant_prompt)
                self.history_grounded_bboxes.extend(bbox)
    
    def prepare_infer_request(self, image_query: np.ndarray, step_idx: int) -> dict:
        
        image_path = os.path.join(self.save_dir, f"step_{step_idx}_image.png")
        imageio.imwrite(image_path, image_query)
        video_prefix = "<video>" if self.video_path else ""
        
        if self.subgoal_type == "simple_subgoal":            
            if len(self.history_simple_subgoals) == 0:
                user_prompt = f"{video_prefix}The task goal is: {self.task_goal}\nThis is the initial turn for prediction\n<image>What's the next language subgoal based on current observation?"
            else:
                user_prompt = f"{video_prefix}The task goal is: {self.task_goal}\nThe history of previous predicted language subgoals are: {self._wrap_history_subgoals(self.history_simple_subgoals)}\n<image>What's the next language subgoal based on current observation?"
                    
        else:        
            if len(self.history_grounded_subgoals) == 0:
                user_prompt = f"{video_prefix}The task goal is: {self.task_goal}\nThis is the initial turn for prediction\n<image>What's the next grounded language subgoal based on current observation?"
            else:            
                user_prompt = f"{video_prefix}The task goal is: {self.task_goal}\nThe history of previous predicted grounded language subgoals are: {self._wrap_history_subgoals(self.history_grounded_subgoals)}\n<image>What's the next grounded language subgoal based on current observation?"
        
        infer_request_dict = {
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "images": [image_path]
        }
        
        if self.video_path is not None:
            infer_request_dict["videos"] = [self.video_path]
            
        if self.subgoal_type == "grounded_subgoal":
            infer_request_dict["objects"] = {"ref": [], "bbox": self.history_grounded_bboxes}
        
        print("\n\n")
        pprint.pprint(infer_request_dict)
        
        with open(self.save_json_path, "a") as f:
            json.dump(infer_request_dict, f)
            f.write("\n")

        return InferRequest(**infer_request_dict)
    
    
    def call(self, image_query: np.ndarray, step_idx: int, keep_period: int = 0) -> str:        
        started = time.monotonic()
        cache_key = None
        cache_payload = None
        cache_hit = False
        input_tokens = None
        output_tokens = None
        reuse_reason = None
        if step_idx <= keep_period and self.last_response is not None:
            # some tasks that require press button, qwen models always skip
            # add some hard-coded rules to fix it
            response = self.last_response
            reuse_reason = "official_keep_period_reuse"
        else:
            if self.cache is not None:
                history = (
                    self.history_simple_subgoals
                    if self.subgoal_type == "simple_subgoal"
                    else self.history_grounded_subgoals
                )
                cache_key, cache_payload = self.cache.key(
                    current_image=image_query,
                    task_instruction=self.task_goal,
                    subgoal_history=list(history),
                    prompt_version=self.prompt_version,
                    generation=self.generation_config,
                    video_hash=self.video_hash,
                )
                cached = self.cache.get(cache_key)
            else:
                cached = None
            if cached is not None:
                response = cached["raw_response"]
                input_tokens = cached.get("input_tokens")
                output_tokens = cached.get("output_tokens")
                cache_hit = True
            else:
                infer_request = self.prepare_infer_request(image_query, step_idx)
                infer_result = self.engine.infer(
                    [infer_request],
                    request_config=RequestConfig(**self.generation_config),
                )[0]
                response = infer_result.choices[0].message.content
                usage = getattr(infer_result, "usage", None)
                input_tokens = (
                    usage.get("prompt_tokens")
                    if isinstance(usage, dict)
                    else getattr(usage, "prompt_tokens", None)
                )
                output_tokens = (
                    usage.get("completion_tokens")
                    if isinstance(usage, dict)
                    else getattr(usage, "completion_tokens", None)
                )
                if self.cache is not None:
                    self.cache.put(
                        cache_key,
                        {
                            "cache_key_payload": cache_payload,
                            "raw_response": response,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                        },
                    )
        
        print("Response: ", response)
        self.last_response = response
        self.update_history_subgoals(response)
        parsed = self._parse_subgoal_for_vla(response)
        parsed_points = re.findall(r'<(\d+), (\d+)>', parsed)
        coordinate_valid = bool(parsed_points) and all(
            0 <= int(first) <= 255 and 0 <= int(second) <= 255
            for first, second in parsed_points
        )
        append_jsonl(
            self.call_log_path,
            {
                "step_idx": step_idx,
                "cache_key": cache_key,
                "cache_hit": cache_hit,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": (time.monotonic() - started) * 1000,
                "raw_response": response,
                "parsed_groundsg": parsed,
                "coordinate_valid": coordinate_valid,
                "reuse_reason": reuse_reason,
                "predictor_checkpoint_sha256": (
                    self.cache.predictor_checkpoint_hash if self.cache is not None else None
                ),
                "base_model_checkpoint_sha256": (
                    self.cache.base_model_checkpoint_hash if self.cache is not None else None
                ),
                "generation": self.generation_config,
            },
        )
        return parsed
