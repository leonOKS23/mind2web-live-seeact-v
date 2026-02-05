import argparse
import os
import pdb
import re
import json
from typing import Any, Optional
import tiktoken
from beartype import beartype
from PIL import Image
import math
from openai import OpenAI
from agent.prompts.grounding_prompt import GROUNDING_PROMPT, GROUNDING_SYSTEM_PROMPT
from agent.prompts.grounding_prompt_constructor import ground_prompt_constructor
from evaluation_harness.openai import GPTGenerator
from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
    create_coordinated_action
)
from browser_env.utils import Observation, StateInfo, pil_to_b64
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)
from llms.providers.openai_utils import chat_with_ug, chat_with_141
from llms.tokenizers import Tokenizer
from browser_env.utils import *


def pil_to_b64(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64


gpt = GPTGenerator()


class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
            self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
            self,
            test_config_file: str,
    ) -> None:
        raise NotImplementedError


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
            self,
            action_set_tag: str,
            lm_config: lm_config.LMConfig,
            prompt_constructor: PromptConstructor,
            captioning_fn=None,
            grounding_method: str = "uground",
            grounding_model: str | None = None,
            grounding_api_base: str | None = None,
            uground_script_path: str | None = None,
            uground_script_python: str | None = None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn
        self.output_dict_list = []
        self.last_offset_y = 0
        # Check if the model is multimodal.
        self.multimodal_inputs = True
        self.memory = []
        self.warning = ""
        self.grounding_method = grounding_method
        self.grounding_model = grounding_model
        self.grounding_api_base = grounding_api_base
        self.uground_script_path = uground_script_path
        self.uground_script_python = uground_script_python

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def get_images(self, trajectory, num_images):
        images = []
        # 获取指定数量的间隔图片
        for i in range(0, len(trajectory), 2):
            images.append(trajectory[i]["observation"]["raw_screenshot"])

        images = images[-num_images:]
        return images

    def call_uground(self, expression, image):

        def pil_to_b64_forug(img: Image.Image) -> str:
            with BytesIO() as image_buffer:
                img.save(image_buffer, format="PNG")
                byte_data = image_buffer.getvalue()
                img_b64 = base64.b64encode(byte_data).decode("utf-8")
            return img_b64



        response = chat_with_ug(os.environ["LOCAL_UG_SERVER"],
                                image=pil_to_b64_forug(image),
                                prompt=expression)

        print("CALLING GROUND FINISHED. Response:", response, flush=True)

        def extract_coordinates(s):
            # 使用正则表达式匹配数字
            numbers = re.findall(r'\d+', s)

            # 将字符串转换为整数
            coordinates = list(map(int, numbers))

            # 确保只返回前四个值
            return coordinates[:4]

        coordinates = response['fix_c']
        return coordinates

    def call_uground2(self, expression, image):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
  Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

  - Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
  - If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
  - Your answer should be a single string (x, y) corresponding to the point of the interest.

  Description: {expression}

  Answer:"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pil_to_b64(image)
                        }
                    }
                ]
            }
        ]

        response = chat_with_141(os.environ["LOCAL_UG_SERVER"],
                                 model="UGround",
                                 messages=messages,
                                 temperature=0.0,
                                 port=23333)

        print("CALLING GROUND FINISHED. Response:", response, flush=True)
        coordinate = eval(response["answer"])
        coordinate = (coordinate[0] / 1000, coordinate[1] / 1000)

        return coordinate

    def calculate_tokens(self, image_path, max_tokens=450, pixels_per_token=768):
        # 打开图像
        if not isinstance(image_path, Image.Image):
            image = Image.open(image_path)
        else:
            image = image_path
        width, height = image.size
        # 计算图像的总像素数
        total_pixels = width * height

        # 计算消耗的token数量
        tokens = math.ceil(total_pixels / pixels_per_token)

        if tokens > max_tokens:
            # 需要resize图像
            scale_factor = math.sqrt(max_tokens * pixels_per_token / total_pixels)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # 保持宽高比进行resize
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            (nw, nh) = resized_image.size
            assert nw * nh >= 2
            return resized_image
        else:
            # 不需要resize
            return image

    def extract_ug_values(self, action_string):

        element_description_pattern = r'"Element Description": (.*?)\n'
        action_pattern = r'"Action": (.*?)\n'
        value_pattern = r'"Value": (.*?)\n'
        element_match = re.search(element_description_pattern, action_string, re.DOTALL)
        action_match = re.search(action_pattern, action_string, re.DOTALL)
        value_match = re.search(value_pattern, action_string, re.DOTALL)
        element_description = element_match.group(1) if element_match else ""
        action = action_match.group(1) if action_match else ""
        value = value_match.group(1) if value_match else ""

        def process_output(output):
            if output.startswith("\""):
                output = output[1:]
            if output.endswith(","):
                output = output[:-1]
            if output.endswith("\""):
                output = output[:-1]
            return output

        element_description = process_output(element_description)
        action = process_output(action)
        value = process_output(value)

        try:
            if action == "":
                element_description_pattern = r'\*\*Element Description\*\*: (.*?)\n'
                action_pattern = r'\*\*Action\*\*: (.*?)\n'
                value_pattern = r'\*\*Value\*\*: (.*?)\n'
                element_match = re.search(element_description_pattern, action_string, re.DOTALL)
                action_match = re.search(action_pattern, action_string, re.DOTALL)
                value_match = re.search(value_pattern, action_string, re.DOTALL)
                element_description = element_match.group(1) if element_match else ""
                action = action_match.group(1) if action_match else ""
                value = value_match.group(1) if value_match else ""
        except:
            return element_description, action, value

        return element_description, action, value

    def find_nearest_point(self, bbox, x, y):
        def is_within_bounds(element, x, y):
            return element['left'] <= x <= element['right'] and element['top'] <= y <= element['bottom']

        def calculate_distance(x1, y1, x2, y2):
            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        nearest_point = None
        min_distance = float('inf')
        Interactable_bbox = []
        bbox = bbox.to_dict(orient='records')
        for b in bbox:
            interactable = b["Interactable"]
            top, right, bottom, left = b["Top"], b["Right"], b["Bottom"], b["Left"]
            if interactable:
                Interactable_bbox.append({"top": top, "right": right, "bottom": bottom, "left": left})
        if len(Interactable_bbox) == 0:
            print("length of Interactable_bbox is 0")
            return (x, y)
        for element in Interactable_bbox:
            if is_within_bounds(element, x, y):
                print("WITHIN BOUNDS")
                center_x = (element['left'] + element['right']) / 2
                center_y = (element['top'] + element['bottom']) / 2
                return (center_x, center_y)

            # Calculate the center point of the rectangle
            center_x = (element['left'] + element['right']) / 2
            center_y = (element['top'] + element['bottom']) / 2

            # Calculate distance from the point to the center of the rectangle
            distance = calculate_distance(x, y, center_x, center_y)

            if distance < min_distance:
                min_distance = distance
                nearest_point = (center_x, center_y)
        print("NEAREST_POINT", nearest_point)
        return nearest_point

    def call_atlas(self, expression, image):

        def calculate_midpoint(x0, y0, x1, y1):
            midpoint_x = (x0 + x1) / 2
            midpoint_y = (y0 + y1) / 2
            midpoint_x = midpoint_x / 1000
            midpoint_y = midpoint_y / 1000
            return (midpoint_x, midpoint_y)

        def pil_to_b64_for_atlas(img: Image.Image) -> str:
            with BytesIO() as image_buffer:
                img.save(image_buffer, format="PNG")
                byte_data = image_buffer.getvalue()
                img_b64 = base64.b64encode(byte_data).decode("utf-8")
            return img_b64

        content = [
            {
                "type": "text",
                'text': f'<IMAGE_TOKEN>\nIn the screenshot of this web page, please give me the coordinates of the element (with point).\n{expression}.',
            },

        ]
        content.extend(
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": pil_to_b64_for_atlas(image),
                        "max_dynamic_patch": 16

                    },
                }
            ]
        )
        user_input = {
            "role": "user",
            "content": content
        }
        messages = [user_input]

        response = chat_with_141(os.environ["LOCAL_UG_SERVER"],
                                 model="/disk2/OS-Atlas-Base-7B/Qwen2-VL-7B-Instruct",
                                 messages=messages,
                                 temperature=0.01,
                                 max_tokens=512,
                                 port=8000)
        response = response["answer"]
        coordinates = re.findall(r'\((\d+),(\d+)\)', response)
        coordinates = [(int(x), int(y)) for x, y in coordinates]
        x0, y0, x1, y1 = coordinates[0][0], coordinates[0][1], coordinates[1][0], coordinates[1][1]
        coordinates = calculate_midpoint(x0, y0, x1, y1)
        return coordinates

    def _normalize_point(self, x, y, width, height):
        x = float(x)
        y = float(y)
        if x > 1 or y > 1:
            if x <= 1000 and y <= 1000:
                x = x / 1000.0
                y = y / 1000.0
            else:
                x = x / width
                y = y / height
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        return (x, y)

    def _parse_grounding_response(self, response_text, width, height):
        if not response_text:
            return (0.5, 0.5)

        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group(0))
                if isinstance(payload, dict):
                    if "x" in payload and "y" in payload:
                        return self._normalize_point(payload["x"], payload["y"], width, height)
                    if "point" in payload and isinstance(payload["point"], (list, tuple)) and len(payload["point"]) >= 2:
                        return self._normalize_point(payload["point"][0], payload["point"][1], width, height)
            except Exception:
                pass

        numbers = re.findall(r"-?\d+(?:\.\d+)?", response_text)
        if len(numbers) >= 4:
            x0, y0, x1, y1 = map(float, numbers[:4])
            x = (x0 + x1) / 2.0
            y = (y0 + y1) / 2.0
            return self._normalize_point(x, y, width, height)
        if len(numbers) >= 2:
            x, y = map(float, numbers[:2])
            return self._normalize_point(x, y, width, height)

        return (0.5, 0.5)

    def call_qwen_grounding(self, expression, image):
        # For Qwen grounding we default to the same server as planning (OPENAI_API_BASE).
        # Use --grounding_api_base only if you explicitly want a different grounding server.
        # Uses the same UGround prompt format (works well with Qwen2-VL/Qwen3-VL based models).
        base_url = self.grounding_api_base or os.environ.get("OPENAI_API_BASE")
        if not base_url:
            raise ValueError("OPENAI_API_BASE must be set for qwen grounding.")

        model = self.grounding_model or self.lm_config.model
        client = OpenAI(api_key="EMPTY", base_url=base_url)
        # Official UGround prompt format (image first, then text)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(image)},
                    },
                    {
                        "type": "text",
                        "text": f"""Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- Your answer should be a single string (x, y) corresponding to the point of the interest.

Description: {expression}

Answer:""",
                    },
                ],
            }
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,  # IMPORTANT: temperature=0 for grounding
            max_tokens=128,
        )
        answer = response.choices[0].message.content
        print("QWEN_GROUNDING_RESPONSE:", answer, flush=True)
        width, height = image.size
        return self._parse_grounding_response(answer, width, height)

    def call_uground_vllm(self, expression, image):
        base_url = (
            self.grounding_api_base
            or os.environ.get("GROUNDING_OPENAI_API_BASE")
            or os.environ.get("OPENAI_API_BASE")
        )
        if not base_url:
            raise ValueError("GROUNDING_OPENAI_API_BASE or OPENAI_API_BASE must be set for uground_vllm.")

        model = self.grounding_model or self.lm_config.model
        client = OpenAI(api_key="EMPTY", base_url=base_url)
        # Official UGround prompt format (image first, then text)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(image)},
                    },
                    {
                        "type": "text",
                        "text": f"""Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- Your answer should be a single string (x, y) corresponding to the point of the interest.

Description: {expression}

Answer:""",
                    },
                ],
            }
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,  # IMPORTANT: UGround requires temperature=0
            max_tokens=128,
        )
        answer = response.choices[0].message.content
        print("UGROUND_VLLM_RESPONSE:", answer, flush=True)
        width, height = image.size
        # Parse (x, y) or bbox, then normalize
        return self._parse_grounding_response(answer, width, height)

    def call_uground_script(self, expression, image):
        import tempfile
        import subprocess

        script_path = (
            self.uground_script_path
            or os.environ.get("UGROUND_SCRIPT_PATH")
            or "uground_qwen3vl.py"
        )
        python_exe = (
            self.uground_script_python
            or os.environ.get("UGROUND_SCRIPT_PYTHON")
            or "python"
        )

        base_url = (
            self.grounding_api_base
            or os.environ.get("GROUNDING_OPENAI_API_BASE")
            or os.environ.get("OPENAI_API_BASE")
        )
        if not base_url:
            raise ValueError("GROUNDING_OPENAI_API_BASE or OPENAI_API_BASE must be set for uground_script.")

        model = self.grounding_model or self.lm_config.model
        width, height = image.size

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img.png")
            image.save(img_path)
            question_path = os.path.join(tmpdir, "question.jsonl")
            answer_path = os.path.join(tmpdir, "answers.jsonl")
            with open(question_path, "w") as f:
                f.write(json.dumps({"description": expression, "img_filename": "img.png"}) + "\n")

            env = os.environ.copy()
            env["OPENAI_API_BASE"] = base_url

            cmd = [
                python_exe,
                script_path,
                "--question-file",
                question_path,
                "--answers-file",
                answer_path,
                "--image-folder",
                tmpdir,
                "--model",
                model,
            ]
            subprocess.run(cmd, check=True, env=env)

            with open(answer_path, "r") as f:
                line = f.readline().strip()
            if not line:
                return (0.5, 0.5)

            payload = json.loads(line)
            coords_text = payload.get("output", "")
            coords = re.findall(r"-?\d+(?:\.\d+)?", coords_text)
            if len(coords) >= 2:
                x_abs, y_abs = float(coords[0]), float(coords[1])
                return self._normalize_point(x_abs, y_abs, width, height)

        return (0.5, 0.5)

    def _tokenize(self, text):
        return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 2]

    def _infer_url_from_intent(self, intent: str) -> str | None:
        if not intent:
            return None
        intent_l = intent.lower()
        # If a domain is explicitly mentioned, use it.
        domain_match = re.search(r"(https?://[^\s]+|[a-z0-9-]+\.[a-z]{2,})", intent_l)
        if domain_match:
            url = domain_match.group(1)
            if not url.startswith("http"):
                url = f"https://{url}"
            return url

        # Simple site-name to URL mapping for common sites
        site_map = {
            "gamestop": "https://www.gamestop.com",
            "amazon": "https://www.amazon.com",
            "wikipedia": "https://www.wikipedia.org",
            "reddit": "https://www.reddit.com",
            "youtube": "https://www.youtube.com",
            "google": "https://www.google.com",
            "ebay": "https://www.ebay.com",
            "walmart": "https://www.walmart.com",
            "target": "https://www.target.com",
            "bestbuy": "https://www.bestbuy.com",
            "best buy": "https://www.bestbuy.com",
            "etsy": "https://www.etsy.com",
            "facebook": "https://www.facebook.com",
            "twitter": "https://x.com",
            "x": "https://x.com",
            "instagram": "https://www.instagram.com",
            "linkedin": "https://www.linkedin.com",
            "microsoft": "https://www.microsoft.com",
            "apple": "https://www.apple.com",
            "imdb": "https://www.imdb.com",
            "steam": "https://store.steampowered.com",
        }
        for key, url in site_map.items():
            if key in intent_l:
                return url
        return None

    def _maybe_override_homepage_click(self, intent, current_url, action, expression, value):
        expr_l = (expression or "").lower()
        if "homepage" in expr_l or "homepage.com" in expr_l:
            inferred = self._infer_url_from_intent(intent)
            if inferred:
                print(f"OVERRIDE: homepage click -> goto {inferred}", flush=True)
                return "", "goto", inferred
        # If we're on about:blank and not doing goto, prefer direct goto
        if (current_url or "").startswith("about:blank") and action not in ["goto", "go_back", "go_forward"]:
            inferred = self._infer_url_from_intent(intent)
            if inferred:
                print(f"OVERRIDE: about:blank -> goto {inferred}", flush=True)
                return "", "goto", inferred
        return expression, action, value

    def call_simple_grounding(self, expression, observation, image):
        text_obs = observation.get("text", "") if isinstance(observation, dict) else ""
        id2center = observation.get("id2center", {}) if isinstance(observation, dict) else {}
        width, height = image.size

        expr_tokens = set(self._tokenize(expression))
        best_id = None
        best_score = 0

        if text_obs and id2center:
            for line in text_obs.splitlines():
                match = re.match(r"^\[(\d+)\]\s*\[[^\]]*\]\s*\[(.*)\]\s*$", line.strip())
                if not match:
                    continue
                elem_id = match.group(1)
                elem_text = match.group(2)
                elem_tokens = set(self._tokenize(elem_text))
                if not elem_tokens:
                    continue
                score = len(expr_tokens & elem_tokens)
                if score > best_score:
                    best_score = score
                    best_id = elem_id

        if best_id is not None and str(best_id) in id2center:
            center = id2center[str(best_id)]
            return self._normalize_point(center[0], center[1], width, height)

        bboxes = observation.get("bboxes", None) if isinstance(observation, dict) else None
        if bboxes is not None:
            try:
                point = self.find_nearest_point(bboxes, width / 2.0, height / 2.0)
                return self._normalize_point(point[0], point[1], width, height)
            except Exception:
                pass

        return (0.5, 0.5)

    @beartype
    def next_action(
            self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any],
            images=None,
            output_response: bool = False, **kwargs
    ) -> Action:
        # Create page screenshot image for multimodal models.
        # images = [self.calculate_tokens(image, 600) for image in images]
        grounding_method = self.grounding_method

        page_screenshot_list = self.get_images(trajectory, 1)
        page_screenshot_list = [Image.fromarray(image) for image in page_screenshot_list]
        observation = trajectory[-1]["observation"]
        bboxes = observation.get("bboxes")
        construct_kwargs = dict(kwargs)
        # Only pass supported kwargs (some prompt constructors don't accept "warnings")
        try:
            import inspect
            sig = inspect.signature(self.prompt_constructor.construct)
            if "warnings" in sig.parameters:
                construct_kwargs["warnings"] = self.warning
        except Exception:
            pass

        prompt = self.prompt_constructor.construct(
            trajectory, intent, page_screenshot_list,
            images, meta_data,
            **construct_kwargs
        )

        lm_config = self.lm_config
        n = 0
        while True:
            response = call_llm(lm_config, prompt, num_outputs=1)
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            if output_response:
                print(f'Agent: {response}', flush=True)
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(
                    response
                )
                parsed_response = parsed_response.strip()
                expression, action, value = self.extract_ug_values(parsed_response)
                expression = expression.strip()
                action = action.strip()
                value = value.strip()
                action_coordinate = None

                print(expression, action, value, flush=True)
                raw_action = action
                if action in ["memorize"]:
                    self.memory.append(value)
                    while action != "memorize":
                        prompt = self.prompt_constructor.construct(
                            trajectory, intent, page_screenshot_list,
                            images, meta_data, memory=self.memory, **construct_kwargs
                        )
                        response = call_llm(lm_config, prompt, num_outputs=1)
                        force_prefix = self.prompt_constructor.instruction[
                            "meta_data"
                        ].get("force_prefix", "")
                        response = f"{force_prefix}{response}"
                        if output_response:
                            print(f'Agent: {response}', flush=True)
                        parsed_response = parsed_response.strip()
                        expression, action, value = self.extract_ug_values(parsed_response)
                        expression = expression.strip()
                        action = action.strip()
                        value = value.strip()
                action = action.lower()
                current_url = ""
                try:
                    current_url = trajectory[-1]["info"]["page"].url
                except Exception:
                    current_url = ""
                expression, action, value = self._maybe_override_homepage_click(
                    intent,
                    current_url,
                    action,
                    expression,
                    value,
                )

                #     action_set = ["click", "clear", "hover", "type", "scroll", "scroll [down]", "scroll [up]",
                #                  "press", "new_tab", "page_focus", "close_tab", "goto"]

                if action == "scroll down":
                    action = "scroll [down]"
                if action == "scroll up":
                    action = "scroll [up]"

                if action == "scroll" and "up" in value:
                    action = "scroll [up]"
                if action == "scroll" and "down" in value:
                    action = "scroll [down]"

                if action in ["click", "clear", "hover", "type"]:

                    # som_image = trajectory[-1]["observation"]["image"]
                    # id2center = trajectory[-1]["observation"]["id2center"]
                    # text_obs = trajectory[-1]["observation"]["text"]
                    # grounding_prompt = GROUNDING_PROMPT.format(referring_expression=expression,
                    #                                            text_obs=text_obs)
                    # grounding_prompt = ground_prompt_constructor(grounding_prompt, Image.fromarray(som_image), GROUNDING_SYSTEM_PROMPT)
                    # grounding_response = gpt.generate(grounding_prompt)
                    #
                    # matches = re.findall(r'\[(.*?)\]', grounding_response)
                    # print(f"GROUNDING RESPONSE: {grounding_response}, match: {matches}", flush=True)
                    # try:
                    #     action_coordinate = (id2center[matches[0]][0] / trajectory[-1]['info']['viewport_info']['width'],
                    #                          id2center[matches[0]][1] / trajectory[-1]['info']['viewport_info']['height'])
                    #
                    # except:
                    #     print("using uground")
                    #     x0, y0, x1, y1 = self.call_uground(expression, page_screenshot_list[-1])
                    if grounding_method == "uground":
                        action_coordinate = self.call_uground(expression, page_screenshot_list[-1])
                        action_coordinate = (action_coordinate[0] / page_screenshot_list[-1].size[0],
                                             action_coordinate[1] / page_screenshot_list[-1].size[1])
                        print("UGROUND_COORD", action_coordinate)
                    elif grounding_method == "uground_script":
                        action_coordinate = self.call_uground_script(expression, page_screenshot_list[-1])
                        print("UGROUND_SCRIPT_COORD", action_coordinate)
                    elif grounding_method == "uground_vllm":
                        action_coordinate = self.call_uground_vllm(expression, page_screenshot_list[-1])
                        print("UGROUND_VLLM_COORD", action_coordinate)
                    elif grounding_method == "qwen3_vl":
                        action_coordinate = self.call_qwen_grounding(expression, page_screenshot_list[-1])
                        print("OPENAI_GROUND_COORD", action_coordinate)
                    elif grounding_method == "simple":
                        action_coordinate = self.call_simple_grounding(
                            expression,
                            observation,
                            page_screenshot_list[-1],
                        )
                        print("SIMPLE_GROUND_COORD", action_coordinate)
                    elif grounding_method == "atlas":
                        action_coordinate = self.call_atlas(expression, page_screenshot_list[-1])
                    else:
                        action_coordinate = (0.5, 0.5)

                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "ug":
                    action = create_coordinated_action(action, value, action_coordinate)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
                action["raw_prediction"] = response

                # output_dict = process_rationale(response)
                self.output_dict_list.append((response, f"{raw_action} [{expression}] [{value}]"))
                break
            except ActionParsingError as e:
                print("ACTION PARSING ERROR", e, flush=True)
                # Fallback: try to extract a simple action from raw text
                fallback = self._fallback_parse_action(response)
                if fallback is not None:
                    expression, action, value = fallback
                    action = action.lower().strip()
                    if action == "scroll down":
                        action = "scroll [down]"
                    if action == "scroll up":
                        action = "scroll [up]"

                    action_coordinate = None
                    if action in ["click", "clear", "hover", "type"]:
                        if grounding_method == "uground":
                            action_coordinate = self.call_uground(expression, page_screenshot_list[-1])
                            action_coordinate = (action_coordinate[0] / page_screenshot_list[-1].size[0],
                                                 action_coordinate[1] / page_screenshot_list[-1].size[1])
                        elif grounding_method == "uground_script":
                            action_coordinate = self.call_uground_script(expression, page_screenshot_list[-1])
                        elif grounding_method == "uground_vllm":
                            action_coordinate = self.call_uground_vllm(expression, page_screenshot_list[-1])
                        elif grounding_method == "qwen3_vl":
                            action_coordinate = self.call_qwen_grounding(expression, page_screenshot_list[-1])
                        elif grounding_method == "simple":
                            action_coordinate = self.call_simple_grounding(
                                expression,
                                observation,
                                page_screenshot_list[-1],
                            )
                        elif grounding_method == "atlas":
                            action_coordinate = self.call_atlas(expression, page_screenshot_list[-1])

                    if self.action_set_tag == "id_accessibility_tree":
                        action = create_id_based_action(f"{action} [{value}]")
                    elif self.action_set_tag == "playwright":
                        action = create_playwright_action(f"{action} [{value}]")
                    elif self.action_set_tag == "som":
                        action = create_id_based_action(f"{action} [{value}]")
                    elif self.action_set_tag == "ug":
                        action = create_coordinated_action(action, value, action_coordinate)
                    else:
                        raise ValueError(
                            f"Unknown action type {self.action_set_tag}"
                        )
                    action["raw_prediction"] = response
                    self.output_dict_list.append((response, f"{action} [{expression}] [{value}]"))
                    break
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    self.output_dict_list.append("None")
                    break
        return action

    def _fallback_parse_action(self, response_text: str):
        text = response_text or ""
        # Extract a goto URL
        url_match = re.search(r"(https?://[^\s\]\)\"']+)", text)
        if url_match:
            return ("", "goto", url_match.group(1))

        # Extract scroll
        if re.search(r"scroll\s*(down|\[down\])", text, re.IGNORECASE):
            return ("", "scroll [down]", "")
        if re.search(r"scroll\s*(up|\[up\])", text, re.IGNORECASE):
            return ("", "scroll [up]", "")

        # Extract click/type/hover/clear with a description in quotes
        action_match = re.search(r"\b(click|type|hover|clear)\b", text, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).lower()
            quoted = re.findall(r"\"([^\"]+)\"|'([^']+)'", text)
            description = ""
            if quoted:
                description = quoted[0][0] or quoted[0][1]
            return (description, action, "")

        return None

    def reset(self, test_config_file: str) -> None:
        self.output_dict_list = []
        self.memory = []
        self.warning = ""


def construct_agent(args: argparse.Namespace, captioning_fn=None) -> Agent:
    llm_config = lm_config.construct_llm_config(args)

    agent: Agent
    if args.agent_type == "teacher_forcing":
        pass
    elif args.agent_type == "prompt":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        # tokenizer = Tokenizer(args.provider, args.model)
        tokenizer = None
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn,
            grounding_method=getattr(args, "grounding_method", "uground"),
            grounding_model=getattr(args, "grounding_model", None),
            grounding_api_base=getattr(args, "grounding_api_base", None),
            uground_script_path=getattr(args, "uground_script_path", None),
            uground_script_python=getattr(args, "uground_script_python", None),
        )

    else:
        raise NotImplementedError(
            f"agent type {args.agent_type} not implemented"
        )
    return agent


def get_input():
    lines = []
    print('Please Input Action', flush=True)
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return '\n'.join(lines)
