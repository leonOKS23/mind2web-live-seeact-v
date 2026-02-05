import argparse
import base64
import json
import os
import re
from io import BytesIO
from PIL import Image
from openai import OpenAI
from tqdm import tqdm


def pil_to_b64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def parse_coords(text: str):
    if not text:
        return None
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])
    return None


def eval_model(args):
    base_url = args.openai_base or os.environ.get("OPENAI_API_BASE")
    if not base_url:
        raise ValueError("Set --openai-base or OPENAI_API_BASE for the Qwen3 vLLM server.")

    client = OpenAI(api_key="EMPTY", base_url=base_url)
    questions = [json.loads(q) for q in open(args.question_file, "r")]

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    if os.path.exists(args.answers_file):
        os.remove(args.answers_file)

    for question in tqdm(questions, desc="Running grounding"):
        image_base_dir = os.path.expanduser(args.image_folder)
        image_path = os.path.join(image_base_dir, question[args.image_key])
        description = question["description"]

        with Image.open(image_path) as img:
            width, height = img.size
            image_b64 = pil_to_b64(img)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Your task is to return the precise (x, y) coordinate for the described element.\n"
                            "Return ONLY a Python tuple like: (x, y)\n"
                            "Coordinates must be in the 0-1000 range (not normalized).\n"
                            f"Description: {description}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_b64},
                    },
                ],
            }
        ]

        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        generated_text = response.choices[0].message.content.strip()
        coords = parse_coords(generated_text)
        if coords is None:
            continue

        x_ratio, y_ratio = coords
        x_abs = int(x_ratio / 1000 * width)
        y_abs = int(y_ratio / 1000 * height)

        result = dict(question)
        result.update(
            {
                "output": f"({x_abs}, {y_abs})",
                "model_id": args.model,
                "scale": 1.0,
                "raw_output": generated_text,
            }
        )

        with open(args.answers_file, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--openai-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--image-key", type=str, default="img_filename")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)

    args = parser.parse_args()
    eval_model(args)
