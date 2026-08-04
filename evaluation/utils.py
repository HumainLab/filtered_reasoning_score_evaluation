import os
import json
import random
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any

from examples import get_examples


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def lower_keys(example):
    new_example = {}
    for key, value in example.items():
        if key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example


EXAMPLES = get_examples()


def load_prompt(data_name, prompt_type, num_shots):
    if not num_shots:
        return []

    if data_name in ["gsm_hard", "svamp", "tabmwp", "asdiv", "mawps"]:
        data_name = "gsm8k"
    if data_name in ["math_oai", "hungarian_exam", "math-oai", "aime24", "amc23"]:
        data_name = "math"
    if data_name in ["sat_math"]:
        data_name = "mmlu_stem"
    if data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        data_name = "gaokao"

    if prompt_type in ["tool-integrated"]:
        prompt_type = "tora"

    return EXAMPLES[data_name][:num_shots]


PROMPT_TEMPLATES = {
    "direct": ("Question: {input}\nAnswer: ", "{output}", "\n\n"),
    "cot": ("Question: {input}\nAnswer: ", "{output}", "\n\n\n"),
    "auto-cot": (
        "### Question:\n{input}\n\n### Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "pal": ("Question: {input}\n\n", "{output}", "\n---\n"),
    "tool-integrated": ("Question: {input}\n\nSolution:\n", "{output}", "\n---\n"),
    "self-instruct": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "tora": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "wizard_zs": (
        "### Instruction:\n{input}\n\n### Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "platypus_fs": (
        "### Instruction:\n{input}\n\n### Response:\n",
        "{output}",
        "\n\n\n",
    ),
    "deepseek-math": (
        "User: {input}\nPlease reason step by step, "
        "and put your final answer within \\boxed{{}}.\n\nAssistant:",
        "{output}",
        "\n\n\n",
    ),
    "kpmath": (
        "User: Please reason step by step and put your final answer at the end "
        'with "The answer is: ".\n\n{input}\n\nAssistant:',
        "{output}",
    ),
    "jiuzhang": (
        "## Question\n{input}\n\n## Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_tora": (
        "## Question\n{input}\n\n## Code Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_nl": (
        "## Question\n{input}\n\n## Natural Language Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "mmiqc": (
        'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{input}\n\n',
        "{output}",
        "\n\n\n",
    ),
    "abel": (
        "Question:\n{input}\nAnswer:\nLet's think step by step.\n",
        "{output}",
        "\n\n",
    ),
    "shepherd": ("{input}\n", "{output}", "\n\n\n"),
    "qwen-boxed": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen25-math-cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mathstral": (
        "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "{output}",
        "\n\n",
    ),
    "internlm-math-fs": ("Question:{input}\nAnswer:", "{output}", "\n"),
    "internlm-math-chat": (
        "<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mistral": (
        "[INST] {input}[/INST]",
        "{output}",
        "\n\n",
    ),
    "numina": ("### Problem: {input}\n### Solution:", " {output}", "\n\n"),
    "aime": (
        "You are solving an AIME (American Invitational Mathematics Examination) problem.\n\n"
        "These are competition math problems that require multiple steps of reasoning.\n\n"
        "Show all your work carefully, think step by step, and reason in detail.\n\n"
        "Important rules:\n\n"
        "- The final answer is always a non-negative integer from 000 to 999.\n\n"
        "- Format your final answer as: \\boxed{{XYZ}}\n\n"
        "- Only put the final integer in the box.\n\n"
        "Problem:\n\n"
        "{input}\n\n"
        "Now, think step-by-step and solve the problem.",
        "{output}",
        "\n\n\n",
    ),
    "gsm8k": (
        "Solve the following math word problem step by step. Show your reasoning clearly.\n\n"
        "At the end, write only the final answer in the format: \\boxed{{your_answer}}.\n\n\n"
        "Problem: {input}\n\n\n"
        "Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "math500": (
        "Solve the following math problem step by step. Show all intermediate reasoning clearly and write any equations used.\n\n"
        "At the end, write only the final boxed answer using the format: \\boxed{{your_answer}}.\n\n\n"
        "Problem: {input}\n\n\n"
        "Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "aqua": (
        "Solve the following math word problem step by step. Show your reasoning clearly.\n\n"
        "At the end, write only the final answer choice in the format: \\boxed{{A}}, \\boxed{{B}}, etc.\n\n\n"
        "Problem: {input}\n\n\n"
        "Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "svamp": (
        "Solve the following math word problem step by step. Show your reasoning clearly.\n\n"
        "At the end, write only the final answer in the format: \\boxed{{your_answer}}.\n\n\n"
        "Problem: {input}\n\n\n"
        "Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "asdiv": (
        "Solve the following algebra word problem step by step. Show your reasoning clearly and any equations used.\n\n"
        "At the end, write only the final answer in the format: \\boxed{{your_answer}}.\n\n\n"
        "Problem: {input}\n\n\n"
        "Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "humaneval": (
        "You are a Python coding assistant. Read the following function specification and write the complete Python function that satisfies it.\n\n"
        "Follow these rules:\n\n"
        "1. Return only the Python code.\n\n"
        "2. Do not include explanations or comments.\n\n"
        "3. Do not write tests.\n\n"
        "4. Do not include imports unless required.\n\n\n"
        "Problem:\n\n"
        "{input}\n\n\n"
        "Write your final answer as Python code only.",
        "{output}",
        "\n\n\n",
    ),
    "humaneval_reason": (
        "You are an expert Python programmer. Read the function specification below.\n\n"
        "Work in two parts:\n\n"
        "1. **Reasoning**: Explain your approach step by step in plain language before writing code.\n\n"
        "2. **Solution**: Provide the complete function implementation.\n\n"
        "Requirements for the solution code:\n"
        "- Include clear inline comments between logical steps (one comment per major step or line group).\n"
        "- Write only the function **body** (indented lines inside the function); do not repeat the `def` line or imports from the stub.\n"
        "- Do not write tests or `assert` statements.\n"
        "- Put the final code in a markdown block under a header `### Solution Code`:\n\n"
        "### Solution Code\n"
        "```python\n"
        "    # your commented implementation here\n"
        "```\n\n"
        "Problem (function stub):\n\n"
        "{input}\n\n\n"
        "Response: Let's reason through this carefully, then write commented code.",
        "{output}",
        "\n\n\n",
    ),
    "bigbenchhard": (
        "You are a reasoning assistant. Solve the following problem step-by-step.\n\n\n"
        "Problem:\n\n"
        "{input}\n\n\n"
        "Think through the problem carefully. Then give your final answer in the format:\n\n\n"
        "Final Answer: <your_answer>",
        "{output}",
        "\n\n\n",
    ),
    "gpqa": (
        "You are a scientific reasoning model. Solve the problem below step-by-step using clear scientific reasoning.\n\n\n"
        "Problem:\n\n"
        "{input}\n\n\n"
        "Show your reasoning. Then give your answer in the format:\n\n\n"
        "Final Answer: <letter>",
        "{output}",
        "\n\n\n",
    ),
    "arc_challenge": (
        "You are an abstract reasoning assistant. Each ARC puzzle consists of input grids and output grids. Deduce the pattern that transforms the input into the output, then apply that pattern to the test input.\n\n\n"
        "Problem:\n\n"
        "{input}\n\n\n"
        "Explain your reasoning step-by-step. Then provide the final output grid in valid JSON format as the only output after:\n\n\n"
        "Final Answer:\n\n"
        "<json>",
        "{output}",
        "\n\n\n",
    ),
    "commonsense_qa": (
        "You are a commonsense reasoning assistant. Read the question and the answer choices carefully.\n\n"
        "{input}\n\n"
        "Think step-by-step and explain your reasoning. After you finish reasoning, give your final answer in the format:\n"
        "Final Answer: <letter>",
        "{output}",
        "\n\n\n",
    ),
    "gsm8k_fewshot": (
        "Solve the following math word problem step by step. Show your reasoning clearly.\n"
        "At the end, write only the final answer in the format: \\boxed{{your_answer}}.\n\n"
        "Example 1:\n"
        "Problem: Sarah has 5 apples and buys 3 more apples. How many apples does she have in total?\n\n"
        "Response:\n"
        "Sarah starts with 5 apples.\n"
        "She buys 3 more apples.\n"
        "So the total number of apples is 5 + 3 = 8.\n"
        "\\boxed{{8}}\n\n"
        "Example 2:\n"
        "Problem: A box contains 12 pencils. If 4 pencils are taken away, how many pencils remain?\n\n"
        "Response:\n"
        "The box starts with 12 pencils.\n"
        "4 pencils are taken away.\n"
        "So the remaining number of pencils is 12 - 4 = 8.\n"
        "\\boxed{{8}}\n\n"
        "Problem: {input}\n\n"
        "Response:\n"
        "Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "math500_fewshot": (
        "Solve the following math problem step by step. Show all intermediate reasoning clearly and write any equations used.\n"
        "At the end, write only the final boxed answer using the format: \\boxed{{your_answer}}.\n\n"
        "Example 1:\n"
        "Problem: Solve for x: 2x + 3 = 11.\n\n"
        "Response:\n"
        "We start with the equation 2x + 3 = 11.\n"
        "Subtract 3 from both sides to get 2x = 8.\n"
        "Divide both sides by 2 to get x = 4.\n"
        "\\boxed{{4}}\n\n"
        "Example 2:\n"
        "Problem: What is the value of x if x^2 = 25?\n\n"
        "Response:\n"
        "We are given x^2 = 25.\n"
        "Taking the square root of both sides gives x = 5 or x = -5.\n"
        "\\boxed{{{{5, -5}}}}\n\n"
        "Problem: {input}\n\n"
        "Response:\n"
        "Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "aqua_fewshot": (
        "Solve the following math word problem step by step. Show your reasoning clearly.\n"
        "At the end, write only the final answer choice in the format: \\boxed{{A}}, \\boxed{{B}}, etc.\n\n"
        "Example 1:\n"
        "Problem:\n"
        "If x = 2, what is the value of 3x + 1?\n"
        "A. 5\n"
        "B. 6\n"
        "C. 7\n"
        "D. 8\n\n"
        "Response:\n"
        "We substitute x = 2 into the expression 3x + 1.\n"
        "This gives 3(2) + 1 = 6 + 1 = 7.\n"
        "The correct answer choice is C.\n"
        "\\boxed{{C}}\n\n"
        "Example 2:\n"
        "Problem:\n"
        "What is the value of 10 - 4?\n"
        "A. 4\n"
        "B. 5\n"
        "C. 6\n"
        "D. 7\n\n"
        "Response:\n"
        "Subtracting 4 from 10 gives 6.\n"
        "The correct answer choice is C.\n"
        "\\boxed{{C}}\n\n"
        "Problem: {input}\n\n"
        "Response:\n"
        "Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "svamp_fewshot": (
        "Solve the following math word problem step by step. Show your reasoning clearly.\n"
        "At the end, write only the final answer in the format: \\boxed{{your_answer}}.\n\n"
        "Example 1:\n"
        "Problem: Tom has 10 marbles. He gives 3 marbles to his friend. How many marbles does Tom have left?\n\n"
        "Response:\n"
        "Tom starts with 10 marbles.\n"
        "He gives away 3 marbles.\n"
        "So he has 10 - 3 = 7 marbles left.\n"
        "\\boxed{{7}}\n\n"
        "Example 2:\n"
        "Problem: A farmer has 6 baskets with 5 apples in each basket. How many apples does the farmer have?\n\n"
        "Response:\n"
        "There are 6 baskets.\n"
        "Each basket has 5 apples.\n"
        "So the total number of apples is 6 × 5 = 30.\n"
        "\\boxed{{30}}\n\n"
        "Problem: {input}\n\n"
        "Response:\n"
        "Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "commonsense_qa_fewshot": (
        "You are a commonsense reasoning assistant. Read the question and the answer choices carefully.\n\n"
        "Example 1:\n"
        "Question: Where would you most likely find a refrigerator?\n"
        "A. Bathroom\n"
        "B. Kitchen\n"
        "C. Bedroom\n"
        "D. Garage\n\n"
        "Reasoning:\n"
        "A refrigerator is used to store food and keep it cold.\n"
        "Kitchens are designed for storing and preparing food.\n"
        "So the most likely place to find a refrigerator is the kitchen.\n\n"
        "Final Answer: B\n\n"
        "Example 2:\n"
        "Question: What would someone use to write on a chalkboard?\n"
        "A. Pen\n"
        "B. Marker\n"
        "C. Chalk\n"
        "D. Crayon\n\n"
        "Reasoning:\n"
        "A chalkboard is designed to be written on with chalk.\n"
        "Pens and markers are typically used on paper or whiteboards.\n"
        "So the correct answer is chalk.\n\n"
        "Final Answer: C\n\n"
        "{input}\n\n"
        "Think step-by-step and explain your reasoning. After you finish reasoning, give your final answer in the format:\n"
        "Final Answer: <letter>",
        "{output}",
        "\n\n\n",
    ),
    "gpqa_fewshot": (
        "You are a scientific reasoning model. Solve the problem below step-by-step using clear scientific reasoning.\n\n"
        "Example 1:\n"
        "Problem:\n"
        "What happens to the pressure of an ideal gas if its volume decreases while temperature remains constant?\n\n"
        "Choices:\n"
        "A. Pressure decreases\n"
        "B. Pressure remains the same\n"
        "C. Pressure increases\n"
        "D. Pressure becomes zero\n\n"
        "Reasoning:\n"
        "According to Boyle's law, pressure is inversely proportional to volume when temperature is constant.\n"
        "If the volume decreases, the pressure must increase.\n"
        "Therefore, the correct answer is C.\n\n"
        "Final Answer: C\n\n"
        "Example 2:\n"
        "Problem:\n"
        "What quantity is conserved in an isolated system according to the law of conservation of energy?\n\n"
        "Choices:\n"
        "A. Momentum\n"
        "B. Mass\n"
        "C. Energy\n"
        "D. Force\n\n"
        "Reasoning:\n"
        "The law of conservation of energy states that energy cannot be created or destroyed in an isolated system.\n"
        "Therefore, total energy remains constant.\n"
        "The correct answer is C.\n\n"
        "Final Answer: C\n\n"
        "Problem:\n"
        "{input}\n\n"
        "Show your reasoning. Then give your answer in the format:\n\n"
        "Final Answer: <letter>",
        "{output}",
        "\n\n\n",
    ),
}


def construct_prompt(example, data_name, args):
    # Check if custom prompt is provided
    if hasattr(args, 'prompt') and args.prompt and args.prompt.strip():
        # Use custom prompt directly - replace only {question} with the actual question
        # All other curly braces are preserved as literal text (e.g., {your_answer}, \boxed{...})
        question_text = example.get("question", "")
        # Replace {question} with actual question, but preserve all other curly braces
        prompt = args.prompt.strip()
        # Only replace the exact {question} placeholder, nothing else
        return prompt.replace("{question}", question_text)
    
    # Check if this is a few-shot prompt with embedded examples
    fewshot_prompts = ["gsm8k_fewshot", "math500_fewshot", "aqua_fewshot", "svamp_fewshot", 
                       "commonsense_qa_fewshot", "gpqa_fewshot"]
    if args.prompt_type in fewshot_prompts:
        # These prompts already have examples embedded, so just format with the question
        prompt_temp = PROMPT_TEMPLATES[args.prompt_type]
        input_template = prompt_temp[0]
        full_prompt = input_template.format(input=example["question"])
        return full_prompt.strip(" ")
    
    # Otherwise use the original prompt_type logic
    if args.adapt_few_shot and data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        demos = load_prompt(data_name, args.prompt_type, 5)
    else:
        demos = load_prompt(data_name, args.prompt_type, args.num_shots)
    prompt_type = args.prompt_type
    if prompt_type == "platypus_fs":
        prompt_type = "cot"
    if prompt_type == "tool-integrated":
        prompt_type = "tora"

    prompt_temp = PROMPT_TEMPLATES[args.prompt_type]

    splitter = prompt_temp[2]
    input_template, output_template, splitter = (
        prompt_temp[0],
        prompt_temp[1],
        prompt_temp[2],
    )
    if args.prompt_type == "qwen25-math-cot":
        # Hotfix to support putting all demos into a single turn
        demo_prompt = splitter.join([q + "\n" + a for q, a in demos])
    else:
        demo_prompt = splitter.join(
            [
                input_template.format(input=q) + output_template.format(output=a)
                for q, a in demos
            ]
        )
    context = input_template.format(input=example["question"])
    if len(demo_prompt) == 0 or (
        args.adapt_few_shot and example["gt_ans"] not in ["A", "B", "C", "D", "E"]
    ):
        full_prompt = context
    else:
        if args.prompt_type == "qwen25-math-cot":
            # Hotfix to supportting put all demos into a single turn
            full_prompt = demo_prompt + splitter + example["question"]
            full_prompt = input_template.format(input=full_prompt)
        else:
            full_prompt = demo_prompt + splitter + context

    if args.prompt_type == "platypus_fs":
        full_prompt_temp = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        )
        full_prompt = full_prompt_temp.format(instruction=full_prompt)

    if prompt_type == "tora":
        full_prompt = (
            """Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:

- Analyze the question and write functions to solve the problem; the function should not take any arguments.
- Present the final result in LaTeX using a `\boxed{}` without any units.
- Utilize the `pi` symbol and `Rational`` from Sympy for $\pi$ and fractions, and simplify all fractions and square roots without converting them to decimal values.

Here are some examples you may refer to:

---

"""
            + full_prompt
        )

    return full_prompt.strip(" ")  # important!


key_map = {
    "gt": "Ground Truth",
    "pred": "Prediction",
    "gt_cot": "Reference CoT",
    "score": "Score",
}


def show_sample(sample, print_all_preds=False):
    print("==" * 20)
    for key in ["idx", "type", "level", "dataset"]:
        if key in sample:
            # capitalize
            print("{}: {}".format(key[0].upper() + key[1:], sample[key]))
    print("Question:", repr(sample["question"]))
    if "code" in sample:
        if print_all_preds:
            for code in sample["code"]:
                print("-" * 20)
                print("code:", code)
            print("Execution:", sample["report"])
        else:
            print("Solution:\n", sample["code"][0])
            print("Execution:", sample["report"][0])
    if "pred" in sample:
        print("Prediction:", repr(sample["pred"][0]))
    for key in ["gt", "score", "unit", "gt_cot"]:
        if key in sample:
            _key = key_map.get(key, key)
            print("{}: {}".format(_key, repr(sample[key])))
    print()
