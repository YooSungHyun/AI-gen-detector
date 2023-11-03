import json
import os
from simple_parsing import ArgumentParser
import time
from multiprocessing import Pool, cpu_count
import openai
from tqdm import tqdm
from functools import partial


# test commit
def make_inst(i, api_config_path):
    max_retries = 3
    retry_delay = 5
    retries = 0

    with open(api_config_path, "r") as f:
        api_configs = json.load(f)
    openai.api_key = api_configs["api_key"]
    openai.organization = api_configs["organization"]
    engine = api_configs["engine"]

    result_dict = dict()
    while retries < max_retries:
        try:
            prompt = "make some assay plz.\n"
            prompt += "maked assay\n"

            conversation = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=engine,
                messages=conversation,
                stop=None,
                temperature=0.2,
            )

            # Extract numeric questions using regular expression
            text = response.choices[-1].message["content"]
            if text.strip() == "":
                print("아웃풋이 없습니다. 다시 시도")
                time.sleep(retry_delay)
                continue

            result_dict["prompt"] = prompt
            result_dict["text"] = text
            result_dict["generated"] = 1
            break
        except openai.error.OpenAIError as e:
            print(f"OpenAI API error occurred: {e.__class__.__name__}")
            print("Retrying in a few seconds...")
            time.sleep(retry_delay)
            retries += 1
            if retries >= max_retries:
                print("Max retries reached. Skipping this request.")
    return result_dict


def main(args):
    machine_instructions = []
    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info)
        print(f"Loaded {len(machine_instructions)} {args.output_path}")

    n_cpu = cpu_count()
    with Pool(n_cpu - 1) as pool:
        with open(args.output_path, "a") as fout:
            with tqdm(total=args.num_samples) as pbar:
                for _ in range(len(machine_instructions)):
                    pbar.update()

                while len(machine_instructions) < args.num_samples:
                    for temp_inst in tqdm(
                        pool.imap(partial(make_inst, api_config_path=args.api_config_path), range(n_cpu))
                    ):
                        machine_instructions.append(temp_inst)
                        fout.write(json.dumps(temp_inst, ensure_ascii=False) + "\n")
                        pbar.update()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--api_config_path",
        type=str,
        default="./configs/openai_api_config.json",
        help="where is your openai api config json file?",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/outputs.jsonl",
        help="output path plz",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50000,
        help="how many samples do you want?",
    )
    args = parser.parse_args()
    main(args)
