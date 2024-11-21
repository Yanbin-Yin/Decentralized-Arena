import json
import numpy as np
import copy
import tqdm
import csv
from utils_final import existing_model_paths, gt_scores
from multiprocessing import Pool, cpu_count
import fire
import random
import uuid
import scipy.stats
from scipy.optimize import minimize
import pandas as pd
import os
from tqdm import tqdm
from scipy.stats import spearmanr
import itertools
from openai import OpenAI
import google.generativeai as genai
import anthropic
from tokencost import calculate_completion_cost, calculate_prompt_cost
from decimal import Decimal
import sys
import google.generativeai as genai
import time
import re
from judge_responses_new import get_question_with_reference, judge_prompt_pair_reference, judge_prompt_pairwise, \
    fetch_responses, determine_winner, load_records

judge_api_model = ['ChatGPT-4o-latest (2024-09-03)', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-08-06']#,'gemini-1.5-flash-001']  # ]#,"gemini-1.0-pro-001",'gpt3.5-turbo-0125',
judge_open_model = ['athene-70b', 'gemma-1.1-7b-it', 'gemma-2-27b-it', 'gemma-2-9b-it-simpo', 'gemma-1.1-2b-it',
                    'gemma-2b-it', 'yi-1.5-34b-chat', 'mistral-7b-instruct-1',
                    'mistral-8x7b-instruct-v0.1', 'command-r-(04-2024)', 'command-r-(08-2024)', 'qwen1.5-14b-chat',
                    'qwen1.5-32b-chat', 'qwen2-72b-instruct', 'qwen1.5-4b-chat', 'qwen1.5-72b-chat', 'openchat-3.5',
                    'openchat-3.5-0106', 'starling-lm-7b-alpha', 'gemma-2-2b-it', 'google-gemma-2-9b-it',
                    "starling-lm-7b-beta", "llama3-8b-instruct", "meta-llama-3.1-70b-instruct",
                    "meta-llama-3.1-8b-instruct", "llama-3-70b-instruct",
                    'mistral-7b-instruct-2'] # "gemma-7b-it"

judge_model_list = judge_api_model + judge_open_model

overall_ids = [i for i in range(81,161)]
# Algebra
# overall_ids = [296, 48, 292, 323, 129, 342, 133, 301, 37, 270, 385, 363, 356, 112, 285, 66, 26, 149, 16, 30, 55, 163, 97, 31, 339, 126, 263, 396, 103, 92, 161, 188, 111, 257, 282, 125, 393, 398, 122, 293, 236, 219, 341, 154, 261, 23, 381, 348, 99, 24, 195, 250, 255, 335, 326, 260, 325, 305, 86, 229, 197, 391, 153, 361, 265, 46, 215, 226, 190, 123, 117, 81, 231, 284, 281, 25, 207, 128, 331, 93, 316, 277, 57, 362, 354, 223, 47, 165, 307, 18, 157, 303, 74, 45, 104, 189, 210, 294, 248, 134]
# Geometry
# overall_ids = [172, 140, 210, 69, 222, 136, 79, 151, 26, 373, 208, 96, 258, 29, 144, 324, 66, 244, 128, 237, 239, 360,
#                358, 88, 241, 104, 168, 235, 94, 183, 70, 300, 156, 76, 125, 127, 261, 243, 152, 307, 147, 47, 54, 110,
#                339, 80, 181, 395, 196, 95, 179, 116, 264, 12, 178, 223, 218, 377, 41, 101, 114, 219, 174, 327, 30, 213,
#                175, 375, 63, 98, 267, 357, 7, 124, 231, 379, 126, 272, 315, 370, 323, 296, 118, 68, 180, 135, 10, 304,
#                157, 121, 249, 160, 28, 229, 346, 215, 4, 238, 159, 103]
# Probability
# overall_ids = [268, 169, 176, 118, 132, 377, 23, 167, 393, 211, 226, 301, 83, 27, 280, 182, 210, 64, 242, 338, 347, 238, 147, 29, 255, 225, 174, 8, 329, 260, 107, 249, 380, 276, 35, 270, 170, 128, 399, 376, 322, 294, 388, 195, 379, 398, 265, 279, 121, 360, 163, 361, 141, 153, 51, 41, 54, 34, 221, 4, 188, 162, 177, 101, 359, 241, 313, 246, 285, 325, 358, 144, 125, 137, 111, 365, 44, 138, 305, 357, 288, 286, 59, 394, 72, 348, 273, 319, 203, 293, 277, 333, 206, 40, 371, 228, 32, 334, 150, 158]
# Logical
# overall_ids = [492, 200, 205, 60, 197, 236, 223, 220, 210, 93, 79, 209, 487, 224, 383, 229, 238, 398, 44, 494, 227, 328, 47, 237, 283, 354, 216, 300, 374, 340, 270, 287, 222, 189, 486, 369, 239, 489, 208, 285, 213, 304, 56, 217, 180, 120, 158, 476, 244, 100, 127, 360, 274, 225, 235, 280, 196, 313, 408, 448, 187, 362, 207, 462, 295, 162, 305, 282, 119, 78, 23, 215, 202, 416, 198, 370, 434, 261, 95, 199, 311, 191, 292, 479, 201, 410, 273, 323, 483, 243, 99, 399, 28, 123, 440, 214, 490, 350, 182, 252]
# Social
# overall_ids = [80, 457, 0, 246, 426, 394, 97, 312, 488, 405, 268, 122, 257, 434, 215, 141, 89, 85, 57, 379, 459, 341, 8, 315, 418, 251, 388, 496, 182, 332, 431, 365, 22, 358, 339, 12, 136, 433, 289, 129, 207, 112, 361, 491, 235, 310, 410, 91, 278, 474, 72, 438, 445, 222, 87, 412, 302, 36, 54, 391, 367, 345, 415, 401, 212, 259, 192, 77, 389, 30, 178, 460, 336, 346, 273, 38, 465, 5, 347, 162, 27, 458, 486, 451, 258, 356, 284, 297, 402, 173, 262, 468, 393, 413, 333, 76, 42, 291, 183, 198]

# code cpp
# overall_ids = [258, 388, 265, 257, 302, 339, 385, 317, 498, 482, 496, 278, 290, 255, 338, 382, 244, 321, 329, 363, 273, 249, 334, 260, 284, 359, 292, 368, 394, 151, 436, 370, 313, 356, 224, 296, 349, 369, 283, 310, 474, 452, 190, 390, 403, 286, 322, 437, 335, 411, 490, 279, 81, 352, 242, 387, 285, 301, 364, 391, 367, 287, 331, 340, 366, 438, 397, 213, 319, 303, 221, 358, 362, 271, 395, 470, 374, 330, 263, 298, 327, 445, 276, 299, 404, 288, 428, 350, 376, 36, 281, 30, 269, 208, 418, 325, 422, 103, 243, 275]
# code java
# overall_ids = [230, 205, 198, 159, 251, 246, 258, 242, 304, 170, 150, 165, 253, 215, 265, 473, 217, 387, 108, 388, 284, 351, 254, 47, 212, 419, 119, 493, 391, 264, 340, 177, 156, 239, 275, 399, 209, 245, 372, 196, 240, 206, 241, 272, 164, 189, 324, 248, 216, 481, 121, 416, 112, 317, 195, 408, 313, 79, 273, 402, 207, 203, 326, 120, 181, 122, 200, 161, 355, 249, 132, 348, 255, 278, 221, 287, 134, 244, 202, 327, 400, 204, 199, 92, 222, 291, 237, 354, 107, 218, 259, 227, 174, 343, 247, 263, 257, 173, 185, 224]
# code python
# overall_ids = [431, 449, 169, 170, 247, 278, 174, 298, 343, 313, 285, 155, 443, 400, 181, 226, 242, 306, 344, 362, 318, 168, 259, 202, 264, 203, 310, 189, 212, 387, 481, 332, 167, 353, 270, 325, 186, 171, 231, 335, 249, 237, 293, 136, 230, 263, 201, 452, 114, 277, 267, 195, 172, 173, 191, 461, 250, 224, 281, 101, 493, 185, 334, 300, 219, 215, 194, 217, 110, 183, 322, 192, 495, 280, 157, 364, 448, 339, 241, 311, 225, 324, 198, 128, 182, 74, 235, 213, 323, 482, 208, 187, 389, 229, 107, 108, 193, 262, 314, 294]

# science chemistry 
# overall_ids = [453, 145, 96, 9, 377, 183, 40, 436, 118, 441, 85, 159, 390, 401, 228, 380, 76, 141, 425, 351, 189, 55, 13, 239, 150, 37, 498, 142, 479, 124, 391, 421, 359, 149, 308, 100, 35, 157, 64, 405, 438, 26, 423, 230, 223, 18, 243, 104, 300, 310, 28, 170, 251, 176, 32, 332, 174, 255, 7, 3, 215, 162, 420, 109, 186, 119, 43, 214, 130, 52, 370, 84, 381, 8, 173, 73, 133, 156, 106, 59, 434, 432, 246, 79, 42, 114, 382, 38, 158, 191, 488, 409, 1, 22, 477, 431, 71, 62, 143, 225]
# science biology
# overall_ids = [376, 286, 347, 440, 145, 464, 16, 95, 362, 487, 402, 165, 46, 71, 267, 171, 91, 169, 239, 409, 137, 350, 358, 316, 255, 300, 144, 127, 240, 206, 22, 401, 133, 102, 490, 471, 280, 434, 58, 277, 56, 472, 306, 104, 265, 379, 15, 408, 69, 353, 81, 290, 213, 325, 312, 289, 254, 178, 450, 368, 332, 210, 187, 130, 86, 77, 357, 53, 411, 194, 142, 27, 149, 297, 79, 50, 134, 264, 320, 459, 496, 340, 88, 361, 313, 63, 241, 26, 13, 25, 273, 191, 183, 159, 469, 360, 139, 47, 429, 470]
overall_ids = sorted(overall_ids)
# Define the file path where you want to save the output
save_output_file_path = 'overall small model.txt'


def rank_scores(scores):
    indexed_scores = list(enumerate(scores))
    sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    ranks = [0] * len(scores)
    for rank, (index, _) in enumerate(sorted_scores):
        ranks[index] = rank
    return ranks


def save_to_jsonl(data, filename):
    """Saves a Python data structure to a .jsonl file."""
    with open(filename, 'w') as f:
        f.write(json.dumps(data) + '\n')


def update_voting_records(model, response_A_name, response_B_name, won, question_id, data_id):
    """Updates the voting records with a new voting result."""
    # 需要修改path 
    records_path = f"/data/shared/decentralized_arena/judgements_mt_bench_kun/judgements_mt_bench/{model}/voting_records.jsonl"
    # print(records_path)
    # Ensure the directory exists
    os.makedirs(os.path.dirname(records_path), exist_ok=True)

    # Load existing records or create an empty list if the file does not exist
    try:
        records = load_records(records_path)[0]
    except:
        records = []

    # Append a new record to the list of records
    new_record = {
        "response_A": response_A_name,
        "response_B": response_B_name,
        "Won": won,
        "question_id": question_id,
        "data_id": data_id
    }
    records.append(new_record)  # Ensure this is a flat append operation

    # Save updated records back to the JSONL file
    save_to_jsonl(records, records_path)


def run_judging_trials(judge_model, model_name, path="/data/shared/decentralized_arena/mt_bench_questions.jsonl",
                       tensor_parallel_size=1):
    # print(judge_model,model_name)
    model_index_map = {name: idx for idx, name in enumerate(model_name)}
    initial_question_ids = overall_ids
    responses_dict = dict()
    # Fetch responses for each model
    for model in model_name:
        responses_dict[model] = fetch_responses("/data/shared/decentralized_arena/mt_bench_responses", model)
    # print(responses_dict)
    combination_models = list(itertools.combinations(model_name, 2))

    # Iterate over combinations of model pairs for comparison
    for model_a, model_b in tqdm(combination_models):
        responses_a = responses_dict[model_a]
        responses_b = responses_dict[model_b]

        batch_size = 80  # Set batch size for processing
        num_batches = (len(initial_question_ids) + batch_size - 1) // batch_size  # Calculate the number of batches

        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(initial_question_ids))
            prompts = list()
            swapped_prompts = list()
            question_ids = list()

            # Create prompts and swapped prompts for comparison
            for idx in range(start_idx, end_idx):
                question_id = initial_question_ids[idx]
                # # algebra
                # if model_a == "o1-preview" or model_b == "o1-preview":
                #     if question_id in [81,316,323,325,326,331,335,341,354,362,381,385,393]:
                #         continue
                # if model_a == "o1-mini" or model_b == "o1-mini":
                #     if question_id in [316,323,326,342,362,381,385]:
                #         continue
                # # probability
                # if model_a == "o1-preview" or model_b == "o1-preview":
                #     if question_id in [265,380,398,393]:
                #         continue
                # # Logical
                # if model_a == "o1-preview" or model_b == "o1-preview":
                #     if question_id in [261,354,220,47]:
                #         continue

                # print(question_id)
                question, reference = get_question_with_reference(path, question_id)
                response_a = responses_a[idx]['response']
                response_b = responses_b[idx]['response']
                # print(question,reference)
                # print(response_a,response_b)
                # breakpoint()
                if reference != "":
                    prompt = judge_prompt_pair_reference(question, response_a, response_b, reference)
                    swapped_prompt = judge_prompt_pair_reference(question, response_b, response_a, reference)
                else:
                    prompt = judge_prompt_pairwise(question, response_a, response_b)
                    swapped_prompt = judge_prompt_pairwise(question, response_b, response_a)
                    # print(prompt)
                # breakpoint()
                prompts.append(prompt)
                swapped_prompts.append(swapped_prompt)
                question_ids.append(question_id)
            try:
                # Adjust logic based on the type of judge_model
                if 'gpt' in judge_model or 'GPT' in judge_model:  # For OpenAI models
                    judge_responses = run_openai_model(prompts, judge_model)
                    swapped_judge_responses = run_openai_model(swapped_prompts, judge_model)
                elif "gemini" in judge_model:  # For Gemini models
                    judge_responses = run_gemini_model(prompts, judge_model)
                    swapped_judge_responses = run_gemini_model(swapped_prompts, judge_model)
                # print(judge_responses)
                # print(swapped_judge_responses)
            except Exception as e:
                print(f"Error evaluating model pair ({model_a}, {model_b}) with judge {judge_model}: {e}")
                continue  # Skip to the next model pair if there's an error

            cnt = 0
            # Process responses and determine winners
            for response, swapped_response in zip(judge_responses, swapped_judge_responses):
                winner = determine_winner(response, model_a, model_b)
                swapped_winner = determine_winner(swapped_response, model_b, model_a)
                final_winner = winner if winner == swapped_winner else "TIE"
                data_id = str(uuid.uuid4())
                update_voting_records(judge_model, model_a, model_b, final_winner, question_ids[cnt], data_id)
                cnt += 1


def run_openai_model(prompts, model_name, temperature=0.7, max_tokens=15):
    # Handle model selection for OpenAI models
    if "3.5-turbo-0125" in model_name:
        model_name = "gpt-3.5-turbo-0125"
        client = OpenAI(api_key="")
        
    elif "gpt-4o-mini" in model_name:
        model_name = "gpt-4o-mini-2024-07-18"
        client = OpenAI(api_key="")

    elif "ChatGPT-4o-latest" in model_name:
        model_name = "chatgpt-4o-latest"
        client = OpenAI(api_key="")

    elif "gpt-4o-2024-08-06" in model_name:
        model_name = "gpt-4o-2024-08-06"
        client = OpenAI(api_key="")


    responses = []
    # Modify each prompt to ask the model to evaluate dataset quality
    for prompt in prompts:
        # Call OpenAI API with the modified quality evaluation prompt
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Extract and store the response
        text = completion.choices[0].message.content
        responses.append(str(text))

    return responses


def run_gemini_model(prompts, model_name="gemini-1.5-flash", max_tokens=3):
    if model_name == "gemini-1.5-pro-001":
        model_name = "gemini-1.5-pro-001"
        client = ''
    elif model_name == "gemini-1.0-pro-001":
        model_name = "gemini-1.0-pro-001"
        client = ""
    elif model_name == "gemini-1.5-flash-001":
        model_name = "gemini-1.5-flash-001"
        client = ""

    responses = []
    genai.configure(api_key=client)
    model = genai.GenerativeModel(model_name)
    for prompt in prompts:
        cnt = 0
        while 1:
            cnt += 1
            if cnt > 5:
                responses.append("")
                break
            try:
                message = model.generate_content(
                    prompt
                )
                response_text = message.text
                responses.append(response_text)
                break
            except Exception as e:
                print(f"Error : {e}")
                time.sleep(5)
                continue

    return responses


def rugged_rank(base_dir, new_model, base_model_list, base_model_ranking, model_weights, judge_model_list, judge_model_states, valid_question_ids=overall_ids):
    rank_list = list()
    weight_list = list()
    total_weight = 0
    weighted_rank_sum = 0
    judge_models = [model for model in base_model_list if model in judge_model_list]
    # print(judge_models)

    paras = list()
    for judge_model in judge_models:
        models_to_sort = [model for model in base_model_list if model != judge_model]
        paras.append(
            [base_dir, judge_model, new_model, models_to_sort, valid_question_ids, base_model_ranking, model_weights])
    with Pool(processes=(cpu_count() - 1)) as pool:
        result = pool.starmap(binary_search, paras)
    # print(result)
    ranking = list()
    for i in range(len(result)):
        ranking.append((result[i][0],result[i][2]))
        weighted_rank_sum += result[i][0] * result[i][1]
        total_weight += result[i][1]
        judge_model_states[result[i][2]] += result[i][3]
    print(ranking)

    weighted_average_rank = weighted_rank_sum / total_weight
    return weighted_average_rank,judge_model_states


def binary_search(base_dir, judge_model, new_model, models_to_sort, valid_question_ids, base_model_ranking,
                  model_weights):
    model_pair_list = list()
    left, right = 0, len(models_to_sort)
    # print(models_to_sort)
    while left < right:
        mid = (left + right) // 2
        model_pair_list.append((new_model, models_to_sort[mid]))
        if get_vote_result_for_judge(base_dir, judge_model, new_model, models_to_sort[mid], valid_question_ids) <= 0:
            left = mid + 1
        else:
            right = mid
    if left == 0:
        rank = 1
    else:
        rank = base_model_ranking[models_to_sort[left - 1]] + 1
    # print(left, rank)
    if rank == base_model_ranking[judge_model]:
        # vote_diff = 0
        # for model in judge_models:
        #     if model == judge_model:
        #         continue
        #     vote_diff += get_vote_result_for_judge(base_dir, model, new_model, judge_model,valid_question_ids)
        # if vote_diff > 0:
        #     rank = base_model_ranking[judge_model]
        # else:
        rank = base_model_ranking[judge_model] + 1
    weight = model_weights.get(judge_model, 0.3)
    return rank, weight, judge_model, model_pair_list


def get_vote_result_for_judge(base_dir, judge_model, model1, model2, valid_question_ids=overall_ids):
    vote_diff = 0
    jsonl_path = os.path.join(base_dir, judge_model, "voting_records.jsonl")
    # print(judge_model, model1, model2)
    # if model1 == "o1-preview" or model2 == "o1-preview":
    #     to_remove = [265, 380, 398, 393]
    #     valid_question_ids = [item for item in valid_question_ids if item not in to_remove]
    if model1 == "o1-preview" or model2 == "o1-preview":
        to_remove = [148,151,153,154,160]
        valid_question_ids = [item for item in valid_question_ids if item not in to_remove]
    if judge_model in judge_open_model:
        jsonl_path = os.path.join(base_dir, judge_model, "voting_records.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as file:
                for line in file:
                    # print(jsonl_path)
                    record = json.loads(line)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue
                        if each['response_A'] == model1 and each['response_B'] == model2:
                            # print(each)
                            if each['Won'] == model1:
                                vote_diff += 1
                            elif each['Won'] == model2:
                                vote_diff -= 1
                        elif each['response_A'] == model2 and each['response_B'] == model1:
                            # print(each)
                            if each['Won'] == model2:
                                vote_diff -= 1
                            elif each['Won'] == model1:
                                vote_diff += 1
        jsonl_path = os.path.join(base_dir, judge_model, "voting_records_1.jsonl")
        if os.path.exists(jsonl_path):
            # print(jsonl_path)
            with open(jsonl_path, 'r') as file:
                for line in file:
                    # print(jsonl_path)
                    # print(line)
                    record = json.loads(line)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue
                        if each['response_A'] == model1 and each['response_B'] == model2:
                            # print(each)
                            if each['Won'] == model1:
                                vote_diff += 1
                            elif each['Won'] == model2:
                                vote_diff -= 1
                        elif each['response_A'] == model2 and each['response_B'] == model1:
                            # print(each)
                            if each['Won'] == model2:
                                vote_diff -= 1
                            elif each['Won'] == model1:
                                vote_diff += 1
        jsonl_path = os.path.join(base_dir, judge_model, "voting_records_2.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as file:
                for line in file:
                    # print(jsonl_path)
                    record = json.loads(line)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue
                        if each['response_A'] == model1 and each['response_B'] == model2:
                            # print(each)
                            if each['Won'] == model1:
                                vote_diff += 1
                            elif each['Won'] == model2:
                                vote_diff -= 1
                        elif each['response_A'] == model2 and each['response_B'] == model1:
                            # print(each)
                            if each['Won'] == model2:
                                vote_diff -= 1
                            elif each['Won'] == model1:
                                vote_diff += 1
        jsonl_path = os.path.join(base_dir, judge_model, "voting_records_3.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as file:
                for line in file:
                    # print(jsonl_path)
                    record = json.loads(line)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue
                        if each['response_A'] == model1 and each['response_B'] == model2:
                            # print(each)
                            if each['Won'] == model1:
                                vote_diff += 1
                            elif each['Won'] == model2:
                                vote_diff -= 1
                        elif each['response_A'] == model2 and each['response_B'] == model1:
                            # print(each)
                            if each['Won'] == model2:
                                vote_diff -= 1
                            elif each['Won'] == model1:
                                vote_diff += 1
        jsonl_path = os.path.join(base_dir, judge_model, "voting_records_4.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as file:
                for line in file:
                    # print(jsonl_path)
                    record = json.loads(line)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue
                        if each['response_A'] == model1 and each['response_B'] == model2:
                            # print(each)
                            if each['Won'] == model1:
                                vote_diff += 1
                            elif each['Won'] == model2:
                                vote_diff -= 1
                        elif each['response_A'] == model2 and each['response_B'] == model1:
                            # print(each)
                            if each['Won'] == model2:
                                vote_diff -= 1
                            elif each['Won'] == model1:
                                vote_diff += 1
    elif judge_model in judge_api_model:
        # print(judge_model)
        if not os.path.exists(jsonl_path):
            directory = os.path.join(base_dir, judge_model)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if not os.path.exists(jsonl_path):
                with open(jsonl_path, 'w') as f:
                    pass
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as file:
                flag = False
                for line in file:
                    # print(jsonl_path)
                    record = json.loads(line)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue
                        if each['response_A'] == model1 and each['response_B'] == model2:
                            flag = True
                            # print(each)
                            if each['Won'] == model1:
                                vote_diff += 1
                            elif each['Won'] == model2:
                                vote_diff -= 1
                        elif each['response_A'] == model2 and each['response_B'] == model1:
                            flag = True
                            # print(each)
                            if each['Won'] == model2:
                                vote_diff -= 1
                            elif each['Won'] == model1:
                                vote_diff += 1
            # print(vote_diff)
            # 调用API
            if flag == False:
                with open(jsonl_path, 'r') as file:
                    # print(judge_model)
                    if 'gpt' in judge_model or "GPT" in judge_model:
                        # print(judge_model)
                        run_judging_trials(judge_model, [model1, model2])
                        for line in file:
                            # print(jsonl_path)
                            record = json.loads(line)
                            for each in record:
                                if valid_question_ids and each.get('question_id') not in valid_question_ids:
                                    continue
                                if each['response_A'] == model1 and each['response_B'] == model2:
                                    flag = True
                                    # print(each)
                                    if each['Won'] == model1:
                                        vote_diff += 1
                                    elif each['Won'] == model2:
                                        vote_diff -= 1
                                elif each['response_A'] == model2 and each['response_B'] == model1:
                                    flag = True
                                    # print(each)
                                    if each['Won'] == model2:
                                        vote_diff -= 1
                                    elif each['Won'] == model1:
                                        vote_diff += 1
                        print(vote_diff)
                    elif 'gemini' in judge_model:
                        run_judging_trials(judge_model, [model1, model2])
                        for line in file:
                            # print(jsonl_path)
                            record = json.loads(line)
                            for each in record:
                                if valid_question_ids and each.get('question_id') not in valid_question_ids:
                                    continue
                                if each['response_A'] == model1 and each['response_B'] == model2:
                                    flag = True
                                    # print(each)
                                    if each['Won'] == model1:
                                        vote_diff += 1
                                    elif each['Won'] == model2:
                                        vote_diff -= 1
                                elif each['response_A'] == model2 and each['response_B'] == model1:
                                    flag = True
                                    # print(each)
                                    if each['Won'] == model2:
                                        vote_diff -= 1
                                    elif each['Won'] == model1:
                                        vote_diff += 1
        print(judge_model)

    return vote_diff


def integrate_rankings(original_ranking, new_ranking, relative_ranking):
    # print(new_ranking.keys())
    if len(new_ranking) == 1:
        final_ranking = original_ranking.copy()
        final_ranking[next(iter(new_ranking))] = len(final_ranking)
        return final_ranking
    # 找到new_ranking中对应模型在original_ranking中的最小排名
    min_new_rank = min(original_ranking[model] for model in new_ranking)

    # 创建一个字典来存储最终的整体排名，先复制原始排名
    final_ranking = original_ranking.copy()

    max_ranking = max(relative_ranking) + min_new_rank
    for model, rank in final_ranking.items():
        if rank >= max_ranking:
            final_ranking[model] += 1

    # 遍历new_ranking中的每个模型
    for model, idx in new_ranking.items():
        # 获取该模型在relative_ranking中的排名
        new_relative_rank = relative_ranking[idx]
        # 更新final_ranking中的排名，新排名加上new_ranking中的最小排名减一
        final_ranking[model] = new_relative_rank + min_new_rank

    return final_ranking


def full_comparsion(base_dir, new_model, base_model_list, sort_rank, model_weights, judge_model_list, judge_model_states, window=1,
                    valid_question_ids=overall_ids):
    rank_idx = int(sort_rank)
    min_rank_idx = max(1, rank_idx - window + 1)
    max_rank_idx = min(len(base_model_list), rank_idx + window)
    model_names = list()
    # print(min_rank_idx,max_rank_idx)
    for i in range(min_rank_idx - 1, max_rank_idx):
        model_names.append(base_model_list[i])
    model_names.append(new_model)
    print(model_names)
    combinations = list(itertools.combinations(model_names, 2))
    remaining_combinations = set(combinations)
    base_model_list.append(new_model)
    # judge_models = [model for model in base_model_list if model in judge_model_list]
    judge_models = [model for model in base_model_list if model in judge_model_list and model not in model_names]

    for i in judge_models:
        judge_model_states[i] += combinations
    # print(judge_models)
    sort_model_index_map = {name: idx for idx, name in enumerate(model_names)}
    judge_model_index_map = {name: idx for idx, name in enumerate(judge_models)}
    # Initialize an empty comparison matrix
    print(judge_model_index_map)
    final_comparison_matrix = np.zeros((len(judge_models), len(judge_models)))
    weights = list()

    paras = list()
    for subdir in os.listdir(base_dir):
        # print(subdir)
        if subdir not in base_model_list:
            continue
        if subdir not in judge_models:
            continue
        if subdir not in judge_model_list:
            continue
        comparison_matrix = np.zeros((len(judge_models), len(judge_models)))
        paras.append([base_dir, subdir, model_weights, sort_model_index_map, judge_model_index_map, comparison_matrix,
                      remaining_combinations, valid_question_ids])
    with Pool(processes=(cpu_count() - 1)) as pool:
        result = pool.starmap(pairwise_judge, paras)
    # print(result)
    # print(comparison_matrix)
    for i in range(len(result)):
        final_comparison_matrix += result[i][0]
        weights.append(result[i][1])
    print(final_comparison_matrix)
    # 计算每一行的和
    row_sums = final_comparison_matrix.sum(axis=1, keepdims=True)

    # 创建归一化矩阵，初始化为原始矩阵
    normalized_matrix = np.copy(final_comparison_matrix)

    # 对每一行进行归一化（忽略和为0的行）
    nonzero_row_indices = row_sums.flatten() != 0
    normalized_matrix[nonzero_row_indices] = final_comparison_matrix[nonzero_row_indices] / row_sums[
        nonzero_row_indices]
    return sort_model_index_map, min_rank_idx, normalized_matrix, weights, model_names, judge_model_states


def bubble_window(base_dir, new_model, base_model_list, new_model_rank, model_weights, judge_model_list,judge_model_states,window=1,
                  valid_question_ids=overall_ids):
    # print(new_model_rank,base_model_list)
    model_names = list()
    model_names.append(base_model_list[new_model_rank - 2])
    model_names.append(base_model_list[new_model_rank])
    model_names.append(base_model_list[new_model_rank - 1])
    combinations = list(itertools.combinations(model_names, 2))
    remaining_combinations = set(combinations)
    # print(model_names)
    # judge_models = [model for model in base_model_list if model in judge_model_list]
    judge_models = [model for model in base_model_list if model in judge_model_list and model not in model_names]
    for i in judge_models:
        judge_model_states[i] += combinations
    sort_model_index_map = {name: idx for idx, name in enumerate(model_names)}
    judge_model_index_map = {name: idx for idx, name in enumerate(judge_models)}
    print(judge_model_index_map)
    # Initialize an empty comparison matrix
    final_comparison_matrix = np.zeros((len(judge_models), len(judge_models)))
    weights = list()

    paras = list()
    for subdir in os.listdir(base_dir):
        # print(subdir)
        if subdir not in base_model_list:
            continue
        if subdir not in judge_models:
            continue
        if subdir not in judge_model_list:
            continue
        comparison_matrix = np.zeros((len(judge_models), len(judge_models)))
        paras.append([base_dir, subdir, model_weights, sort_model_index_map, judge_model_index_map, comparison_matrix,
                      remaining_combinations, valid_question_ids])
    with Pool(processes=(cpu_count() - 1)) as pool:
        result = pool.starmap(pairwise_judge, paras)
    # print(result)
    # print(comparison_matrix)
    for i in range(len(result)):
        final_comparison_matrix += result[i][0]
        weights.append(result[i][1])
    print(final_comparison_matrix)
    # 计算每一行的和
    row_sums = final_comparison_matrix.sum(axis=1, keepdims=True)

    # 创建归一化矩阵，初始化为原始矩阵
    normalized_matrix = np.copy(final_comparison_matrix)

    # 对每一行进行归一化（忽略和为0的行）
    nonzero_row_indices = row_sums.flatten() != 0
    normalized_matrix[nonzero_row_indices] = final_comparison_matrix[nonzero_row_indices] / row_sums[
        nonzero_row_indices]
    # print(comparison_matrix)
    return model_names, sort_model_index_map, normalized_matrix, weights, model_names, judge_model_states


def pairwise_judge(base_dir, subdir, model_weights, sort_model_index_map, judge_model_index_map, comparison_matrix,
                   remaining_combinations, valid_question_ids):
    if subdir in judge_open_model:
        if sort_model_index_map.get('o1-preview') is not None:
            to_remove = [148,151,153,154,160]
            valid_question_ids = [item for item in valid_question_ids if item not in to_remove]
        jsonl_path = os.path.join(base_dir, subdir, "voting_records.jsonl")
        # print(jsonl_path)
        if os.path.exists(jsonl_path):
            weight = model_weights.get(subdir, 0.3)
            with open(jsonl_path, 'r') as file:
                for line in file:
                    record = json.loads(line, strict=False)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue

                        model1 = each['response_A']
                        model2 = each['response_B']
                        winner = each['Won']
                        if model1 == 'yi-lightning' and model2 == 'nemotron-70b':
                            continue
                        if model1 == 'glm-4-plus' and model2 == 'nemotron-70b':
                            continue  
                        if model1 == 'glm-4-plus' and model2 == 'yi-lightning':
                            continue                                 
                        if model1 == model2:
                            continue
                        # if model1 == "o1-preview" or model2 == "o1-preview":
                        #     if each.get('question_id') in [265, 380, 398, 393]:
                        #         continue

                        idx1 = sort_model_index_map.get(model1)
                        idx2 = sort_model_index_map.get(model2)
                        judge_idx = judge_model_index_map.get(subdir)

                        if idx1 is not None and idx2 is not None and judge_idx is not None:
                            if winner == model1:
                                comparison_matrix[judge_idx, idx1] += 1
                            elif winner == model2:
                                comparison_matrix[judge_idx, idx2] += 1
                            else:
                                comparison_matrix[judge_idx, idx1] += 0
                                comparison_matrix[judge_idx, idx2] += 0
        # print(subdir)
        jsonl_path = os.path.join(base_dir, subdir, "voting_records_1.jsonl")
        # print(jsonl_path)
        if os.path.exists(jsonl_path):
            weight = model_weights.get(subdir, 0.3)
            with open(jsonl_path, 'r') as file:
                for line in file:
                    record = json.loads(line, strict=False)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue

                        model1 = each['response_A']
                        model2 = each['response_B']
                        winner = each['Won']
                        if model1 == 'yi-lightning' and model2 == 'nemotron-70b':
                            continue
                        if model1 == 'glm-4-plus' and model2 == 'nemotron-70b':
                            continue  
                        if model1 == 'glm-4-plus' and model2 == 'yi-lightning':
                            continue                                 
                        if model1 == model2:
                            continue
                        # if model1 == "o1-preview" or model2 == "o1-preview":
                        #     if each.get('question_id') in [265, 380, 398, 393]:
                        #         continue

                        idx1 = sort_model_index_map.get(model1)
                        idx2 = sort_model_index_map.get(model2)
                        judge_idx = judge_model_index_map.get(subdir)

                        if idx1 is not None and idx2 is not None and judge_idx is not None:
                            if winner == model1:
                                comparison_matrix[judge_idx, idx1] += 1
                            elif winner == model2:
                                comparison_matrix[judge_idx, idx2] += 1
                            else:
                                comparison_matrix[judge_idx, idx1] += 0
                                comparison_matrix[judge_idx, idx2] += 0
        jsonl_path = os.path.join(base_dir, subdir, "voting_records_2.jsonl")
        # print(jsonl_path)
        if os.path.exists(jsonl_path):
            weight = model_weights.get(subdir, 0.3)
            with open(jsonl_path, 'r') as file:
                for line in file:
                    record = json.loads(line, strict=False)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue

                        model1 = each['response_A']
                        model2 = each['response_B']
                        winner = each['Won']
                        if model1 == 'yi-lightning' and model2 == 'nemotron-70b':
                            continue
                        if model1 == 'glm-4-plus' and model2 == 'nemotron-70b':
                            continue  
                        if model1 == 'glm-4-plus' and model2 == 'yi-lightning':
                            continue                                 
                        if model1 == model2:
                            continue
                        # if model1 == "o1-preview" or model2 == "o1-preview":
                        #     if each.get('question_id') in [265, 380, 398, 393]:
                        #         continue

                        idx1 = sort_model_index_map.get(model1)
                        idx2 = sort_model_index_map.get(model2)
                        judge_idx = judge_model_index_map.get(subdir)

                        if idx1 is not None and idx2 is not None and judge_idx is not None:
                            if winner == model1:
                                comparison_matrix[judge_idx, idx1] += 1
                            elif winner == model2:
                                comparison_matrix[judge_idx, idx2] += 1
                            else:
                                comparison_matrix[judge_idx, idx1] += 0
                                comparison_matrix[judge_idx, idx2] += 0
        jsonl_path = os.path.join(base_dir, subdir, "voting_records_3.jsonl")
        # print(jsonl_path)
        if os.path.exists(jsonl_path):
            weight = model_weights.get(subdir, 0.3)
            with open(jsonl_path, 'r') as file:
                for line in file:
                    record = json.loads(line, strict=False)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue

                        model1 = each['response_A']
                        model2 = each['response_B']
                        winner = each['Won']

                        # if model1 == "o1-preview" or model2 == "o1-preview":
                        #     if each.get('question_id') in [265, 380, 398, 393]:
                        #         continue

                        idx1 = sort_model_index_map.get(model1)
                        idx2 = sort_model_index_map.get(model2)
                        judge_idx = judge_model_index_map.get(subdir)

                        if idx1 is not None and idx2 is not None and judge_idx is not None:
                            if winner == model1:
                                comparison_matrix[judge_idx, idx1] += 1
                            elif winner == model2:
                                comparison_matrix[judge_idx, idx2] += 1
                            else:
                                comparison_matrix[judge_idx, idx1] += 0
                                comparison_matrix[judge_idx, idx2] += 0
        jsonl_path = os.path.join(base_dir, subdir, "voting_records_4.jsonl")
        # print(jsonl_path)
        if os.path.exists(jsonl_path):
            weight = model_weights.get(subdir, 0.3)
            with open(jsonl_path, 'r') as file:
                for line in file:
                    record = json.loads(line, strict=False)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue

                        model1 = each['response_A']
                        model2 = each['response_B']
                        winner = each['Won']

                        # if model1 == "o1-preview" or model2 == "o1-preview":
                        #     if each.get('question_id') in [265, 380, 398, 393]:
                        #         continue

                        idx1 = sort_model_index_map.get(model1)
                        idx2 = sort_model_index_map.get(model2)
                        judge_idx = judge_model_index_map.get(subdir)

                        if idx1 is not None and idx2 is not None and judge_idx is not None:
                            if winner == model1:
                                comparison_matrix[judge_idx, idx1] += 1
                            elif winner == model2:
                                comparison_matrix[judge_idx, idx2] += 1
                            else:
                                comparison_matrix[judge_idx, idx1] += 0
                                comparison_matrix[judge_idx, idx2] += 0
    elif subdir in judge_api_model or subdir == 'gemini-1.5-flash-001':
        # print(remaining_combinations)
        jsonl_path = os.path.join(base_dir, subdir, "voting_records.jsonl")
        if os.path.exists(jsonl_path):
            weight = model_weights.get(subdir, 0.3)
            with open(jsonl_path, 'r') as file:
                flag = False
                for line in file:
                    record = json.loads(line, strict=False)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue

                        model1 = each['response_A']
                        model2 = each['response_B']
                        winner = each['Won']

                        idx1 = sort_model_index_map.get(model1)
                        idx2 = sort_model_index_map.get(model2)
                        judge_idx = judge_model_index_map.get(subdir)

                        if idx1 is not None and idx2 is not None and judge_idx is not None:
                            remaining_combinations.discard((model1, model2))
                            remaining_combinations.discard((model2, model1))

                            if winner == model1:
                                comparison_matrix[judge_idx, idx1] += 1
                            elif winner == model2:
                                comparison_matrix[judge_idx, idx2] += 1
                            else:
                                comparison_matrix[judge_idx, idx1] += 0
                                comparison_matrix[judge_idx, idx2] += 0
                # print(subdir,remaining_combinations)
                # 调用API
            with open(jsonl_path, 'r') as file:
                if len(remaining_combinations) != 0:
                    comparison_matrix = np.zeros_like(comparison_matrix)
                    judge_model = subdir
                    if 'gpt' in judge_model or 'GPT' in judge_model:
                        for item in remaining_combinations:
                            run_judging_trials(judge_model, [item[0], item[1]])
                        for line in file:
                            record = json.loads(line, strict=False)
                            for each in record:
                                if valid_question_ids and each.get('question_id') not in valid_question_ids:
                                    continue

                                model1 = each['response_A']
                                model2 = each['response_B']
                                winner = each['Won']

                                idx1 = sort_model_index_map.get(model1)
                                idx2 = sort_model_index_map.get(model2)
                                judge_idx = judge_model_index_map.get(subdir)

                                if idx1 is not None and idx2 is not None and judge_idx is not None:
                                    flag = True
                                    if winner == model1:
                                        comparison_matrix[judge_idx, idx1] += 1 * weight
                                    elif winner == model2:
                                        comparison_matrix[judge_idx, idx2] += 1 * weight
                                    else:
                                        comparison_matrix[judge_idx, idx1] += 0 * weight
                                        comparison_matrix[judge_idx, idx2] += 0 * weight

                    elif 'gemini' in judge_model:
                        for item in remaining_combinations:
                            run_judging_trials(judge_model, [item[0], item[1]])
                        for line in file:
                            record = json.loads(line, strict=False)
                            for each in record:
                                if valid_question_ids and each.get('question_id') not in valid_question_ids:
                                    continue

                                model1 = each['response_A']
                                model2 = each['response_B']
                                winner = each['Won']

                                idx1 = sort_model_index_map.get(model1)
                                idx2 = sort_model_index_map.get(model2)
                                judge_idx = judge_model_index_map.get(subdir)

                                if idx1 is not None and idx2 is not None and judge_idx is not None:
                                    flag = True
                                    if winner == model1:
                                        comparison_matrix[judge_idx, idx1] += 1 * weight
                                    elif winner == model2:
                                        comparison_matrix[judge_idx, idx2] += 1 * weight
                                    else:
                                        comparison_matrix[judge_idx, idx1] += 0 * weight
                                        comparison_matrix[judge_idx, idx2] += 0 * weight
    return comparison_matrix, weight


def vote_to_rank(vote_matrix, weights, n):
    # 提取前三列
    first_three_columns = vote_matrix[:, :n]
    vote_sum = list()
    # 分别计算前三列的和
    for i in range(n):
        vote_sum.append(np.sum(first_three_columns[:, i] * weights))
        print(vote_sum)
        # vote_sum.append(np.sum(first_three_columns[:, i]))
        sorted_indices = sorted(range(len(vote_sum)), key=lambda i: (vote_sum[i] == 0, -vote_sum[i]))
        # print(sorted_indices)
        # 计算排序后的排名
    ranking = [0] * n
    for rank, index in enumerate(sorted_indices):
        ranking[index] = rank
        # print(ranking)
    # if n == 2:
    #     # 最后一名
    #     if ranking[-1] == 1 and int(model_scores[bottle_model[0]]) != 100:
    #         model_scores[new_model] = 60 + model_scores[bottle_model[0]]*vote_sum[1]/vote_sum[0]
    #     if ranking[-1] == 1:
    #         model_scores[new_model] = 60 + model_scores[bottle_model[0]]*vote_sum[1]/vote_sum[0]
    # breakpoint()
    return vote_sum, ranking

def update_model_weight(initial_weight, base_model_ranking, judge_model_list):
    return model_scores


def update_bubble_window_rank(base_model_ranking, model_names, new_model_rank, ranking):
    base_model_ranking[model_names[0]] = new_model_rank + ranking[0] - 1
    base_model_ranking[model_names[1]] = new_model_rank + ranking[1] - 1
    base_model_ranking[model_names[2]] = new_model_rank + ranking[2] - 1
    return base_model_ranking


def judge_bubble(sort_model_index_map, min_rank_idx, base_model_list, ranking, new_model_rank):
    flag_bubble = False
    if len(sort_model_index_map) == 2 and min_rank_idx == 1:
        if ranking[1] == 1:
            flag_bubble = -1
    if len(sort_model_index_map) == 2 and min_rank_idx == len(base_model_list) - 1:
        if ranking[1] == 0:
            flag_bubble = 1
    if len(sort_model_index_map) == 3 and ranking[2] != 1:
        if new_model_rank != 1 and new_model_rank != len(base_model_list):
            if ranking[2] == 0:
                flag_bubble = 1
            if ranking[2] == 2:
                flag_bubble = -1
    print(flag_bubble)
    return flag_bubble


def judge_continue_bubble(sort_model_index_map, base_model_list, ranking, new_model_rank):
    flag_bubble = False
    if len(sort_model_index_map) == 3 and ranking[2] != 1:
        if new_model_rank != 1 and new_model_rank != len(base_model_list):
            if ranking[2] == 0:
                flag_bubble = 1
            if ranking[2] == 2:
                flag_bubble = -1
    return flag_bubble


def get_final_avg_rank(final_model_list):
    # 用于存储模型排名信息的字典
    ranking_stats = {}
    # 遍历每种方法的排名
    for method_rank in final_model_list:
        for rank, model in enumerate(method_rank):
            if model not in ranking_stats:
                ranking_stats[model] = {
                    'total_rank': 0,
                    'count': 0,
                    'min_rank': float('inf'),
                    'max_rank': float('-inf')
                }

            # 更新总排名和计数（rank + 1 使其为 1-based ranking）
            ranking_stats[model]['total_rank'] += rank + 1
            ranking_stats[model]['count'] += 1

            # 更新最低和最高排名
            ranking_stats[model]['min_rank'] = min(ranking_stats[model]['min_rank'], rank + 1)
            ranking_stats[model]['max_rank'] = max(ranking_stats[model]['max_rank'], rank + 1)
    print(ranking_stats)
    # 计算平均排名并准备最终结果
    final_results = {}
    for model, stats in ranking_stats.items():
        average_rank = stats['total_rank'] / stats['count']
        final_results[model] = average_rank
    print(final_results)
    # 根据平均排名排序
    sorted_models = sorted(final_results.items(), key=lambda x: x[1])
    print(sorted_models)
    # 处理并列情况生成最终排名
    final_ranked_list = []
    last_rank = 0
    last_average = None
    for index, (model, average_rank) in enumerate(sorted_models):
        if last_average is not None and average_rank == last_average:
            final_ranked_list.append(last_rank)  # 并列情况，保持上一个排名
        else:
            last_rank = index + 1  # 1-based ranking
            final_ranked_list.append(last_rank)
        last_average = average_rank
    # 输出结果
    print("最终模型排名列表:", final_ranked_list)
    with open(save_output_file_path, 'a') as f:
        f.write(f"ranking_stats: {ranking_stats}\n")
        f.write(f"avg ranking: {sorted_models}\n")
        f.write(f"final_ranked_list: {final_ranked_list}\n")
    return sorted_models, final_ranked_list


# 用来排名base model，第一步base model先进行full sample
def base_model_judge(base_dir, base_model_list, valid_question_ids=overall_ids):
    model_weights = {model: 1 for model in base_model_list}
    # print(judge_models)
    sort_model_index_map = {name: idx for idx, name in enumerate(base_model_list)}
    judge_model_index_map = {name: idx for idx, name in enumerate(base_model_list)}
    print(judge_model_index_map)
    # Initialize an empty comparison matrix
    final_comparison_matrix = np.zeros((len(base_model_list), len(base_model_list)))

    paras = list()
    for subdir in os.listdir(base_dir):
        # print(subdir)
        if subdir not in base_model_list:
            continue
        if subdir not in base_model_list:
            continue
        models_to_sort = [model for model in base_model_list if model != subdir]
        print(models_to_sort)
        combinations = list(itertools.combinations(models_to_sort, 2))
        remaining_combinations = set(combinations)
        comparison_matrix = np.zeros((len(base_model_list), len(base_model_list)))
        paras.append([base_dir, subdir, model_weights, sort_model_index_map, judge_model_index_map, comparison_matrix,
                      remaining_combinations, valid_question_ids])
    with Pool(processes=(cpu_count() - 1)) as pool:
        result = pool.starmap(pairwise_judge, paras)
    for i in range(len(result)):
        final_comparison_matrix += result[i][0]
    print(final_comparison_matrix)
    row_sums = final_comparison_matrix.sum(axis=1, keepdims=True)
    # 创建归一化矩阵，初始化为原始矩阵
    normalized_matrix = np.copy(final_comparison_matrix)
    # 对每一行进行归一化（忽略和为0的行）
    nonzero_row_indices = row_sums.flatten() != 0
    normalized_matrix[nonzero_row_indices] = final_comparison_matrix[nonzero_row_indices] / row_sums[
        nonzero_row_indices]
    return normalized_matrix

def generate_comparison_matrix(comparsion,n):
    # 填充矩阵
    for i in range(n):
        if i == 0 :
            comparsion[i][i] = 1
            comparsion[i][i+1] = 1
        elif i == n-1:
            comparsion[i][i-1] = 1
            comparsion[i][i] = 1
        else:
            comparsion[i][i-1] = 1
            comparsion[i][i] = 1
            comparsion[i][i+1] = 1
    return comparsion

def update_model_scores(winrate_matrix,base_model_list,model_to_idx):
    print(base_model_list)
    n = len(base_model_list)
    tmp_winrate_matrix = np.zeros((n,n))

    # 遍历矩阵并保留所需元素
    for i in range(tmp_winrate_matrix.shape[0]):
        if i < tmp_winrate_matrix.shape[0]:  # 确保不超出列数
            tmp_winrate_matrix[i, i] = 0.5  # 保留对角线元素
        if i + 1 < tmp_winrate_matrix.shape[0] and i < tmp_winrate_matrix.shape[0]:  # 保留下方斜线元素
            tmp_winrate_matrix[i + 1, i] = winrate_matrix[model_to_idx[base_model_list[i+1]], model_to_idx[base_model_list[i]]]
        if i - 1 >= 0 and i < tmp_winrate_matrix.shape[0]:  # 保留上方斜线元素
            tmp_winrate_matrix[i - 1, i] = winrate_matrix[model_to_idx[base_model_list[i-1]], model_to_idx[base_model_list[i]]]

    # Initialize parameters
    initial_scores = np.zeros(n)
    comparison = np.zeros((tmp_winrate_matrix.shape[0],tmp_winrate_matrix.shape[0]))
    comparison = generate_comparison_matrix(comparison,n)
    # Perform optimization to find the best scores
    result = minimize(bt_likelihood, initial_scores, args=(comparison, tmp_winrate_matrix), method='BFGS')

    # Extract the optimized scores
    bt_scores = result.x
    print(bt_scores)
    # 计算最小值和最大值
    bt_min = np.min(bt_scores)
    bt_max = np.max(bt_scores)

    # 归一化到 0-100 的区间
    normalized_bt = (bt_scores - bt_min) / (bt_max - bt_min) * 100

    print(normalized_bt)
    model_scores = {base_model_list[i]:normalized_bt[i] for i in range(n)}
    print(model_scores)
    return model_scores

def bt_likelihood(scores, comparison, matrix):
    n = len(scores)
    likelihood = 0
    for i in range(n):
        for j in range(n):
            if comparison[i, j] == 1:  # Only include pairs where comparisons are known
                prob = np.exp(scores[i]) / (np.exp(scores[i]) + np.exp(scores[j]))
                likelihood += matrix[i, j] * np.log(prob) + (1 - matrix[i, j]) * np.log(1 - prob)
    return -likelihood  # We minimize negative log-likelihood


def main(base_dir="/data/shared/decentralized_arena/judgements_mt_bench_kun/judgements_mt_bench", valid_question_ids=overall_ids,
         existing_model_paths=existing_model_paths, gt_scores=gt_scores):
    judge_model_list = judge_api_model + judge_open_model
    # Overall
    model_list = list(existing_model_paths.keys())
    judge_model_states = {model: list() for model in judge_model_list}
    judge_model_states['gemini-1.5-flash-001'] = list()

    overall_model_list = ['o1-mini', 'o1-preview', 'ChatGPT-4o-latest (2024-09-03)', 'gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-08-06', 'qwen2-72b-instruct', 'meta-llama-3.1-70b-instruct', 'athene-70b', 'gpt-4-turbo-2024-04-09', 'claude-3.5-sonnet', 'gemini-1.5-pro-001', 'gpt4-1106', 'claude-3-opus', 'gemini-1.5-flash-001', 'gemma-2-27b-it', 'yi-1.5-34b-chat', 'google-gemma-2-9b-it', 'gemma-2-9b-it-simpo', 'llama-3-70b-instruct', 'claude-3-sonnet', 'meta-llama-3.1-8b-instruct', 'claude-3-haiku', 'qwen1.5-72b-chat', 'qwen1.5-32b-chat', 'gpt3.5-turbo-0125', 'claude-2.0', 'mistral-8x7b-instruct-v0.1', 'claude-2.1', 'starling-lm-7b-beta', 'qwen1.5-14b-chat', 'openchat-3.5-0106', 'command-r-(08-2024)', 'openchat-3.5', 'starling-lm-7b-alpha', 'llama3-8b-instruct', 'gemma-2-2b-it', 'gemini-1.0-pro-001', 'gemma-1.1-7b-it', 'command-r-(04-2024)', 'mistral-7b-instruct-2', 'mistral-7b-instruct-1', 'vicuna-33b', 'gemma-7b-it', 'qwen1.5-4b-chat', 'vicuna-13b', 'llama2-13b-chat', 'gemma-1.1-2b-it', 'zephyr-7b-beta', 'llama2-7b-chat', 'vicuna-7b', 'gemma-2b-it', 'koala-13b', 'openassistant-pythia-12b','nemotron-70b','llama-3.2-3b-it','yi-lightning','glm-4-plus',"claude-3.5-sonnet-20241022",'qwen2.5-1.5b','smollm2-1.7b','llama-3.2-1b-it','ministral-8b-it']
    print(len(overall_model_list))
    model_to_idx = {model:i for i,model in enumerate(overall_model_list)}

    base_model_lists = [["ChatGPT-4o-latest (2024-09-03)","gpt-4o-2024-08-06","gpt-4o-mini-2024-07-18","gemma-2-27b-it","qwen2-72b-instruct","google-gemma-2-9b-it"]]

    base_matrix = base_model_judge(base_dir,base_model_lists[0])

    final_result = [["base_model_list", "final_spearman_rank_correlation_mean", "final_spearman_rank_correlation_var",
                     "final_binary_search_mean", "final_binary_search_var", "spearman_rank_correlation_mean",
                     "spearman_rank_correlation_var", "iter_binary_search_mean", "iter_binary_search_var"]]

    print(len(judge_model_list))

    for ori_base_model_list in tqdm(base_model_lists, desc="Processing models"):
        # ori_base_model_list = sorted(ori_base_model_list, key=lambda model: gt_scores[model], reverse=True)
        gt_scores_list = [gt_scores[model] for model in ori_base_model_list]
        ori_base_model_ranking = dict()
        for rank, model in enumerate(ori_base_model_list):
            ori_base_model_ranking[model] = rank + 1

        spearman_rank_correlation_list = list()
        final_corr = list()
        rugged_rank_diff = list()

        print(ori_base_model_list)

        final_model_list = list()
        start_time = time.time()
        # 重复排名
        for idx in tqdm(range(3), desc="Processing shuffle"):

            judge_model_list = judge_api_model + judge_open_model
            # model_scores = {'ChatGPT-4o-latest (2024-09-03)': 100.0, 'gpt-4o-2024-08-06': 97.58454106280193, 'gpt-4o-mini-2024-07-18': 94.32367149758454, 'meta-llama-3.1-70b-instruct': 93.23671497584542, 'qwen2-72b-instruct': 91.42512077294685, 'athene-70b': 91.18357487922705, 'gemini-1.5-flash-001': 91.06280193236715, 'openassistant-pythia-12b': 50.0}
            base_model_list = copy.deepcopy(ori_base_model_list)
            base_model_ranking = copy.deepcopy(ori_base_model_ranking)
            winrate_matrix = np.zeros((len(model_to_idx), len(model_to_idx)))
            vote_sum = np.sum(base_matrix, axis=0)
            for i in range(len(base_model_lists[0])):
                for j in range(i+1,len(base_model_lists[0])):
                    winrate_matrix[model_to_idx[base_model_lists[0][i]]][model_to_idx[base_model_lists[0][j]]] = vote_sum[i]/(vote_sum[j]+vote_sum[i])
                    winrate_matrix[model_to_idx[base_model_lists[0][j]]][model_to_idx[base_model_lists[0][i]]] = vote_sum[j]/(vote_sum[j]+vote_sum[i])
            model_scores = update_model_scores(winrate_matrix,base_model_lists[0],model_to_idx)
            print(model_scores)
            original_model_list = copy.deepcopy(base_model_list)
            model_names = list(existing_model_paths.keys())
            models_to_sort = [model for model in model_names if model not in base_model_list]
            gt_scores_list = [gt_scores[model] for model in models_to_sort]
            # 将模型和得分配对
            models_with_scores = list(zip(models_to_sort, gt_scores_list))
            # 按照得分排序并选取前十个
            # 过滤出在 judge model list 中的模型
            judge_top_models = [model for model in models_with_scores if model[0] in judge_model_list]
            top_models = sorted(judge_top_models, key=lambda x: x[1], reverse=True)[:7]
            # 仅提取模型名称
            top_models_names = [model[0] for model in top_models]
            print(top_models)
            # 随机插入其他模型
            remaining_models = [model for model in models_to_sort if model not in top_models_names]
            random.seed(idx)
            random.shuffle(remaining_models)
            # 合并结果
            models_to_sort = top_models_names + remaining_models
            # overall
            models_to_sort = ['o1-mini','o1-preview',"gemini-1.5-flash-001","gemini-1.5-pro-001","gemini-1.0-pro-001"] + top_models_names + remaining_models  + ["llama3-8b-instruct","meta-llama-3.1-70b-instruct","meta-llama-3.1-8b-instruct","llama-3-70b-instruct",'llama-3.2-3b-it',"nemotron-70b","yi-lightning","glm-4-plus","claude-3.5-sonnet-20241022",'qwen2.5-1.5b','smollm2-1.7b','llama-3.2-1b-it','ministral-8b-it']
            add_model = list()
            spearman_rank_correlation_list.append([])

            model_weights = model_scores
            bottle_model_list = list()
            for new_model in tqdm(models_to_sort, desc="Processing sorting"):
                if new_model in ["llama3-8b-instruct","meta-llama-3.1-70b-instruct","meta-llama-3.1-8b-instruct","llama-3-70b-instruct"]:
                    judge_model_list = judge_api_model+['gemini-1.5-flash-001']
                else:
                    judge_model_list = judge_api_model+judge_open_model
                
                # 二分查找排名
                sort_rank,judge_model_states = rugged_rank(base_dir, new_model, base_model_list, base_model_ranking, model_weights,
                                        judge_model_list, judge_model_states)
                print(new_model, sort_rank)
                # 第一次细粒度排名
                sort_model_index_map, min_rank_idx, vote_matrix, weights, bottle_model_list, judge_model_states = full_comparsion(base_dir,
                                                                                                              new_model,
                                                                                                              base_model_list,
                                                                                                              sort_rank,
                                                                                                              model_weights,
                                                                                                              judge_model_list,
                                                                                                              judge_model_states)
                print(vote_matrix)
                vote_sum, ranking = vote_to_rank(vote_matrix, weights, len(sort_model_index_map))
                print(bottle_model_list,vote_sum)
                for i in range(len(bottle_model_list)):
                    for j in range(i+1,len(bottle_model_list)):
                        winrate_matrix[model_to_idx[bottle_model_list[i]]][model_to_idx[bottle_model_list[j]]] = vote_sum[i]/(vote_sum[j]+vote_sum[i])
                        winrate_matrix[model_to_idx[bottle_model_list[j]]][model_to_idx[bottle_model_list[i]]] = vote_sum[j]/(vote_sum[j]+vote_sum[i])
                        # print(winrate_matrix[model_to_idx[bottle_model_list[i]]][model_to_idx[bottle_model_list[j]]])
                # print(winrate_matrix)
                # breakpoint()
                if len(sort_model_index_map) == 1:
                    base_model_ranking[new_model] = len(vote_matrix)
                else:
                    base_model_ranking[new_model] = 10000
                gt_scores_list = [gt_scores[model] for model in base_model_list]
                new_model_gt_ranks = rank_scores(gt_scores_list)[-1]
                base_model_ranking = integrate_rankings(base_model_ranking, sort_model_index_map, ranking)
                print(base_model_ranking)
                base_model_list = sorted(base_model_ranking, key=base_model_ranking.get)
                model_scores = update_model_scores(winrate_matrix,base_model_list,model_to_idx)

                model_weights = model_scores
                # 判断是否继续进行bubble window
                new_model_rank = base_model_ranking[new_model]
                flag_bubble = judge_bubble(sort_model_index_map, min_rank_idx, base_model_list, ranking, new_model_rank)
                while flag_bubble:
                    print("============")
                    model_names, sort_model_index_map, vote_matrix, weights, bottle_model_list,judge_model_states = bubble_window(base_dir,
                                                                                                               new_model,
                                                                                                               base_model_list,
                                                                                                               new_model_rank,
                                                                                                               model_weights,
                                                                                                               judge_model_list,
                                                                                                               judge_model_states)
                    print(vote_matrix)
                    vote_sum, ranking = vote_to_rank(vote_matrix, weights, len(sort_model_index_map))
                    print(bottle_model_list,vote_sum)
                    for i in range(len(bottle_model_list)):
                        for j in range(i+1,len(bottle_model_list)):
                            winrate_matrix[model_to_idx[bottle_model_list[i]]][model_to_idx[bottle_model_list[j]]] = vote_sum[i]/(vote_sum[j]+vote_sum[i])
                            winrate_matrix[model_to_idx[bottle_model_list[j]]][model_to_idx[bottle_model_list[i]]] = vote_sum[j]/(vote_sum[j]+vote_sum[i])

                    # 更新排名
                    base_model_ranking = update_bubble_window_rank(base_model_ranking, model_names, new_model_rank,
                                                                   ranking)
                    base_model_list = sorted(base_model_ranking, key=base_model_ranking.get)
                    model_scores = update_model_scores(winrate_matrix,base_model_list,model_to_idx)
                    model_weights = model_scores

                    new_model_rank = base_model_ranking[new_model]
                    # 判断是否仍要继续bubble sort
                    if flag_bubble != judge_continue_bubble(sort_model_index_map, base_model_list, ranking,
                                                            new_model_rank):
                        if judge_continue_bubble(sort_model_index_map, base_model_list, ranking,
                                                 new_model_rank) != False:
                            idx_flag = judge_continue_bubble(sort_model_index_map, base_model_list, ranking,
                                                             new_model_rank)
                        flag_bubble = False
                # 统计最终结果
                ranking = list()
                for i in base_model_list:
                    ranking.append(base_model_ranking[i])
                gt_scores_list = [gt_scores[model] for model in base_model_list]
                gt_ranks = rank_scores(gt_scores_list)
                rugged_rank_diff[idx].append(abs(sort_rank - new_model_gt_ranks))
                spearman_rank_correlation, _ = spearmanr(ranking, gt_ranks)
                add_model.append(new_model)
                spearman_rank_correlation_list[idx].append(spearman_rank_correlation)
                print(spearman_rank_correlation)
                print(base_model_list)
                with open(save_output_file_path, 'a') as f:
                    f.write(
                        f"new model: {new_model}, binary_rank: {sort_rank}, corrleation: {spearman_rank_correlation}\n")
                    f.write(f"step model list: {base_model_list}\n")
                    f.write(f"step model scores: {model_scores}\n")
                print(model_scores)

            print(spearman_rank_correlation_list[idx])
            final_model_list.append(base_model_list)
            final_corr.append(spearman_rank_correlation_list[idx][-1])
            with open(save_output_file_path, 'a') as f:
                f.write(f"model socres: {model_scores}\n")
                f.write(f"\n ------------------------------ \n")


        with open(save_output_file_path, 'a') as f:
            f.write(f"judge model states: {judge_model_states}\n")

        # 计算每一步插入的corrleation均值和方差
        spearman_array = np.array(spearman_rank_correlation_list)
        # 对每一列计算均值
        spearman_column_means = np.mean(spearman_array, axis=0)
        # 对每一列计算方差
        spearman_column_variances = np.var(spearman_array, axis=0)
        print("spearman_Column Means:")
        print(spearman_column_means)
        print("spearman_Column Variances:")
        print(spearman_column_variances)

        sorted_models, final_ranked_list = get_final_avg_rank(final_model_list)
        gt_scores_list = [gt_scores[model[0]] for model in sorted_models]
        gt_ranks = rank_scores(gt_scores_list)
        spearman_rank_correlation, _ = spearmanr(ranking, gt_ranks)
        print(spearman_rank_correlation)

    end_time = time.time()
    print(end_time - start_time)
    # # 指定 CSV 文件名
    # file_name = 'output_overall.csv'

    # # 打开文件并写入二维列表
    # with open(file_name, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(final_result)  # 写入所有行

    # print(f"数据已写入 {file_name}")


if __name__ == "__main__":
    fire.Fire(main)
