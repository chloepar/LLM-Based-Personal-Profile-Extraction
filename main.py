import argparse
import LLMPersonalInfoExtraction as PIE
from LLMPersonalInfoExtraction.utils import open_config
from LLMPersonalInfoExtraction.utils import open_txt, load_image
from LLMPersonalInfoExtraction.utils import load_instruction, parsed_data_to_string
import time
import os
import numpy as np
from pathlib import Path

def is_valid_response(response):
    if not response or len(response.strip()) < 3:
        return False
    error_phrases = ['rate limit', 'exceeded', 'quota', 'too many requests', 'error']
    return not any(p in response.lower() for p in error_phrases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIE Execution')
    parser.add_argument('--model_config_path', default='./configs/model_configs/palm2_config.json', type=str)
    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument('--task_config_path', default='./configs/task_configs/synthetic.json', type=str)
    parser.add_argument('--api_key_pos', default=0, type=int)
    parser.add_argument('--defense', default='no', type=str)
    parser.add_argument('--prompt_type', default='direct', type=str)
    parser.add_argument('--gpus', default='', type=str)
    parser.add_argument('--icl_num', default=0, type=int)
    parser.add_argument('--adaptive_attack', default='no', type=str)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--redundant_info_filtering', default='True', type=str)
    args = parser.parse_args()

    task_config = open_config(config_path=args.task_config_path)
    task_manager, icl_manager = PIE.create_task(task_config)

    defense = PIE.create_defense(args.defense)

    model_config = open_config(config_path=args.model_config_path)
    model_config['model_info']['name'] = args.model_name
    if 'palm' in args.model_config_path or 'gemini' in args.model_config_path or 'gpt' in args.model_config_path:
        assert (0 <= args.api_key_pos < len(model_config["api_key_info"]["api_keys"]))
        model_config["api_key_info"]["api_key_use"] = args.api_key_pos
        print(f'API KEY POS = {model_config["api_key_info"]["api_key_use"]}')
    else:
        args.gpus = args.gpus.split(',')
        assert (args.gpus is not [])
        model_config["params"]["gpus"] = args.gpus
        print(f'GPUS = {model_config["params"]["gpus"]}')

    model = PIE.create_model(config=model_config)
    model.print_model_info()
    attacker = PIE.create_attacker(model, adaptive_attack=args.adaptive_attack, icl_manager=icl_manager, prompt_type=args.prompt_type)

    info_cats = open_txt('./data/system_prompts/info_category.txt')
    evaluator = PIE.create_evaluator(model.provider, info_cats)
    instructions = load_instruction(args.prompt_type, info_cats)

    need_adaptive_attack = (args.defense in ('pi_ci', 'pi_id', 'pi_ci_id', 'no'))
    if not need_adaptive_attack:  args.adaptive_attack = 'no'
    email_only = defense.defense not in ['no', 'pi_ci', 'pi_id', 'pi_ci_id', 'image']
    
    if args.redundant_info_filtering == 'True':
        res_save_path = f'./result/{model.provider}_{model.name.split("/")[-1]}/{task_manager.dataset}_{args.defense}_{args.prompt_type}_{args.icl_num}_adaptive_attack_{args.adaptive_attack}'
    else:
        res_save_path = f'./result/{model.provider}_{model.name.split("/")[-1]}/{task_manager.dataset}_{args.defense}_{args.prompt_type}_{args.icl_num}_adaptive_attack_{args.adaptive_attack}_{args.redundant_info_filtering}'
    os.makedirs(res_save_path, exist_ok=True)

    # Checkpoint for resume safety
    checkpoint_path = f'{res_save_path}/.checkpoint.npz'
    start_idx = 0
    
    # Load existing checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f'[RESUME] Loading checkpoint from {checkpoint_path}')
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        all_raw_responses = dict(checkpoint['res'].item())
        all_labels = dict(checkpoint['label'].item())
        start_idx = int(checkpoint['last_profile_idx']) + 1
        print(f'[RESUME] Resuming from profile {start_idx}/{len(task_manager)}')
    else:
        all_raw_responses = dict(zip(info_cats, [[] for _ in range(len(info_cats))]))
        all_labels = dict(zip(info_cats, [[] for _ in range(len(info_cats))]))

    for i in range(start_idx, len(task_manager)):
        raw_list, curr_label = task_manager[i]

        # Flatten structured fields to strings for evaluator compatibility
        if 'education' in curr_label and isinstance(curr_label['education'], list):
            curr_label['education'] = ", ".join(
                f"{e.get('degree', '')} from {e.get('institution', '')}".strip()
                + (f" in {e.get('year')}" if e.get('year') else "")
                for e in curr_label['education']
            )

        try:
            raw_list = defense.apply(raw_list, curr_label)
        except ValueError:
            print('Not applicable. Skip')
            continue
        
        if args.verbose > 0:
            print(f'{i+1} / {len(task_manager)}: {task_manager.filenames[i].replace(".html", "")}')

        if args.defense == 'image':
            if model.provider == 'gemini':
                img = load_image(f'./data/synthetic_images/{task_manager.filenames[i].replace(".html", "")}.jpg')
                if img is None:
                    print(f'Skip bad image: ./data/synthetic_images/{task_manager.filenames[i].replace(".html", "")}.jpg\n')
                    continue
            elif model.provider == 'gpt':
                img = f'./data/synthetic_images/{task_manager.filenames[i].replace(".html", "")}.jpg'
            else:
                raise ValueError
        else:
            img = None

        raw = '\n'.join(raw_list)

        if args.redundant_info_filtering == 'True':
            redundant_info_filter = PIE.get_parser(task_manager.dataset)
            redundant_info_filter.feed(raw)
            processed_data = redundant_info_filter.data
            try:
                processed_data = defense.apply(parsed_data_to_string(task_manager.dataset, processed_data, model.name), curr_label)
            except ValueError:
                print('Not applicable. Skip')
                continue
        else:
            processed_data = raw

        cnt = 0
        for info_cat, instruction in instructions.items():
            if email_only and info_cat != 'email':
                continue
            
            # Sleep for a while to avoid exceeding the rate limit
            if cnt == 1 and model.provider in ('palm2', 'gpt', 'gemini', 'groq'):
                time.sleep(3)

            try:
                raw_response = attacker.query(
                    instruction, 
                    processed_data,
                    icl_num=args.icl_num,
                    info_cat=info_cat,
                    need_adaptive_attack=need_adaptive_attack,
                    verbose=1 if i == 0 else 0,
                    idx=i,
                    total=len(task_manager),
                    image=img
                )
            except RuntimeError:
                # This can happen for various reasons: API outdated, instable network, rate limit exceeded, etc. 
                # For simplicity we mark the prediciton as empty. 
                raw_response = ""

            all_raw_responses[info_cat].append(raw_response)
            all_labels[info_cat].append(curr_label[info_cat])
            _ = evaluator.update(raw_response, curr_label, info_cat, defense, verbose=args.verbose)
            cnt = (cnt + 1) % 2
        if args.verbose > 0:  print('\n----------------\n')
        
        # Save checkpoint every 10 profiles for resume safety
        if (i + 1) % 10 == 0 or i == len(task_manager) - 1:
            # Flag any invalid responses before saving
            invalid_count = sum(
                1 for info_cat in info_cats
                for r in all_raw_responses[info_cat]
                if not is_valid_response(r)
            )
            if invalid_count > 0 and args.verbose > 0:
                print(f'[CHECKPOINT WARNING] {invalid_count} potentially invalid responses detected')
            np.savez(checkpoint_path, res=all_raw_responses, label=all_labels, last_profile_idx=i)
    
    np.savez(f'{res_save_path}/all_raw_responses.npz', res=all_raw_responses, label=all_labels)
    print(f'\nResults are saved at: {res_save_path}\n')
    print('[END]')