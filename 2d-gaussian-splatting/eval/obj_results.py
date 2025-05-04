import os
import json
import csv
from glob import glob
from argparse import ArgumentParser


def get_obj_merge_results(exp_root_path, exp_key):

    ave_results = {}
    max_results = {}
    min_results = {}
    merge_results = {}

    exp_list = os.listdir(exp_root_path)
    key_exp_num = 0
    for exp_name in exp_list:
        if exp_key in exp_name:

            if not os.path.isdir(os.path.join(exp_root_path, exp_name)):
                continue

            all_obj_results_dict = {}
            obj_num = 0

            obj_results_list = glob(os.path.join(exp_root_path, exp_name, 'obj_*_metrics.json'))
            for obj_results_path in obj_results_list:
                obj_results_name = os.path.basename(obj_results_path)
                obj_id = obj_results_name.split('_')[1]

                if obj_id == 0:             # skip background
                    print(f'We not merge background obj results, because background range is not same with other obj')
                    continue

                with open(obj_results_path, 'r') as f:
                    obj_results = json.load(f)

                for key, value in obj_results.items():
                    if key not in all_obj_results_dict:
                        all_obj_results_dict[key] = value
                    else:
                        all_obj_results_dict[key] += value

                obj_num += 1

            for key, value in all_obj_results_dict.items():
                all_obj_results_dict[key] = value * 1.0 / obj_num * 1.0

            save_all_obj_results_path = os.path.join(exp_root_path, exp_name, 'all_obj_results.json')
            with open(save_all_obj_results_path, 'w') as f:
                json.dump(all_obj_results_dict, f)

            for key, value in all_obj_results_dict.items():
                if key not in ave_results:
                    ave_results[key] = value
                else:
                    ave_results[key] += value

                if key not in max_results:
                    max_results[key] = value
                else:
                    max_results[key] = max(max_results[key], value)

                if key not in min_results:
                    min_results[key] = value
                else:
                    min_results[key] = min(min_results[key], value)

            key_exp_num += 1

    merge_results['exp_num'] = key_exp_num
    for key, value in ave_results.items():
        ave_results[key] = value * 1.0 / key_exp_num * 1.0

        merge_results[f'{key}_ave'] = ave_results[key]
        merge_results[f'{key}_max'] = max_results[key]
        merge_results[f'{key}_min'] = min_results[key]

    return merge_results


if __name__ == '__main__':

    parser = ArgumentParser(description="Calculate Replica Scene")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--exp_key", type=str, required=True)
    args = parser.parse_args()

    root_path = args.root_path
    exp_key = args.exp_key

    scene_num = 8
    merge_all_scene_results = {}
    merge_key_list = ['exp_num', 'Chamfer-L1', 'F-score', 'Normal-Consistency']
    for scan_id in range(1, scene_num+1):

        merge_all_scene_results[f'scan{scan_id}'] = {}

        exp_root_path = os.path.join(root_path, f'scan{scan_id}')
        merge_results = get_obj_merge_results(exp_root_path, exp_key)

        for key, value in merge_results.items():
            for save_key in merge_key_list:
                if save_key in key:
                    merge_all_scene_results[f'scan{scan_id}'][key] = round(value, 3)

    # get merge all scene results
    merge_all_scene_results[f'average'] = {}
    for scan_id in range(1, scene_num+1):

        for key, value in merge_all_scene_results[f'scan{scan_id}'].items():

            if key not in merge_all_scene_results[f'average']:
                merge_all_scene_results[f'average'][key] = value
            else:
                merge_all_scene_results[f'average'][key] += value

    for key, value in merge_all_scene_results[f'average'].items():
        if key == 'exp_num':
            continue
        merge_all_scene_results[f'average'][key] = round(value * 1.0 / scene_num * 1.0, 3)

    save_json_path = os.path.join(root_path, f'average_{exp_key}_merge_all_objs_results.json')
    if os.path.exists(save_json_path):
        os.remove(save_json_path)
    with open(save_json_path, 'w') as f:
        json.dump(merge_all_scene_results, f)
    save_csv_path = os.path.join(root_path, f'average_{exp_key}_merge_all_objs_results.csv')
    columns = list(next(iter(merge_all_scene_results.values())).keys())
    # print(columns)
    rows = merge_all_scene_results.keys()
    with open(save_csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # write header
        writer.writerow(["scan name"] + columns)

        # write data
        for name, values in merge_all_scene_results.items():
            writer.writerow([name] + [values[col] for col in columns])

    print('done')



