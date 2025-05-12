import os
import json
import csv
from argparse import ArgumentParser


def get_merge_results(exp_root_path, exp_key, iter):

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

            exp_results_path = os.path.join(exp_root_path, exp_name, f'result_iter_{iter}.json')

            if not os.path.exists(exp_results_path):
                print(f'{exp_results_path} not exists')
                continue

            with open(exp_results_path, 'r') as f:
                exp_results = json.load(f)

            for key, value in exp_results.items():
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

    parser = ArgumentParser(description="Calculate Mip-NeRF 360 Scene")
    parser.add_argument("--root_path", type=str, required=True)
    args = parser.parse_args()

    root_path = args.root_path

    all_exp_list = os.listdir(root_path)
    all_exp_list = [x for x in all_exp_list if '.json' not in x and '.csv' not in x]
    all_exp_list.sort()
    all_exp_num = len(all_exp_list)

    scene_list = [x.split('-')[0] for x in all_exp_list]
    scene_list = list(set(scene_list))
    scene_list.sort()
    scene_num = len(scene_list)

    iter_list = [7000, 14000]
    for iter in iter_list:

        merge_all_scene_results = {}
        # merge_key_list = ['exp_num', 'Chamfer-L1', 'F-score', 'Normal-Consistency', 'Average-PSNR', 'Average-SSIM', 'Average-LPIPS']
        merge_key_list = ['exp_num', 'Average-PSNR', 'Average-SSIM', 'Average-LPIPS']
        for scene_name in scene_list:

            merge_all_scene_results[scene_name] = {}

            exp_root_path = root_path
            merge_results = get_merge_results(exp_root_path, scene_name, iter)          # merge each try

            for key, value in merge_results.items():
                for save_key in merge_key_list:
                    if save_key in key:
                        merge_all_scene_results[scene_name][key] = round(value, 3)

            save_json_path = os.path.join(exp_root_path, f'{scene_name}_iter{iter}_merge_results.json')
            if os.path.exists(save_json_path):
                # print(f'{save_json_path} exists, remove it')
                os.remove(save_json_path)
            with open(save_json_path, 'w') as f:
                json.dump(merge_results, f, indent=4)

            # print(f'scan{scan_id} merge results saved to {save_json_path}')
                
        # get merge all scene results
        merge_all_scene_results[f'average'] = {}
        for scene_name in scene_list:

            for key, value in merge_all_scene_results[scene_name].items():

                if key not in merge_all_scene_results[f'average']:
                    merge_all_scene_results[f'average'][key] = value
                else:
                    merge_all_scene_results[f'average'][key] += value

        for key, value in merge_all_scene_results[f'average'].items():
            if key == 'exp_num':
                continue
            merge_all_scene_results[f'average'][key] = round(value * 1.0 / scene_num * 1.0, 3)

        save_json_path = os.path.join(root_path, f'average_iter{iter}_merge_results.json')
        if os.path.exists(save_json_path):
            # print(f'{save_json_path} exists, remove it')
            os.remove(save_json_path)
        with open(save_json_path, 'w') as f:
            json.dump(merge_all_scene_results, f, indent=4)
        # save as .csv for easy reading
        save_csv_path = os.path.join(root_path, f'average_iter{iter}_merge_results.csv')
        rows = list(merge_all_scene_results.keys())
        columns = list(merge_all_scene_results[rows[0]].keys())
        with open(save_csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            # write header
            writer.writerow(["scan name"] + columns)

            # write data
            for name, values in merge_all_scene_results.items():
                writer.writerow([name] + [values[col] for col in columns])

        print(f'{iter} done')

    print('all done')

