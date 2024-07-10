from collections import defaultdict

import json
from tqdm import tqdm

gt_dir = './results'

def convert(root):
    # read the gt.txt file
    with open(f'{root}/gt.txt', 'r') as file:
        lines = file.readlines()

    # process the lines and group by directory
    json_list = []
    d = defaultdict()

    for line in tqdm(lines):
        parts = line.strip().split('\t')
        path_parts = parts[0].split('/')
        file_except_images = '/'.join(path_parts[1:])
        words = parts[1]
        json_list.append([file_except_images, words])
        d[file_except_images] = words
    json_filename = f'{root}/labels.json'
    # save to json as json object not array
    with open(json_filename, 'w') as f:
        json.dump(d, f, ensure_ascii=False)

def main():
    convert(f'{gt_dir}/train_v11')
    # convert(f'{gt_dir}/test_v10')


if __name__ == '__main__':
    main()