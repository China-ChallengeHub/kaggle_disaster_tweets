# -*- coding: utf-8 -*-
# @Time    : 2021/1/22 12:05 下午
# @Author  : jeffery
# @FileName: submit.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
from pathlib import Path
import json
from collections import defaultdict,OrderedDict,Counter
import csv
import pandas as pd

def make_submit(result_dir:Path,submit_file:Path):
    all_data = defaultdict(list)
    result_files = result_dir.glob('*.jsonl')
    for result_file in result_files:
        with result_file.open('r') as f:
            for line in f:
                json_line = json.loads(line)
                if int(json_line['labels']) == 1:
                    all_data[json_line['id']].append(1)
                else:
                    all_data[json_line['id']].append(0)




    with submit_file.open('w') as sf:
        for kid, lb_list in all_data.items():
            counter = Counter(lb_list)
            label = counter.most_common()[0][0]
            sf.write('{},{}\n'.format(kid,label))


def convert_single_result(infile:Path,outfile:Path):
    outwriter = outfile.open('w')
    with infile.open('r') as f:
        for line in f:
            json_line = json.loads(line)
            outwriter.write('{},{}'.format(json_line['id'],json_line['labels'])+'\n')

    outwriter.close()




if __name__ == '__main__':
    result_dir = Path('./')
    submit_file = Path('submit.csv')
    make_submit(result_dir,submit_file)
    # convert_single_result(Path('./TransformersModel-roberta-base-5-2-0.8555482625961304.jsonl'),Path('submit.csv'))




