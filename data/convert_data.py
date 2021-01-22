# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 9:26 下午
# @Author  : jeffery
# @FileName: convert_data.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import pandas as pd
from pathlib import Path
import json
import numpy as np

def make_jsonl_files(raw_data_dir:Path,train_valid_test_dir:Path):
    # train
    train_file = raw_data_dir / 'train.csv'
    train_output = train_valid_test_dir / 'all.jsonl'
    train_writer = train_output.open('w')

    text_length = []
    with train_file.open('r') as csvfile:
        data_df = pd.read_csv(csvfile)
        for idx,row in data_df.iterrows():
            text_length.append(len(row['text']+('' if row['keyword'] is np.nan else row['keyword'])+('' if row['location'] is np.nan else row['location'])))

            train_writer.write(json.dumps({
                'id':row['id'],
                'text':row['text'],
                'keyword': '' if row['keyword'] is np.nan else row['keyword'],
                'location':'' if row['location'] is np.nan else row['location'],
                'label':row['target']
            },ensure_ascii=False)+'\n')
    train_writer.close()
    print('max length in train set:{}'.format(max(text_length)))
    text_length = []
    test_file = raw_data_dir / 'test.csv'
    test_output = train_valid_test_dir / 'test.jsonl'
    test_writer = test_output.open('w')

    with test_file.open('r') as csvfile:
        data_df = pd.read_csv(csvfile)
        for idx, row in data_df.iterrows():
            text_length.append(len(row['text']))

            test_writer.write(json.dumps({
                'id': row['id'],
                'text': row['text'],
                'keyword': '' if row['keyword'] is np.nan else row['keyword'],
                'location':'' if row['location'] is np.nan else row['location'],
                'label': -1
            },ensure_ascii=False)+'\n')
    test_writer.close()
    print('max length in test set:{}'.format(max(text_length)))



if __name__ == '__main__':
    raw_data_dir = Path('raw_data')
    train_valid_test_dir = Path('train_valid_test')
    make_jsonl_files(raw_data_dir,train_valid_test_dir)

