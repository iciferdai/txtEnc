from data_dict import demo_data
from myTrans.base_params import *
import pprint
import re

def process_data():
    full_str = ''
    max_len=0
    for msg, res in demo_data:
        if len(msg) > max_len: max_len = len(msg)
        full_str += msg
    txt_list = list(full_str)
    cn_set = set(txt_list)

    id = CUS_START_ID
    t_dict = dict()

    for i in cn_set:
        t_dict[i] = id
        id += 1

    print(f"length: {len(demo_data)}|{len(full_str)}|{len(cn_set)}|{len(t_dict)}|{max_len}")
    print("CN set: ")
    print(cn_set)
    print("VOCAB Dict:")
    print(t_dict)

if __name__ == '__main__':
    process_data()


