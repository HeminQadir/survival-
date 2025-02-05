"""
Author: Hemin Qadir
Date: 15.01.2024
Task: Spliting the dataset into train and validation subset 

"""

import json

def datafold_read(args, key="training"): #datalist, fold=0, ):
    with open(args.dataset_json) as f:
        json_data = json.load(f)
    json_data = json_data[key]
    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == args.fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val



def datafold_read_inference(args, key="training"): #datalist, fold=0, ):
    with open(args.dataset_json) as f:
        json_data = json.load(f)
    val = json_data[key]
    return val