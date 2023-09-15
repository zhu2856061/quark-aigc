# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/8/1|下午 03:12
# @Moto   : Knowledge comes from decomposition

import time
import json
import pandas as pd
"""
特征名
特征类型: 1 float ; 2 string ; 3 int
特征长度
"""
all_feature = [
    ("text", 2, 1),
    ("image", 2, 1),
    ("width", 3, 1),
    ("height", 3, 1),
    ("clip_score", 1, 1),
    ("aesth", 1, 1),
]

names=['hash_id', 'sku_id', 'url', 'image', 'text', 'width', 'height', 'clip_score', 'aesth']
lines = pd.read_csv("./part-00000-2bde0185-c417-4441-8c7b-d25d80e72c66-c000.csv", names=names)

lines = lines.to_dict(orient='index')

result = open("./part-1.data.format", "w", encoding='utf8')
for i, line in lines.items():
    tmp_dic = {
        "x": {},
    }
    for fea in all_feature:
        if fea[1] == 1:
            tmp_dic["x"][fea[0]]={
                "FVA": [float(line[fea[0]])]
            }
        if fea[1] == 2:
            tmp_dic["x"][fea[0]]={
                "SVA": [str(line[fea[0]])]
            }
        if fea[1] == 3:
            tmp_dic["x"][fea[0]]={
                "IVA": [int(line[fea[0]])]
            }

    result.write(json.dumps(tmp_dic, ensure_ascii=False) + "\n")

result.close()
