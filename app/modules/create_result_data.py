# -*- coding:utf-8 -*-
from modules.predict import *
from model.doc import *
import os
import json


def write_to_file(seq_no, img, case_type, doc_ocr):
    """將辨識結果存成Json
    Args:
        seqNo (str): 辨識案件號(唯一值同時用做檔名)
        img (byte[])): 圖片檔.
        caseType (str): 文件類型 : LE2\LE3 => 存管公文、GA => 總務公文.
    """

    # 先建立Json出來
    rec_model = DocModel()
    rec_model.documentType = case_type
    rec_model.status = '1' # 處理中
    rec_model.msg = '辨識中'

    file_path = f'data/{seq_no}.json'
    
    if not os.path.exists(file_path):
        with open(file_path, 'w''', encoding='utf-8') as f:
            f.write(json.dumps(rec_model.__dict__, ensure_ascii=False))
            f.close()

    # 判斷文件類型
    if case_type == 'LE2' or case_type == 'LE3' or case_type == 'GA1': # LE2/LE3 => 存管公文
        
        json_str = general_affair_doc_rec(img) if case_type == 'GA1' else depository_doc_rec(img, doc_ocr)   # 辨識完的json
        # 這一段的目的是要處理公文欄位資訊分散在多張的問題，
        # 當多張公文傳入，每次都會去做全文辨識，並產生json，
        # 每次新產生的json欄位內容都會去跟前一次的json比較，
        # 並取值比較長的欄位內容當作新的json欄位值
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                pre_json_obj = json.load(f)
                f.close()

            new_json_obj = json.loads(json_str)

            if len(pre_json_obj['value']) > 0:
                for i in range(len(new_json_obj['value'])):
                    if new_json_obj['value'][i]['name'] not in ['義務人_債務人', '發文字號', '發文日期', '發文機關']:
                        if len(new_json_obj['value'][i]['text']) > len(pre_json_obj['value'][i]['text']):
                            pre_json_obj['value'][i]['text'] = new_json_obj['value'][i]['text']
                # update json
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(pre_json_obj, ensure_ascii=False))
                    f.close()
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(new_json_obj, ensure_ascii=False))
                    f.close()
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
                f.close()
