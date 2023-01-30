# -*- coding:utf-8 -*-
# from paddleocr import PaddleOCR, draw_ocr
# from fastapi import Depends
import cv2
import numpy as np
import re
import json
import traceback
import threading

from model.doc import DocModel
# lock = threading.Lock()  # 建立 Lock
# from config.config import Settings
# from functools import lru_cache

# @lru_cache()
# def get_settings():
#     return settings()


# """paddle parameter"""
# ocr = PaddleOCR(
#     det_model_dir=r'ml_models/ppocv_server_v2.0/ch_ppocr_server_v2.0_det_infer/',
#     rec_model_dir=r'ml_models/ppocv_server_v2.0/ch_ppocr_server_v2.0_rec_infer/',
#     # det_model_dir=r'app/ml_models/ch_PP-OCRv3/ch_PP-OCRv3_det_infer/',
#     # rec_model_dir=r'app/ml_models/ch_PP-OCRv3/ch_PP-OCRv3_rec_infer/',
#     rec_char_dict_path=r'dict/cathay_dict.txt',
#     cls_model_dir=r'ml_models/ppocr_mobile_v2.0/ch_ppocr_mobile_v2.0_cls_infer/',
#     use_gpu=False,
#     det_db_thresh = 0.1,
#     det_db_box_thresh=0.1,
#     use_angle_cls=True,
#     use_space_char=True,
#     lang="ch",
# )

def exception_to_string(excp):
    """錯誤訊息追蹤"""
    stack = traceback.extract_stack()[:-3] + traceback.extract_tb(excp.__traceback__)
    pretty = traceback.format_list(stack)
    return ''.join(pretty) + '\n  {} {}'.format(excp.__class__,excp)


def depository_doc_rec(img, ocr):
    # lock.acquire()         # 鎖定
    """存管公文辨識"""
    rec_result = DocModel() # Create Object
    rec_result.documentType = 'LE2/LE3'
    rec_result.documentName = '存管公文'
    
    try:
        res = ocr.ocr(img) # 開始辨識
    except Exception as err:
        rec_result.status = '-1' # 異常
        rec_result.msg = '辨識過程失敗，請至log查看原因。' # 辨識失敗
        return json.dumps(rec_result.__dict__, ensure_ascii=False)

    """正規辨識結果"""
    results = ''
    results_lst = [res[i][1][0] for i in range(len(res))] # 將辨識出來的陣列儲存起來
    results = '|'.join(results_lst) # 將辨識出來的文字用「|」join起來
    results_txt = ''.join(results_lst) # 全文結果，無分割

    # 發文機關
    try:
        new_results = '|' + results # 避免關鍵字出現在第一排
        authority = re.search(r'([|][\w\s]{1,25})(執行署|地方法院|公路局|檢察署|監理所|法務部|稅捐|交通部|移民署|內政部|稅務局|警察局)(.*?)[|]', new_results).group()
        authority = re.sub(r'[^\w\s]', '', authority) # 去除符號
        authority = re.sub(r'台', '臺', authority) # 去除符號
        authority = re.sub(r'臺南', '台南', authority) # 去除符號
    except IndexError:
        try:
            authority = results_lst[0]
            authority = re.sub(r'台', '臺', authority) # 去除符號
            authority = re.sub(r'臺南', '台南', authority) # 去除符號
        except:
            authority = ''
    except:
        authority = ''

    # 發文日期
    try:
        date = re.findall(r'中華民國(\d*?年\d*?月\d*?日)', results_txt) # 日期
        date = date[0]
        date = re.sub(u'[^\u0030-\u0039]', '-', date) # 只保留數字部分
        date = re.split('-', date) # 將日期轉為List
        date = [x for x in date if x != ''] # 將空白資料移出list
        date = str(int(date[0]) + 1911) + str('{:0>2d}'.format(int(date[1]))) + str('{:0>2d}'.format(int(date[2]))) # 將日期轉為西元
    except IndexError:
        try:
            date_index = results.find('文日期') # 找出日期位置
            date = results_lst[results[:date_index].count('|')] # 發文日期
            date = re.sub(u'[^\u0030-\u0039]', '-', date) # 只保留數字部分
            date = re.split('-', date) # 將日期轉為List
            date = [x for x in date if x != ''] # 將空白資料移出list
            date = str(int(date[0]) + 1911) + str('{:0>2d}'.format(int(date[1]))) + str('{:0>2d}'.format(int(date[2]))) # 將日期轉為西元
        except:
            date = ''
    except:
        date = ''

    # 發文字號
    try:
        documentId = re.findall(r'發文字號[^\w\s](.*?)[^\w\s]', results) # 發文字號
        documentId = documentId[0]
    except IndexError:
        try:
            documentId_index = results.find('文字號') # 找出發文字號
            documentId = results_lst[results[:documentId_index].count('|')] # 發文字號
            documentId = re.split('[：]', documentId)[1] # 發文字號用冒號切開
        except:
            documentId = ''
    except:
        documentId = ''

    # 主旨
    try:
        subject = re.findall(r'主旨[^\w\s](.*?)[^\w\s]說明[^\w\s]', results_txt) # 主旨
        subject = subject[0]
        subject = re.sub(r'債耀', '債權', subject) # 
        subject = re.sub(r'命今', '命令', subject) #
    except IndexError:
        try:
            subject_index = results.find('主旨') # 找出主旨
            description_idnex = results.find('|說明') # 找出說明
            subject = ''.join(results_lst[results[:subject_index].count('|'):results[:description_idnex].count('|') + 1])
            subject = re.sub(r'債耀', '債權', subject) # 
            subject = re.sub(r'命今', '命令', subject) #
            subject = re.sub(r'主旨：', '', subject) # 
        except:
            subject = ''
    except:
        subject = ''

    # 義務人 / 債務人
    try:
        debtor = re.findall(r'[發受債義存][^權耀][人]([\w\s]{1,20})[（( ]', results_txt) # 找出 發票人　債務人　義務人　受刑人
        debtor = debtor[0]
    except IndexError:
        try:
            debtor = re.findall(r'[發受債義存][^權耀][人]([\w\s]{1,20})[，,]', results_txt) # 找出 發票人　債務人　義務人　受刑人
            debtor = debtor[0]
        except:
            try:
                debtor = re.findall(r'[發受債義存][^權耀][人]([\w\s]{3})', results_txt) # 找出 發票人　債務人　義務人　受刑人
                debtor = debtor[0]
            except:
                debtor = ''
    except:
        debtor = ''

    # 帳號
    try:
        account = re.findall(r'[號][:：]([0][0-9]{11,20})[^\d]', results_txt) # 找出帳號
        account = account[0]
    except:
        account = ''

    # 身分證統一編號
    try:
        idNumber = re.findall(r'[a-zA-Z][12][0-9]{8}', results_txt) # 找出身分證
        idNumber = idNumber[0]
    except:
        idNumber = ''

    # 營運事業編號
    try:
        businessNumber =  re.findall(r'[號][:：]([0-9]{7,8})[^\d]', results_txt) # 找出營運事業編號(公司統編)
        businessNumber = businessNumber[0]
    except:
        businessNumber = ''

    """將辨識結果寫入json"""

    rec_result.value.append({'name':'發文機關', 'text':authority})
    rec_result.value.append({'name':'發文日期', 'text':date})
    rec_result.value.append({'name':'發文字號', 'text':documentId})
    rec_result.value.append({'name':'主旨', 'text':subject})
    rec_result.value.append({'name':'義務人_債務人', 'text':debtor})
    rec_result.value.append({'name':'帳號', 'text':account})
    rec_result.value.append({'name':'身分證統一編號', 'text':idNumber})
    rec_result.value.append({'name':'營利事業編號', 'text':businessNumber})
    rec_result.status = '0' # 完成
    rec_result.msg = '辨識完成'
    
    #convert to JSON string
    jsonStr = json.dumps(rec_result.__dict__, ensure_ascii=False)
    # lock.release()  # 解除鎖定
    return jsonStr

async def general_affair_doc_rec(img):
    """總務公文辨識"""
    rec_result = DocModel() # Create Object
    rec_result.documentType = 'GA1'
    rec_result.documentName = '總務公文'
    
    try:
        res = ocr.ocr(img) # 開始辨識
    except Exception as err:
        rec_result.status = '-1' # 異常
        rec_result.msg = '辨識過程失敗，請至log查看原因。' # 辨識失敗
        return json.dumps(rec_result.__dict__, ensure_ascii=False)

    """正規辨識結果"""
    results = ''
    results_lst = [res[i][1][0] for i in range(len(res))] # 將辨識出來的陣列儲存起來
    results = '|'.join(results_lst) # 將辨識出來的文字用「|」join起來
    results_txt = ''.join(results_lst) # 全文結果，無分割

    # 發文機關
    try:
        new_results = '|' + results # 避免關鍵字出現在第一排
        authority = re.search(r'([|][\w\s]{1,25})(執行署|地方法院|公路局|檢察署|監理所|法務部|稅捐|交通部|移民署|內政部|稅務局|警察局|國防部|社團法人|中華民國)(.*?)[|]', new_results).group()
        authority = re.sub(r'[^\w\s]', '', authority) # 去除符號
        authority = re.sub(r'台', '臺', authority) # 去除符號
        authority = re.sub(r'臺南', '台南', authority) # 去除符號
    except IndexError:
        try:
            address_index = results.find('址：') # 找出地址位置
            authority = results_lst[results[:address_index].count('|')-1] # 發文機關，找出地址位於陣列第幾列之後取前一列
            authority = re.sub(r'台', '臺', authority) # 去除符號
            authority = re.sub(r'臺南', '台南', authority) # 去除符號
        except:
            authority = ''
    except:
        authority = ''

    # 發文日期
    try:
        date = re.findall(r'中華民國(\d*?年\d*?月\d*?日)', results_txt) # 日期
        date = date[0]
        date = re.sub(u'[^\u0030-\u0039]', '-', date) # 只保留數字部分
        date = re.split('-', date) # 將日期轉為List
        date = [x for x in date if x != ''] # 將空白資料移出list
        date = str(int(date[0]) + 1911) + str('{:0>2d}'.format(int(date[1]))) + str('{:0>2d}'.format(int(date[2]))) # 將日期轉為西元
    except IndexError:
        try:
            date_index = results.find('文日期') # 找出日期位置
            date = results_lst[results[:date_index].count('|')] # 發文日期
            date = re.sub(u'[^\u0030-\u0039]', '-', date) # 只保留數字部分
            date = re.split('-', date) # 將日期轉為List
            date = [x for x in date if x != ''] # 將空白資料移出list
            date = str(int(date[0]) + 1911) + str('{:0>2d}'.format(int(date[1]))) + str('{:0>2d}'.format(int(date[2]))) # 將日期轉為西元
        except:
            date = ''
    except:
        date = ''

    # 發文字號
    try:
        documentId = re.findall(r'發文字號[^\w\s](.*?)[^\w\s]', results) # 發文字號
        documentId = documentId[0]
    except IndexError:
        try:
            documentId_index = results.find('文字號') # 找出發文字號
            documentId = results_lst[results[:documentId_index].count('|')] # 發文字號
            documentId = re.split('[：]', documentId)[1] # 發文字號用冒號切開
        except:
            documentId = ''
    except:
        documentId = ''

    # 速別
    try:
        urgency = re.findall(r'速別[^\w\s](.*?)[^\w\s]', results) # 速別
        urgency = urgency[0]
    except IndexError:
        try:
            urgency_index = results.find('速別') # 找出速別
            urgency = results_lst[results[:urgency_index].count('|')] # 速別
            urgency = re.split('[：]', urgency)[1] # 速別用冒號切開
        except:
            urgency = ''
    except:
        urgency = ''

    # 主旨
    try:
        subject = re.findall(r'主旨[^\w\s](.*?)[^\w\s]說明[^\w\s]', results_txt) # 主旨
        subject = subject[0]
    except IndexError:
        try:
            subject_index = results.find('主旨') # 找出主旨
            description_index = results.find('|說明') # 找出說明
            subject = ''.join(results_lst[results[:subject_index].count('|'):results[:description_index].count('|') + 1])
        except:
            subject = ''
    except:
        subject = ''

    # 說明
    # 國字數字跟阿拉伯數字的對照 (為了把說明號從國字轉阿拉伯數字)
    zh_num = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
        
    # 說明的切分點
    split_delimiters = '一、|二、|三、|四、|五、|六、|七、|八、|九、|十、'

    # 選出主旨中提到的說明
    def choose_target_des(idx: list, each_des: list) -> str:
        '''
        # TODO
        e.g. output: '一、xxxxx二、sdddd'
        '''
        target_idx = [zh_num[i] for i in idx]
        target_des = [each_des[i-1] for i in target_idx]
        target_des = [(idx[i] + '、' + target_des[i]) for i in range(len(idx))]
        target_des = ''.join(target_des)
        return target_des
    
    # 完整說明
    full_description = re.findall(r'說明[^\w\s](.*?)[^\w\s]正本[^\w\s]', results_txt)

    if not full_description:
        target_des = ''
    else:
        which_description = re.findall(r'說明[\S]', subject) # subject 可替換成 subject1, 2, 3測試
        if not which_description or which_description[0][-1] not in zh_num.keys():
            target_des = full_description[0]
        else: 
            which_description_idx = [x[-1] for x in which_description]
            each_description = re.split(split_delimiters, full_description[0])
            target_des = choose_target_des(which_description_idx, each_description)

    description = target_des

    # 附件 attached_file
    if results_txt.rfind('附件') > results_txt.rfind('說明'):
        attached_file_idx = results_txt.rfind('附件') + 2
        attached_file = results_txt[attached_file_idx:]
    else: attached_file = ''

    """將辨識結果寫入json"""

    rec_result.value.append({'name':'發文機關', 'text':authority})
    rec_result.value.append({'name':'發文日期', 'text':date})
    rec_result.value.append({'name':'發文字號', 'text':documentId})
    rec_result.value.append({'name':'速別', 'text':urgency})
    rec_result.value.append({'name':'主旨', 'text':subject})
    rec_result.value.append({'name':'說明', 'text':description})
    rec_result.value.append({'name':'附件', 'text':attached_file})

    rec_result.status = '0' # 完成
    rec_result.msg = '辨識完成'
    
    #convert to JSON string
    jsonStr = json.dumps(rec_result.__dict__, ensure_ascii=False)
    return jsonStr
