# -*- coding:utf-8 -*-

"""辨識結果class"""
class RecResultModel(object):
    def __init__(self):
        self.status = '' # 結果 0:處理中 / 1:完成 / -1:異常
        self.msg = '' # 如果異常紀錄訊息
        self.result = [] # 文件類型

"""辨識結果class / 公版 / 辨識前紀錄"""
class DocModel(object):
    def __init__(self):
        self.documentType = '' # 文件類型
        self.documentName = '' # 檔案名稱
        self.status = '' # 狀態 0:處理中 / 1:完成 / -1:異常
        self.msg = '' # 如果異常紀錄訊息
        self.value = [] # 欄位and值

# """存管公文辨識結果class"""
# class DepositoryDocModel(object):
#     def __init__(self):
#         self.documentType = 'LE2/LE3' # 文件類型
#         self.documentName = '' # 檔案名稱
#         self.status = '' # 狀態 0:處理中 / 1:完成 / -1:異常
#         self.msg = '' # 如果異常紀錄訊息
#         self.value = [] # 欄位and值

# """總務公文辨識結果class"""
# class GeneralAffairsDocModel(object):
#     def __init__(self):
#         self.documentType = 'GA1' # 文件類型
#         self.documentName = '' # 檔案名稱
#         self.status = '' # 狀態 0:處理中 / 1:完成 / -1:異常
#         self.msg = '' # 如果異常紀錄訊息
#         self.value = [] # 欄位and值

