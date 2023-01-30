# -*- coding:utf-8 -*-
from paddleocr import PaddleOCR, draw_ocr
# from pydantic import BaseSettings

class Settings(object):
    def __init__(self):
        """paddle parameter"""
        self.ocr = PaddleOCR(
            det_model_dir=r'ml_models/ppocv_server_v2.0/ch_ppocr_server_v2.0_det_infer/',
            rec_model_dir=r'ml_models/ppocv_server_v2.0/ch_ppocr_server_v2.0_rec_infer/',
            # det_model_dir=r'app/ml_models/ch_PP-OCRv3/ch_PP-OCRv3_det_infer/',
            # rec_model_dir=r'app/ml_models/ch_PP-OCRv3/ch_PP-OCRv3_rec_infer/',
            rec_char_dict_path=r'dict/cathay_dict.txt',
            cls_model_dir=r'ml_models/ppocr_mobile_v2.0/ch_ppocr_mobile_v2.0_cls_infer/',
            use_gpu=False,
            det_db_thresh = 0.1,
            det_db_box_thresh=0.1,
            use_angle_cls=True,
            use_space_char=True,
            lang="ch",
        )
