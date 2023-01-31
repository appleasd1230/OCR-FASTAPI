# -*- coding:utf-8 -*-
import uvicorn
from fastapi import FastAPI, HTTPException, File, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from a2wsgi import ASGIMiddleware
from paddleocr import PaddleOCR, draw_ocr
import cv2
import string
import os

from PIL import Image
import io
import json
import numpy as np
import traceback
import threading
import time
import random

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
fh = logging.FileHandler(filename='./server.log')
formatter = logging.Formatter(
    "%(asctime)s - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s"
)

ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch) # 將日誌輸出至螢幕
logger.addHandler(fh) # 將日誌輸出至文件

logger = logging.getLogger(__name__)

app = FastAPI()


origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.doc_ocr = PaddleOCR(
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
    cpu_threads=12
)

@app.middleware("http")
async def log_requests(request, call_next):
    idem = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    logger.info(f"rid={idem} start request path={request.url.path}")
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    formatted_process_time = '{0:.2f}'.format(process_time)
    logger.info(f"rid={idem} completed_in={formatted_process_time}ms status_code={response.status_code}")
    
    return response


def background_task(post_seq_no, img, post_case_type):
    from modules.create_result_data import write_to_file
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    time.sleep(random.randint(3, 5))
    write_to_file(post_seq_no, img, post_case_type, app.doc_ocr)


@app.post("/rec/doc")
# @limiter.limit("5/minute")
async def add_rec_task(
    request: Request, background_tasks: BackgroundTasks, secs: int = 10,
    file: bytes = File(), case_type: str = Form(), seq_no: str = Form()
):
    class RecTask(BaseModel):
        statusCode: int
        statusDesc: str

    try:
        post_case_type = case_type
        post_image = file
        post_seq_no = seq_no

        stream = io.BytesIO(post_image)
        pil_img = Image.open(stream)
        
        if pil_img.mode == '1':
            pil_img.convert('L')
            buf = io.BytesIO()
            pil_img.save(buf, format='JPEG')
            post_image = buf.getvalue()
            
        stream.close()

        img = cv2.imdecode(np.frombuffer(post_image, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        response = {
            "statusCode" : 0,
            "statusDesc" : f"解析post資料時錯誤，請確認json內容無誤"
        }
        rec_task = RecTask(**response)
        json_compatible_item_data = jsonable_encoder(rec_task)
        raise HTTPException(status_code=200, detail=json_compatible_item_data)

    try:
        t_job = threading.Thread(
            target=background_task,
            args=(post_seq_no, img, post_case_type),
            name=f't_{post_seq_no}'
        )
        time.sleep(3)
    

        response = {
            "statusCode" : 0,
            "statusDesc" : f"已確認收到任務請求，案件號{post_seq_no}"
        }
        rec_task = RecTask(**response)
        json_compatible_item_data = jsonable_encoder(rec_task)
        headers = {"Content-Type": "application/json;charset=UTF-8"}
        return JSONResponse(content=json_compatible_item_data, headers=headers)
    except Exception as err:
        response = {
            "statusCode" : -2,
            "statusDesc" : f"建立線程失敗，案件號{str(err)}"
        }
        rec_task = RecTask(**response)
        json_compatible_item_data = jsonable_encoder(rec_task)
        headers = {"Content-Type": "application/json;charset=UTF-8"}
        return JSONResponse(content=json_compatible_item_data, headers=headers)

@app.get("/rec/doc")
async def rec_result(seq_no: str):
    class RecResult(BaseModel):
        Status: int
        Msg: str

    try:
        seq_no = seq_no
        path = f'data/{seq_no}.json'
        if os.path.exists(path):
            f = open(path, encoding='utf-8')
            data = json.load(f)
            json_compatible_item_data = jsonable_encoder(data)
            headers = {"Content-Type": "application/json;charset=UTF-8"}
            return JSONResponse(content=json_compatible_item_data, headers=headers)
        else:
            response = {
                "Status" : -2,
                "Msg" : f"查無此件號{seq_no}"
            }
            rec_task = RecResult(**response)
            json_compatible_item_data = jsonable_encoder(rec_task)
            headers = {"Content-Type": "application/json;charset=UTF-8"}
            return JSONResponse(content=json_compatible_item_data, headers=headers)
    except Exception as err:
        response = {
            "Status" : -1,
            "Msg" : f"取得辨識結果失敗，案件號{seq_no}"
        }
        rec_task = RecResult(**response)
        json_compatible_item_data = jsonable_encoder(rec_task)
        headers = {"Content-Type": "application/json;charset=UTF-8"}
        return JSONResponse(content=json_compatible_item_data, headers=headers)

@app.get("rec/demo")
async def rec_demo():
    class RecResult(BaseModel):
        Status: int
        Msg: str

    images = glob.glob('demo/input/*.jpg') # 取得所有需要的照片路徑
    for img_path in images:
        file_name = os.path.basename(img_path)
        # org_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        # print(org_img)
        org_img = sharpen(img_path)

        stream = io.BytesIO(org_img)
        pil_img = Image.open(stream)
        
        if pil_img.mode == '1':
            pil_img.convert('L')
            buf = io.BytesIO()
            pil_img.save(buf, format='JPEG')
            org_img = buf.getvalue()
            
        stream.close()

        org_img = cv2.imdecode(np.frombuffer(org_img, np.uint8), cv2.IMREAD_COLOR)
        # org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)

        results = {
            "res": [],
            "boxes": [],
            "txts": [],
            "scores": []
        }
        res = ocr.ocr(org_img)
        for row in res:
            results['res'].append({
                'points': [ [int(p[0]), int(p[1])] for p in row[0]],
                'text': row[1][0],
                'score': round(float(row[1][1]), 2)
            })
        results['boxes'] = sorted_boxes(np.asarray([res[i][0] for i in range(len(res))]))
        results['txts'] = [res[i][1][0] for i in range(len(res))]
        results['scores'] = [res[i][1][1] for i in range(len(res))]

        drop_score = 0.5
        draw_img = draw_ocr_box_txt(
                    Image.fromarray(org_img),
                    results['boxes'],
                    results['txts'],
                    results['scores'],
                    drop_score=drop_score,
                    font_path='doc/fonts/simfang.ttf')

                    
        cv2.imencode('.jpeg', draw_img)[1].tofile('demo/output/' + file_name)

    response = {
        "Status" : 0,
        "Msg" : f"辨識完成"
    }
    rec_task = RecResult(**response)
    json_compatible_item_data = jsonable_encoder(rec_task)
    headers = {"Content-Type": "application/json;charset=UTF-8"}
    return JSONResponse(content=json_compatible_item_data, headers=headers)

# IIS 需要轉換將ASGI轉為WSGI
wsgi_app = ASGIMiddleware(app)

# IIS 上的版本底下兩行要拿掉
if __name__ == "__main__":
    uvicorn.run('main:app', workers=2)
