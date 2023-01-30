# -*- coding:utf-8 -*-
from fastapi import APIRouter, File, Form, BackgroundTasks, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address

import asyncio
import os
import cv2
from PIL import Image
import io
import json
import numpy as np
import traceback
import threading
# from multiprocessing import Pool
# import multiprocessing as mp
# import concurrent
# import queue
import time
import random

from modules.create_result_data import write_to_file

# limiter = Limiter(key_func=get_remote_address)
# lock = threading.Lock()  # 建立 Lock

router = APIRouter(
    prefix='/rec',
    tags=['recognize']
)

class RecTask(BaseModel):
    statusCode: int
    statusDesc: str

class RecResult(BaseModel):
    Status: int
    Msg: str

def background_task(post_seq_no, img, post_case_type):
    # p = Pool(5)
    time.sleep(random.randint(3, 5))
    # lock.acquire()
    # p.apply_async(write_to_file, (post_seq_no, img, post_case_type,))
    write_to_file(post_seq_no, img, post_case_type)
    # p.close()
    # lock.release()
    # return


@router.post("/doc")
# @limiter.limit("5/minute")
async def add_rec_task(
    request: Request, background_tasks: BackgroundTasks, secs: int = 10,
    file: bytes = File(), case_type: str = Form(), seq_no: str = Form()
):
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
        # p_job = mp.Process(
        #     target=write_to_file,
        #     args=(post_seq_no, img, post_case_type),
        #     name=f't_{post_seq_no}'
        # )
        # p_job.daemon = True
        # p_job.start()
        # time.sleep(3)

        t_job = threading.Thread(
            target=background_task,
            args=(post_seq_no, img, post_case_type),
            name=f't_{post_seq_no}'
        )
        # t_job.daemon = True
        t_job.start()
        time.sleep(3)
        
        # background_tasks.add_task(
        #     run_in_process, post_seq_no, img, post_case_type
        # )

        response = {
            "statusCode" : 0,
            "statusDesc" : f"已確認收到任務請求，案件號{post_seq_no}"
        }
        rec_task = RecTask(**response)
        json_compatible_item_data = jsonable_encoder(rec_task)
        headers = {"Content-Type": "application/json;charset=UTF-8"}
        return JSONResponse(content=json_compatible_item_data, headers=headers)
    except Exception as err:
        print(err)
        response = {
            "statusCode" : -2,
            "statusDesc" : f"建立線程失敗，案件號{str(err)}"
        }
        rec_task = RecTask(**response)
        json_compatible_item_data = jsonable_encoder(rec_task)
        headers = {"Content-Type": "application/json;charset=UTF-8"}
        return JSONResponse(content=json_compatible_item_data, headers=headers)

@router.get("/doc")
async def rec_result(seq_no: str):
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
