import numpy as np
import time
from nvjpeg import NvJpeg
from cylindrical_stitching_core import TripletStitcher

def worker(input_q, output_q, params):
    nvj = NvJpeg()
    stitcher = TripletStitcher(**params)
    jpeg_quality = params.get('jpeg_quality', 90)
    while True:
        job = input_q.get()
        if job is None:
            break
        idx = job['idx']
        imgs_jpeg = job['imgs']
        global_M_L = job['global_M_L']
        global_M_R = job['global_M_R']
        frame_id   = job['frame_id']
        t0 = time.time()
        try:
            # ========== decode ===============
            imgs = [nvj.decode(jpeg) for jpeg in imgs_jpeg]
            t1 = time.time()
            # ========== stitch ===============
            pano, new_M_L, new_M_R = stitcher.stitch(imgs, frame_id, global_M_L, global_M_R)
            t2 = time.time()
            # ========== encode ===============
            pano_jpeg = nvj.encode(pano, jpeg_quality)
            t3 = time.time()
            # ========== put to output_q ===============
            output_q.put({'idx': idx, 'result': pano_jpeg, 'frame_id': frame_id,
                          'new_M_L': new_M_L, 'new_M_R': new_M_R})
            t4 = time.time()
            # print(
            #     f"[WORKER] Frame {frame_id} done: "
            #     f"decode={t1-t0:.3f}s, "
            #     f"stitch={t2-t1:.3f}s, "
            #     f"encode={t3-t2:.3f}s, "
            #     f"put={t4-t3:.3f}s, "
            #     f"TOTAL={t4-t0:.3f}s"
            # )
        except Exception as e:
            output_q.put({'idx': idx, 'result': None, 'frame_id': frame_id,
                          'new_M_L': None, 'new_M_R': None, 'err': str(e)})
            # print(f"[WORKER] Frame {frame_id} ERROR: {e}")