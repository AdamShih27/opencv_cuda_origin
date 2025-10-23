import numpy as np
from stitching_core import TripletStitcher
from horizon_core import FastHorizon
from mask_core import PanoHorizonMarker
import concurrent.futures

def worker(input_q, output_q, cfg):
    stitcher_params = cfg['stitcher']
    horizon_params  = cfg['horizon']
    marker_params   = cfg['marker']

    stitcher = TripletStitcher(**stitcher_params)
    marker   = PanoHorizonMarker(**marker_params)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            job = input_q.get()
            if job is None:
                break
            idx = job['idx']
            imgs = job['imgs']  # [imgL, imgM, imgR]
            global_M_L = job['global_M_L']
            global_M_R = job['global_M_R']
            frame_id   = job['frame_id']
            try:
                # 1. 三個 horizon 任務（各自 thread）
                def detect_horizon_and_scale(img):
                    fh = FastHorizon(**horizon_params)
                    return fh.detect_horizon_and_scale_points(img)

                # 三張圖同時跑
                horizon_scale_futures = [executor.submit(detect_horizon_and_scale, img) for img in imgs]
                
                # 2. 拼接任務
                stitch_future = executor.submit(
                    stitcher.stitch, imgs, frame_id, global_M_L, global_M_R)

                # 3. 等全部完成
                horizon_pts = [f.result() for f in horizon_scale_futures]
                pano, new_M_L, new_M_R, off = stitch_future.result()

                # 4. 畫點到 pano
                pano_mask = marker.mark(pano, horizon_pts, global_M_L, global_M_R, off)

                # 5. output
                output_q.put({
                    'idx': idx,
                    'result': pano,
                    'pano_mask': pano_mask,
                    'frame_id': frame_id,
                    'new_M_L': new_M_L,
                    'new_M_R': new_M_R,
                    'horizon_pts': horizon_pts, 
                })
            except Exception as e:
                import traceback
                print(f"[Worker] Error in frame {frame_id}: {e}")
                traceback.print_exc()