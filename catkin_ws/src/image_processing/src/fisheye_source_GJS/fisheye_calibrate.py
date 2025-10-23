import numpy as np
import cv2
import glob
import sys
import os

# 設定棋盤格內角點的尺寸 (橫向, 縱向)
CHECKERBOARD = (8, 6)
# 定義角點搜尋的終止條件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 準備棋盤格的三維座標 (例如 (0,0,0), (1,0,0), ..., (3,2,0))
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
# objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

square_size = 25  # mm
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

# 儲存所有圖片的3D點與對應的2D角點
_objpoints = []  # 每張圖片對應的3D棋盤格點
_imgpoints = []  # 每張圖片中檢測到的2D角點

# 讀取所有校正圖片（請確認圖片路徑正確）
images = glob.glob('cali_fisheye_left/*.png')
# images = glob.glob('cali_fisheye_mid/*.png')
# images = glob.glob('cali_fisheye_right/*.png')
if not images:
    print("錯誤：找不到 fisheye_source 資料夾下的 jpg 圖片")
    sys.exit(1)

print("讀取到 %d 張圖片" % len(images))

# 處理每張圖片
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print("無法讀取圖片：", fname)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 尋找棋盤格角點
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if ret:
        print("成功在圖片 {} 中找到棋盤格角點".format(fname))
        _objpoints.append(objp)
        # 進一步優化角點位置
        corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
        _imgpoints.append(corners2)
        
        # 繪製角點並顯示
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    else:
        print("在圖片 {} 中未找到棋盤格角點".format(fname))

cv2.destroyAllWindows()

print("總共成功檢測到 %d 張圖片的角點" % len(_objpoints))
if len(_objpoints) == 0:
    print("錯誤：沒有檢測到任何棋盤格角點，請檢查圖片內容與 CHECKERBOARD 設定")
    sys.exit(1)

# 初始化魚眼相機標定所需參數
N_OK = len(_objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

# 進行魚眼相機標定
try:
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        _objpoints,
        _imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
        criteria
    )
except cv2.error as e:
    print("OpenCV 魚眼標定時發生錯誤:")
    print(e)
    sys.exit(1)

print("標定重投影誤差 (RMS):", rms)
print("相機內參矩陣 K:")
# print(K)
# Add "," to all elements in K
def format_matrix(matrix):
    return np.array2string(matrix, separator=", ")
K_str = format_matrix(K)
print(K_str)
print("魚眼畸變參數 D (k1, k2, k3, k4):")
# print(D)
D_str = format_matrix(D)
print(D_str)

# 提取焦距 (單位：像素)
fx = K[0, 0]
fy = K[1, 1]
print("焦距 fx:", fx)
print("焦距 fy:", fy)

# 決定存檔路徑：和校正圖片同資料夾
output_folder = os.path.dirname(images[0])
output_path = os.path.join(output_folder, "fisheye_calib_result.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write("魚眼相機標定結果\n")
    f.write("====================\n")
    f.write(f"標定重投影誤差 (RMS): {rms:.6f}\n\n")
    f.write("相機內參矩陣 K:\n")
    f.write(K_str + "\n\n")
    f.write("魚眼畸變參數 D (k1, k2, k3, k4):\n")
    f.write(D_str + "\n\n")
    f.write(f"焦距 fx: {fx:.6f}\n")
    f.write(f"焦距 fy: {fy:.6f}\n")
    f.write("====================\n")

print(f"\n結果已寫入 {output_path}")