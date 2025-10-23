import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_valid_mask_and_spans(K, D, dim, balance=0.0):
    w, h = dim
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R=np.eye(3), P=new_K, size=(w, h), m1type=cv2.CV_32FC1
    )

    white_img = np.full((h, w, 3), 255, dtype=np.uint8)
    undistorted = cv2.remap(white_img, map1, map2, interpolation=cv2.INTER_LINEAR)
    valid_mask = np.any(undistorted != [0, 0, 0], axis=-1).astype(np.uint8) * 255

    # ✅ 取得主點中心座標
    cx, cy = int(round(new_K[0, 2])), int(round(new_K[1, 2]))

    vertical = valid_mask[:, cx]
    y_indices = np.where(vertical > 0)[0]
    y_start, y_end = (y_indices[0], y_indices[-1]) if len(y_indices) > 0 else (-1, -1)

    horizontal = valid_mask[cy, :]
    x_indices = np.where(horizontal > 0)[0]
    x_start, x_end = (x_indices[0], x_indices[-1]) if len(x_indices) > 0 else (-1, -1)

    result = {
        'balance': balance,
        'cx': cx, 'cy': cy,
        'x_line_valid_y': (y_start, y_end),
        'y_line_valid_x': (x_start, x_end)
    }

    return valid_mask, result

# 主程式
DIM = (640, 480)
K = np.array([[2.94104211e+02, 1.00501200e-01, 3.17351129e+02],
              [0.00000000e+00, 3.90626759e+02, 2.44645859e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
D = np.array([[-0.15439177], [0.45612835], [-0.79521684], [0.46727377]])

balances = [0.0, 0.3, 0.6, 0.9, 1.0]
fig, axes = plt.subplots(1, len(balances), figsize=(20, 4))

for i, b in enumerate(balances):
    mask, r = compute_valid_mask_and_spans(K, D, DIM, balance=b)
    axes[i].imshow(mask, cmap='gray')
    axes[i].set_title(f"balance={b}\ncx,cy={r['cx']},{r['cy']}\n"
                      f"x={r['cx']} y:{r['x_line_valid_y'][0]}~{r['x_line_valid_y'][1]}\n"
                      f"y={r['cy']} x:{r['y_line_valid_x'][0]}~{r['y_line_valid_x'][1]}")
    axes[i].axvline(r['cx'], color='red', linestyle='--')
    axes[i].axhline(r['cy'], color='blue', linestyle='--')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
