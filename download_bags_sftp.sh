#!/bin/bash

# Step 1: 下載
sftp anonymous@140.113.148.83 <<EOF
cd arg-projectfile-download/opencv_cuda
get -r bags
exit
EOF

# Step 2: 取得目前登入帳號
MYUSER=$(whoami)

# Step 3: 將下載的檔案擁有權還給自己
sudo chown -R $MYUSER:$MYUSER bags

# Step 4: 可選 - 加強存取權限
chmod -R u+rwX,go+rX bags

echo "✅ 下載與權限修正完成！"
