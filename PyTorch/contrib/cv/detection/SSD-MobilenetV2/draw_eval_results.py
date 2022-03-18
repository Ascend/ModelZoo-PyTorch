# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import cv2
import pandas as pd
import os

eval_result_file = sys.argv[1]
image_dir = sys.argv[2]
output_dir = sys.argv[3]
threshold = float(sys.argv[4])

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

r = pd.read_csv(eval_result_file, delimiter=" ", names=["ImageID", "Prob", "x1", "y1", "x2", "y2"])
r['x1'] = r['x1'].astype(int)
r['y1'] = r['y1'].astype(int)
r['x2'] = r['x2'].astype(int)
r['y2'] = r['y2'].astype(int)


for image_id, g in r.groupby('ImageID'):
    image = cv2.imread(os.path.join(image_dir, image_id + ".jpg"))
    for row in g.itertuples():
        if row.Prob < threshold:
            continue
        cv2.rectangle(image, (row.x1, row.y1), (row.x2, row.y2), (255, 255, 0), 4)
        label = f"{row.Prob:.2f}"
        cv2.putText(image, label,
                    (row.x1 + 20, row.y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imwrite(os.path.join(output_dir, image_id + ".jpg"), image)
print(f"Task Done. Processed {r.shape[0]} bounding boxes.")