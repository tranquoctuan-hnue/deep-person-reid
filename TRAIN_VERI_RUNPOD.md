# Huấn luyện OSNet trên VeRi-776 bằng RunPod

Hướng dẫn fine-tune `osnet_x1_0` (pretrained ImageNet) cho bài toán **vehicle re-identification** trên bộ VeRi-776, chạy trên GPU thuê của [RunPod](https://www.runpod.io/), dùng repo `deep-person-reid` (torchreid) này.

Tóm tắt các bước:

1. Thêm dataset VeRi vào torchreid (repo gốc chưa hỗ trợ).
2. Tạo file config cho VeRi.
3. Tạo pod trên RunPod.
4. Đưa code, dữ liệu và trọng số pretrained lên pod.
5. Cài môi trường.
6. Train, theo dõi tiến trình, train tiếp nếu bị ngắt.
7. Đánh giá và tải model về.

---

## 0. Thông số dữ liệu và lựa chọn chính

| Thông tin | Giá trị |
|---|---|
| Số ID | 576 (train) + 200 (test) |
| Số ảnh | 37.778 train / 1.678 query / 11.579 gallery |
| Số camera | 20 |
| Tỷ lệ W/H trung vị | ~1,14 (xe hơi rộng hơn cao, dao động 0,75–1,9) |
| Kích thước ảnh trung vị | ~221×192 px |
| Dung lượng | ~1,1 GB |

**Kích thước đầu vào: `256×256` (H×W).** Không dùng `256×128` (kích thước mặc định cho người): ảnh xe sẽ bị bóp ngang khoảng 2,3 lần. Nếu muốn khớp đúng tỷ lệ trung vị thì có thể thử `224×256`.

**Loss:** softmax (label smoothing) + triplet, dùng `RandomIdentitySampler` (mỗi batch gồm 16 ID × 4 ảnh khi batch = 64).

**Learning rate:** `0.00035` (amsgrad, cosine).

> ⚠️ Đã chạy thử với `lr=0.0015` (giá trị trong config OSNet gốc cho người) cùng triplet loss. Model **bị collapse** sau khoảng 500 iteration: `loss_t` kẹt ở `0.3000` (đúng bằng margin), `loss_x` kẹt ở `6.35` (≈ ln 576), acc ≈ 0. Với `lr=0.00035`, loss giảm đều và acc tăng bình thường.

---

## 1. Thêm dataset VeRi vào torchreid

> ✅ Mục 1 và 2 **đã có sẵn** trong fork `tranquoctuan-hnue/deep-person-reid`. Chỉ cần `git clone` (mục 4.1) là dùng được. Hai mục này chỉ để giải thích code đã thêm.

### 1.1. Tạo file `torchreid/data/datasets/image/veri.py`

```python
from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp

from ..dataset import ImageDataset


class VeRi(ImageDataset):
    """VeRi-776.

    Reference:
        Liu et al. A Deep Learning-Based Approach to Progressive Vehicle
        Re-identification for Urban Surveillance. ECCV 2016.

    Dataset statistics:
        - identities: 576 (train) + 200 (test).
        - images: 37778 (train) + 1678 (query) + 11579 (gallery).
        - cameras: 20.
    """
    dataset_dir = 'VeRi'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(VeRi, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # e.g. 0002_c002_00030600_0.jpg -> pid=2, camid=2
        pattern = re.compile(r'(\d+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(osp.basename(img_path)).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(osp.basename(img_path)).groups())
            assert 1 <= camid <= 20
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data
```

### 1.2. Đăng ký dataset

Trong `torchreid/data/datasets/image/__init__.py`, thêm dòng:

```python
from .veri import VeRi
```

Trong `torchreid/data/datasets/__init__.py`, sửa import và thêm `'veri'` vào `__image_datasets`:

```python
from .image import (
    GRID, PRID, CUHK01, CUHK02, CUHK03, MSMT17, CUHKSYSU, VIPeR, SenseReID,
    Market1501, DukeMTMCreID, University1652, iLIDS, VeRi
)
...
__image_datasets = {
    ...
    'cuhksysu': CUHKSYSU,
    'veri': VeRi
}
```

Hàm đánh giá của torchreid (`eval_market1501`) bỏ qua các ảnh gallery **cùng ID và cùng camera** với query. Đây cũng chính là protocol chuẩn image-to-image của VeRi-776, nên kết quả mAP/Rank-1 so sánh trực tiếp được với các bài báo.

---

## 2. Tạo file config `configs/osnet_x1_0_veri_256x256.yaml`

```yaml
model:
  name: 'osnet_x1_0'
  pretrained: True          # tự load trọng số ImageNet từ ~/.cache/torch/checkpoints

data:
  type: 'image'
  root: '/workspace/reid-data'
  sources: ['veri']
  targets: ['veri']
  height: 256
  width: 256
  workers: 8
  combineall: False
  transforms: ['random_flip', 'random_crop', 'color_jitter', 'random_erase']
  save_dir: '/workspace/logs/osnet_x1_0_veri_256x256'

sampler:
  train_sampler: 'RandomIdentitySampler'
  num_instances: 4

loss:
  name: 'triplet'
  softmax:
    label_smooth: True
  triplet:
    margin: 0.3
    weight_t: 1.0
    weight_x: 1.0

train:
  optim: 'amsgrad'
  lr: 0.00035
  weight_decay: 0.0005
  max_epoch: 80
  batch_size: 64
  fixbase_epoch: 0
  lr_scheduler: 'cosine'
  print_freq: 50

test:
  batch_size: 256
  dist_metric: 'euclidean'
  normalize_feature: False
  evaluate: False
  eval_freq: 10             # cứ 10 epoch thì đánh giá và lưu checkpoint một lần
  rerank: False
```

Lưu ý:
- Checkpoint **chỉ được lưu ở các epoch có đánh giá** (theo `eval_freq`) và ở epoch cuối. Nếu muốn lưu thường xuyên hơn (phòng khi pod bị ngắt) thì giảm `eval_freq`, ví dụ xuống 5.
- `save_dir` đặt trong `/workspace` để log và checkpoint không mất khi stop pod.

Nếu sửa config hoặc code ở local, commit rồi `git push` lên fork, sau đó chạy `git pull` trên pod.

---

## 3. Tạo pod trên RunPod

1. Vào **Pods → Deploy**.
2. **Chọn GPU.** OSNet nhỏ (~2,2M tham số, ~2 GFLOPs ở 256×256), không cần GPU đắt tiền:
   - **RTX 4090 (24 GB)** hoặc **RTX A5000 / A4500**: đủ cho batch 64, giá hợp lý.
   - Không cần A100/H100.
3. **Chọn template:** `RunPod PyTorch 2.x` (đã có CUDA, PyTorch, Python 3, `runpodctl`).
4. **Storage:**
   - *Container Disk*: 20 GB. Phần này **bị xoá khi stop pod**.
   - *Volume Disk*: ≥ 20 GB, mount tại `/workspace`. Phần này **được giữ lại** khi stop pod.
   - Nếu muốn dùng lại dữ liệu cho nhiều pod khác nhau thì tạo **Network Volume** rồi gắn vào pod.
5. **Expose ports** (trong *Edit Template*):
   - TCP `22` để SSH và `scp`.
   - HTTP `6006` nếu muốn xem TensorBoard.
6. Thêm SSH public key của bạn tại **Settings → SSH Public Keys** trước khi deploy.
7. Bấm **Deploy**, đợi pod chuyển sang trạng thái Running, rồi bấm **Connect** để lấy lệnh SSH, có dạng:
   ```bash
   ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
   ```

---

## 4. Đưa code, dữ liệu và trọng số lên pod

### 4.1. Code

Trên pod:

```bash
cd /workspace
git clone https://github.com/tranquoctuan-hnue/deep-person-reid.git
```

### 4.2. Dữ liệu VeRi (~1,1 GB)

Ở máy local, nén dataset:

```bash
cd ~/Downloads/archive
zip -rq VeRi.zip VeRi
```

**Cách A: `runpodctl`** (không cần SSH, pod đã cài sẵn công cụ này).
Ở local, cài `runpodctl` theo hướng dẫn trên GitHub `runpod/runpodctl`, rồi chạy:

```bash
runpodctl send VeRi.zip
# Lệnh in ra một mã, ví dụ: 1234-word-word-word
```

Trên pod:

```bash
mkdir -p /workspace/reid-data && cd /workspace/reid-data
runpodctl receive 1234-word-word-word
unzip -q VeRi.zip && rm VeRi.zip
```

**Cách B: `scp`** (cần đã expose TCP 22):

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 VeRi.zip root@<IP>:/workspace/reid-data/
```

Cấu trúc thư mục trên pod phải như sau:

```
/workspace/reid-data/VeRi/
├── image_train/   (37778 ảnh)
├── image_query/   (1678 ảnh)
├── image_test/    (11579 ảnh)
└── ...
```

### 4.3. Trọng số pretrained ImageNet

torchreid tự tải `osnet_x1_0_imagenet.pth` từ Google Drive bằng `gdown`. Trên server, bước này **hay bị lỗi do Google Drive giới hạn lượt tải**. Máy local đã có sẵn file này ở `~/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth` (~11 MB), nên gửi luôn lên pod cho chắc:

```bash
# local
runpodctl send ~/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth
# pod
mkdir -p ~/.cache/torch/checkpoints && cd ~/.cache/torch/checkpoints
runpodctl receive <mã>
```

`~/.cache` nằm trên container disk nên sẽ mất khi stop pod. Nên lưu thêm một bản trong `/workspace`, rồi mỗi lần khởi động lại pod thì copy sang:

```bash
mkdir -p /workspace/weights && cp ~/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth /workspace/weights/
# khi khởi động lại pod:
mkdir -p ~/.cache/torch/checkpoints && cp /workspace/weights/osnet_x1_0_imagenet.pth ~/.cache/torch/checkpoints/
```

---

## 5. Cài môi trường trên pod

```bash
cd /workspace/deep-person-reid

# opencv cần các thư viện hệ thống này (image RunPod thường không có sẵn)
apt-get update && apt-get install -y libgl1 libglib2.0-0 tmux

# PyTorch đã có sẵn trong template, KHÔNG cài lại torch
pip install -r requirements.txt

# Cài torchreid và build module Cython để đánh giá nhanh
pip install -e . --no-build-isolation
```

Kiểm tra lại:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from torchreid.metrics.rank_cylib.rank_cy import evaluate_cy; print('cython OK')"
python -c "import torchreid; d = torchreid.data.datasets.init_image_dataset('veri', root='/workspace/reid-data'); print(d.num_train_pids)"
# Kết quả mong đợi: True ..., cython OK, 576
```

Nếu dòng Cython báo lỗi, chạy `python setup.py build_ext --inplace`. Nếu vẫn không được thì vẫn train bình thường, chỉ có bước đánh giá chậm hơn (torchreid sẽ cảnh báo *"Cython evaluation ... is unavailable"*).

---

## 6. Huấn luyện

### 6.1. Chạy trong `tmux` để không bị dừng khi mất kết nối SSH

```bash
tmux new -s train
cd /workspace/deep-person-reid
python scripts/main.py \
    --config-file configs/osnet_x1_0_veri_256x256.yaml \
    --root /workspace/reid-data
```

- Thoát tmux mà vẫn để train chạy: `Ctrl+B` rồi `D`.
- Vào lại: `tmux attach -t train`.

Có thể ghi đè tham số ngay trên dòng lệnh mà không cần sửa file yaml. Ví dụ:

```bash
python scripts/main.py --config-file configs/osnet_x1_0_veri_256x256.yaml \
    --root /workspace/reid-data train.batch_size 128 train.lr 0.0007 data.workers 12
```

(Nếu tăng batch size gấp đôi thì có thể tăng lr theo tỷ lệ tương ứng, nhưng phải kiểm tra lại để chắc model không bị collapse như mục 0.)

### 6.2. Theo dõi

```bash
# log text
tail -f /workspace/logs/osnet_x1_0_veri_256x256/train.log-*

# GPU
watch -n 2 nvidia-smi

# TensorBoard (nếu đã expose port 6006)
tensorboard --logdir /workspace/logs --host 0.0.0.0 --port 6006
# Mở link "HTTP 6006" trong mục Connect của pod
```

Một dòng log bình thường trông như thế này (lấy từ lần chạy thử với batch 32, nên mỗi epoch có 1155 iteration; với batch 64 sẽ còn khoảng 590):

```
epoch: [1/80][500/1155]  ...  loss_t 0.5661 (1.7259)  loss_x 4.9929 (5.8166)  acc 0.0000 (4.2313)  lr 0.000350
```

Dấu hiệu train ổn: `loss_x` giảm dần từ ~6,4 và `acc` tăng lên.

**Dấu hiệu collapse:** `loss_t ≈ 0.3000` và `loss_x ≈ 6.35` kéo dài liên tục. Khi thấy vậy, dừng train và giảm lr.

### 6.3. Kết quả đánh giá

Cứ sau mỗi `eval_freq` epoch, log sẽ in ra:

```
** Results **
mAP: xx.x%
CMC curve
Rank-1  : xx.x%
Rank-5  : xx.x%
...
Checkpoint saved to "/workspace/logs/osnet_x1_0_veri_256x256/model/model.pth.tar-10"
Best checkpoint saved to "/workspace/logs/osnet_x1_0_veri_256x256/model/model-best.pth.tar"
```

Các file trong `.../model/`:

| File | Ý nghĩa |
|---|---|
| `model.pth.tar-N` | Checkpoint của epoch N (mỗi lần đánh giá lưu một file) |
| `model-last.pth.tar` | Checkpoint mới nhất, ghi đè ở mỗi lần lưu |
| `model-best.pth.tar` | Checkpoint có **mAP cao nhất** tính đến hiện tại |

Dòng `Best checkpoint saved ...` chỉ xuất hiện khi mAP vượt mức cao nhất trước đó. Cuối quá trình train, log in thêm `Best mAP: xx.x%`.

### 6.4. Train tiếp khi bị ngắt

```bash
python scripts/main.py --config-file configs/osnet_x1_0_veri_256x256.yaml \
    --root /workspace/reid-data \
    model.resume /workspace/logs/osnet_x1_0_veri_256x256/model/model-last.pth.tar
```

`resume` khôi phục cả optimizer, scheduler và epoch bắt đầu. Khi train tiếp trong cùng `save_dir`, mAP của `model-best.pth.tar` cũ được đọc lại, nên bản best chỉ bị ghi đè khi có mAP thực sự cao hơn.

---

## 7. Đánh giá và tải model về

### 7.1. Chỉ đánh giá một checkpoint

```bash
python scripts/main.py --config-file configs/osnet_x1_0_veri_256x256.yaml \
    --root /workspace/reid-data \
    model.load_weights /workspace/logs/osnet_x1_0_veri_256x256/model/model-best.pth.tar \
    test.evaluate True
```

Có thể thêm `test.rerank True` để bật re-ranking. Cách này thường tăng mAP, nhưng chậm hơn.

### 7.2. Tải về máy local

```bash
# pod
runpodctl send /workspace/logs/osnet_x1_0_veri_256x256/model/model-best.pth.tar
# local
runpodctl receive <mã>
```

Hoặc dùng `scp`:

```bash
scp -P <PORT> root@<IP>:/workspace/logs/osnet_x1_0_veri_256x256/model/model-best.pth.tar .
```

### 7.3. Dùng model để trích đặc trưng

Nhớ đặt `image_size=(256, 256)` cho khớp với lúc train. Giá trị mặc định của FeatureExtractor là `(256, 128)`.

```python
from torchreid.utils import FeatureExtractor

extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='model-best.pth.tar',
    image_size=(256, 256),
    device='cuda'
)
features = extractor(['img1.jpg', 'img2.jpg'])  # tensor (2, 512)
```

---

## 8. Tiết kiệm chi phí

- **Stop pod ngay khi train xong.** Nếu pod còn chạy thì GPU vẫn bị tính tiền. Khi đã stop, chỉ còn tính phí lưu trữ Volume Disk, khá rẻ.
- **Terminate** pod khi không cần nữa, vì volume vẫn tiếp tục tính phí. Nhớ tải checkpoint về trước khi terminate.
- Có thể chạy thử nhanh `train.max_epoch 2 test.eval_freq 1` để kiểm tra toàn bộ pipeline trước khi chạy đủ 80 epoch.
- *Community Cloud* rẻ hơn *Secure Cloud*, đổi lại có thể khó thuê được GPU vào giờ cao điểm.

---

## 9. Lỗi thường gặp

| Lỗi | Nguyên nhân / cách xử lý |
|---|---|
| `ValueError: Invalid dataset name ... 'veri'` | Chưa đăng ký VeRi trong `torchreid/data/datasets/__init__.py` (xem mục 1.2) |
| `RuntimeError: ".../VeRi" is not found` | Sai `--root`. Thư mục phải là `<root>/VeRi/image_train` |
| `ImportError: libGL.so.1` | `apt-get install -y libgl1 libglib2.0-0` |
| Treo hoặc lỗi khi tải pretrained (gdown) | Copy sẵn `osnet_x1_0_imagenet.pth` vào `~/.cache/torch/checkpoints/` (mục 4.3) |
| `loss_t` kẹt ở 0.3000, `loss_x` kẹt ở ~6.35 | Model bị collapse do lr quá cao, cần giảm lr (0.00035 với batch 64) |
| `CUDA out of memory` | Giảm `train.batch_size` (giữ chia hết cho `num_instances=4`) hoặc giảm `test.batch_size` |
| `DataLoader worker ... killed` / hết RAM | Giảm `data.workers` |
| Mất log hoặc checkpoint sau khi stop pod | `save_dir` không nằm trong `/workspace`. Phải lưu vào Volume Disk |
