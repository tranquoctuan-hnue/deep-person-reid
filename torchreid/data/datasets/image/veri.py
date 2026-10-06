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
