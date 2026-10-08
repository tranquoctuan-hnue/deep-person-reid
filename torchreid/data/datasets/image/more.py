from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp

from ..dataset import ImageDataset


class MoRe(ImageDataset):
    """MoRe (Motorcycle Re-identification).

    Reference:
        Figueiredo et al. MoRe: A Large-Scale Motorcycle Re-Identification
        Dataset. WACV 2021.

    The official split (``train_files.txt`` / ``test_files.txt``) is used.
    Query and gallery are built from the test split:
        - query: the first image of every (identity, camera).
        - gallery: all test images plus the distractors (pid=-1).
    Gallery images with the same identity and camera as the query are
    discarded at evaluation, so every query is matched against the other
    camera of its pair.

    Dataset statistics:
        - identities: 1913 (train) + 1914 (test).
        - images: 7032 (train) + 7109 (test) + 3478 (distractors).
        - cameras: 12 (6 pairs x camA/camB).
    """
    dataset_dir = 'MoRe'
    _distractor_camid = 12

    def __init__(self, root='', use_distractors=True, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow the original folder name inside MoRe/
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.dataset_dir, 'MoRe - Final Version')
        if osp.isdir(data_dir):
            self.data_dir = data_dir

        self.train_list = osp.join(self.data_dir, 'train_files.txt')
        self.test_list = osp.join(self.data_dir, 'test_files.txt')
        self.distractor_dir = osp.join(self.data_dir, 'Distractors')

        required_files = [self.data_dir, self.train_list, self.test_list]
        if use_distractors:
            required_files.append(self.distractor_dir)
        self.check_before_run(required_files)

        train = self.process_list(self.train_list, relabel=True)
        test = self.process_list(self.test_list, relabel=False)

        query = []
        seen = set()
        for item in sorted(test):
            _, pid, camid = item
            if (pid, camid) not in seen:
                seen.add((pid, camid))
                query.append(item)
        gallery = list(test)
        if use_distractors:
            gallery += self.process_distractors(self.distractor_dir)

        super(MoRe, self).__init__(train, query, gallery, **kwargs)

    def process_list(self, list_path, relabel=False):
        # e.g. pair04/camB/camB_id_00002_num_001.png -> pair=4, cam=B, pid=2
        pattern = re.compile(r'pair(\d+)/cam([AB])/cam[AB]_id_(\d+)_num_\d+')
        with open(list_path) as f:
            rel_paths = [line.strip() for line in f if line.strip()]

        items = []
        for rel_path in rel_paths:
            pair, cam, pid = pattern.search(rel_path).groups()
            camid = (int(pair) - 1) * 2 + (0 if cam == 'A' else 1)
            assert 0 <= camid < self._distractor_camid
            items.append((osp.join(self.data_dir, rel_path), int(pid), camid))

        if relabel:
            pid_container = sorted(set(pid for _, pid, _ in items))
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            items = [
                (img_path, pid2label[pid], camid)
                for img_path, pid, camid in items
            ]

        return items

    def process_distractors(self, dir_path):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*', '*.png')))
        return [
            (img_path, -1, self._distractor_camid) for img_path in img_paths
        ]
