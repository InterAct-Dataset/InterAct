import numpy as np
import torch
from os.path import join as pjoin


from tma.data.humanml.scripts.motion_process import (
    process_file,
    recover_from_ric,
    recover_from_root_rot6d,
)

from .base import BASEDataModule
from .humanml.data.dataset import Text2MotionDatasetV3, TextOnlyDataset
from .humanml.common.skeleton import Skeleton



class InterActDataModule(BASEDataModule):

    def __init__(
        self, cfg, batch_size, num_workers, collate_fn=None, phase="train", **kwargs
    ):
        super().__init__(
            batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn
        )
        self.save_hyperparameters(logger=False)
        self.name = "interact"
        self.njoints = 22
        self.hparams["njoints"] = 22
        if phase == "text_only":
            self.Dataset = TextOnlyDataset
        else:
            self.Dataset = Text2MotionDatasetV3
        self.cfg = cfg
        sample_overrides = {"split": "val", "tiny": True, "progress_bar": False}

        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        # Get additional info of the dataset
        # import pdb; pdb.set_trace()
        self.nfeats = self._sample_set.nfeats
        # self.transforms = self._sample_set.transforms

    def feats2joints(self, features, skel=None, motion_type="vector_263"):
        # mean = torch.tensor(self.hparams.mean).to(features)
        # std = torch.tensor(self.hparams.std).to(features)
        # features = features * std + mean
        # return recover_from_ric(features, self.njoints)
        if motion_type in [
            "vector_263",
            "root_position",
            "root_position_vel",
            "root_position_rot6d",
        ]:
            mean = torch.tensor(self.hparams.mean).to(features)
            std = torch.tensor(self.hparams.std).to(features)
            features = features * std + mean
            return recover_from_ric(
                features, self.njoints
            )  # torch.Size([32, 92, 22, 3])
        elif motion_type in ["root_rot6d"]:
            mean = torch.tensor(self.hparams.mean).to(features)
            std = torch.tensor(self.hparams.std).to(features)
            features = features * std + mean
            
            # skeleton = Skeleton(n_raw_offsets, kinematic_chain, )
            return recover_from_root_rot6d(features, self.njoints, skel)
        elif motion_type == "smplx_212":
            assert smplx_model is not None
            mean = torch.tensor(self.hparams.mean).to(features)
            std = torch.tensor(self.hparams.std).to(features)
            features = features * (std + 1e-7) + mean
            bs = features.shape[0]
            features = features.reshape(-1, 212)
            output = smplx_model.smplx_model(
                pose_body=features[:, 3:66],
                pose_hand=features[:, 66:156],
                root_orient=features[:, :3],
            ).Jtr
            return output.reshape(bs, -1, 55, 3)  # torch.Size([32, 96, 55, 3])
        else:
            raise NotImplementedError

    def joints2feats(self, features):
        features = process_file(features, self.njoints)[0]
        # mean = torch.tensor(self.hparams.mean).to(features)
        # std = torch.tensor(self.hparams.std).to(features)
        # features = (features - mean) / std
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        # random select samples for mm
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(
                self.name_list, self.cfg.TEST.MM_NUM_SAMPLES, replace=False
            )
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
    def get_sample_set(self, overrides={}):
        sample_params = self.hparams.copy()
        sample_params.update(overrides)
        # import pdb; pdb.set_trace()

        split=self.cfg.EVAL.SPLIT
        # split_file = pjoin(
        #     eval(f"self.cfg.DATASET.{self.name.upper()}.SPLIT_ROOT"),
        #     self.cfg.DATASET.VERSION,
        #     self.cfg.EVAL.SPLIT + ".txt",
        # )
        # if 'split' not in sample_params.keys():
        # sample_params['split']=split
        # else:
        #     print('SAMPLE_SPLIT',sample_params['split'])

        # import pdb; pdb.set_trace()
        return self.Dataset( split_datapart=split,**sample_params)

    def __getattr__(self, item):
        # train_dataset/val_dataset etc cached like properties
        print(item,'ITEMITEMITEM')
        if item.endswith("_dataset") and not item.startswith("_"):
            subset = item[: -len("_dataset")]
            item_c = "_" + item
            if item_c not in self.__dict__:
                # todo: config name not consistent
                subset = subset.upper() if subset != "val" else "EVAL"
                split = eval(f"self.cfg.{subset}.SPLIT")
                split_file = pjoin(
                    eval(f"self.cfg.DATASET.{self.name.upper()}.SPLIT_ROOT"),
                    self.cfg.DATASET.VERSION,
                    eval(f"self.cfg.{subset}.SPLIT") + ".txt",
                )
                self.__dict__[item_c] = self.Dataset(
                    split_datapart=split,split=split,**self.hparams
                )
            return getattr(self, item_c)
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")
