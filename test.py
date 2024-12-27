from detectron2.engine import DefaultPredictor

import os
import pickle

from utils import *

# cfg_save_path = "OD_cfg.pickle"
# cfg_save_path = "IS_cfg.pickle"
# cfg_save_path = "IS2_cfg.pickle" # The latest update model
cfg_save_path = "./cfg/M4.pickle" # The latest update model

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0002999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

predictor = DefaultPredictor(cfg)

# image_path = "test/244032003741-20180826144046525.jpg"
# image_path = "./dataset/wheat_head_dataset/test/GSGC_test4.jpg"
image_path = "./dataset/self_annotated_dataset/test/unannotated/hs8.jpg"
# image_path = "./dataset/self_annotated_dataset/test/IMG-20241021-WA0157_jpg.rf.4d0ed24366ca227ae18725eb3574aca8.jpg"
# image_path = "./dataset/self_annotated_dataset/test/244032003741-20180826144051935.jpg"
# image_path = "./dataset/self_annotated_dataset/val/IMG-20241021-WA0167_jpg.rf.4d642e303d232e158f72d1edde8eacf2.jpg"
# image_path = "test/t0.jpg"
# image_path = "test/hs5.jpg"
# image_path = "extract_ear_spectra_dataset/ref_uv12_rice_Img-d(s10,g50,99.96ms,350-1100)_20240509_115821.jpg"
on_image(image_path, predictor)