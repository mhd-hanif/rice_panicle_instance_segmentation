from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import EvalHook
from detectron2.evaluation import COCOEvaluator
import os
import pickle
from utils import get_train_cfg

# Paths to config, checkpoint, and datasets
config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

##################M1#######################################################################
# output_dir = "./output/M1"
# num_classes = 1
# device = "cuda"

# # Dataset paths
# train_dataset_name = "WHD_train"
# train_images_path = "./dataset/wheat_head_dataset/train"
# train_json_annot_path = "./dataset/wheat_head_dataset/annotations/train_coco.json"

# test_dataset_name = "WHD_test"
# test_images_path = "./dataset/wheat_head_dataset/test"
# test_json_annot_path = "./dataset/wheat_head_dataset/annotations/test_coco.json"

# val_dataset_name = "WHD_val"
# val_images_path = "./dataset/wheat_head_dataset/val"
# val_json_annot_path = "./dataset/wheat_head_dataset/annotations/val_coco.json"

# cfg_save_path = "./cfg/M1.pickle"


##################M2#######################################################################
# output_dir = "./output/M2"
# num_classes = 1
# device = "cuda"

# # Dataset paths
# train_dataset_name = "WHD_train"
# train_images_path = "./dataset/wheat_head_dataset/train"
# train_json_annot_path = "./dataset/wheat_head_dataset/annotations/train_coco_panicle.json"

# test_dataset_name = "WHD_test"
# test_images_path = "./dataset/wheat_head_dataset/test"
# test_json_annot_path = "./dataset/wheat_head_dataset/annotations/test_coco_panicle.json"

# val_dataset_name = "WHD_val"
# val_images_path = "./dataset/wheat_head_dataset/val"
# val_json_annot_path = "./dataset/wheat_head_dataset/annotations/val_coco_panicle.json"

# cfg_save_path = "./cfg/M2.pickle"



##################M3#######################################################################
output_dir = "./output/M3"
num_classes = 1
device = "cuda"

# Dataset paths
train_dataset_name = "SAD_train"
train_images_path = "./dataset/self_annotated_dataset/train"
train_json_annot_path = "./dataset/self_annotated_dataset/annotations/train.json"

# test_dataset_name = "SAD_test"
# test_images_path = "./dataset/self_annotated_dataset/test"
# test_json_annot_path = "./dataset/self_annotated_dataset/annotations/test.json"

test_dataset_name = "SAD_val"
test_images_path = "./dataset/self_annotated_dataset/val"
test_json_annot_path = "./dataset/self_annotated_dataset/annotations/val.json"

val_dataset_name = "SAD_val"
val_images_path = "./dataset/self_annotated_dataset/val"
val_json_annot_path = "./dataset/self_annotated_dataset/annotations/val.json"

cfg_save_path = "./cfg/M3.pickle"

# Register datasets
register_coco_instances(name=train_dataset_name, metadata={}, 
                        json_file=train_json_annot_path, image_root=train_images_path)

register_coco_instances(name=test_dataset_name, metadata={}, 
                        json_file=test_json_annot_path, image_root=test_images_path)

register_coco_instances(name=val_dataset_name, metadata={}, 
                        json_file=val_json_annot_path, image_root=val_images_path)

# Custom Trainer Class to Include Validation
class TrainerWithValidation(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        # Call the parent class's build_hooks method
        default_hooks = super().build_hooks()
        # Add validation hook
        eval_hook = EvalHook(0, lambda: self.test(self.cfg, self.model))
        default_hooks.insert(-1, eval_hook)
        return default_hooks

# Main Training Function
def main(resume=False):
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, 
                        test_dataset_name, num_classes, device, output_dir, eval_period=500)

    # Ensure checkpoints are saved periodically
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Save checkpoint every 1000 iterations

    # Save the config to a file
    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = TrainerWithValidation(cfg)
    trainer.resume_or_load(resume=resume)  # Resume training if a checkpoint exists

    trainer.train()


if __name__ == '__main__':
    import argparse

    # Argument parser for resume option
    parser = argparse.ArgumentParser(description="Train a Detectron2 model with optional resume.")
    parser.add_argument('--resume', action='store_true', help="Resume training from last checkpoint if available")
    args = parser.parse_args()

    main(resume=args.resume)
