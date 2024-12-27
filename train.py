from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import EvalHook
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
import os
import pickle
import torch
from utils import get_train_cfg

# Paths to configuration, checkpoint, and datasets

# Option 1: Start from the beginning using COCO pre-trained model (.yaml)
# Uncomment the following lines if you want to start training with COCO pre-trained weights:
# config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# checkpoint_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"  # Path to COCO pre-trained weights

# Option 2: Continue from custom-trained weights (.pth)
# Use this section if you want to continue training from a custom-trained model
config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
checkpoint_path = "./output/M2/model_final.pth"  # Path to the model's checkpoint

# Output directory and training parameters
output_dir = "./output/M1_E"
num_classes = 1
device = "cuda"

# Dataset paths

# Training dataset
train_dataset_name = "SAD_train"
train_images_path = "./dataset/self_annotated_dataset/train"
train_json_annot_path = "./dataset/self_annotated_dataset/annotations/train.json"

# Validation dataset
val_dataset_name = "SAD_val"
val_images_path = "./dataset/self_annotated_dataset/val"
val_json_annot_path = "./dataset/self_annotated_dataset/annotations/val.json"

# Test dataset
test_dataset_name = "SAD_val"
test_images_path = "./dataset/self_annotated_dataset/val"
test_json_annot_path = "./dataset/self_annotated_dataset/annotations/val.json"

# Path to save the configuration
cfg_save_path = "./cfg/M1_E.pickle"

# Register COCO-format datasets
register_coco_instances(name=train_dataset_name, metadata={}, 
                        json_file=train_json_annot_path, image_root=train_images_path)

register_coco_instances(name=test_dataset_name, metadata={}, 
                        json_file=test_json_annot_path, image_root=test_images_path)

register_coco_instances(name=val_dataset_name, metadata={}, 
                        json_file=val_json_annot_path, image_root=val_images_path)

# Define augmentation pipeline using Detectron2
use_augmentation = False  # Set to True to include augmentation

def custom_mapper(dataset_dict):
    """
    Custom mapper for data augmentation and preprocessing.
    """
    dataset_dict = dataset_dict.copy()  # Avoid modifying the original dataset_dict

    # Load and augment the image
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    if use_augmentation:
        augmentations = [
            # T.ResizeShortestEdge(short_edge_length=(800, 800), max_size=1333),  # Resize
            T.RandomFlip(horizontal=True, vertical=False),  # Horizontal flip
            T.RandomBrightness(0.8, 1.2),  # Adjust brightness
            T.RandomContrast(0.8, 1.2),  # Adjust contrast
        ]
        image, transforms = T.apply_transform_gens(augmentations, image)

        # Transform annotations and create ground truth instances
        if "annotations" in dataset_dict:
            valid_annos = []
            for annotation in dataset_dict["annotations"]:
                if "bbox" in annotation and len(annotation["bbox"]) == 4:
                    transformed_anno = utils.transform_instance_annotations(
                        annotation, transforms, image.shape[:2]
                    )
                    valid_annos.append(transformed_anno)
            dataset_dict["annotations"] = valid_annos

    # Convert the image into a tensor
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    # Ensure annotations are properly converted to Instances for model training
    if "annotations" in dataset_dict:
        image_shape = image.shape[:2]  # Height, Width
        dataset_dict["instances"] = utils.annotations_to_instances(dataset_dict["annotations"], image_shape)

    return dataset_dict

# Custom Trainer Class with Optional Augmentation and Validation
class TrainerWithAugmentation(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        if use_augmentation:
            return build_detection_train_loader(cfg, mapper=custom_mapper)
        return build_detection_train_loader(cfg)  # Default loader without augmentation

    def build_hooks(self):
        default_hooks = super().build_hooks()
        eval_hook = EvalHook(0, lambda: self.test(self.cfg, self.model))
        default_hooks.insert(-1, eval_hook)
        return default_hooks

# Main Training Function
def main(resume=False):
    """
    Main function to train the Detectron2 model.

    Parameters:
        resume (bool): If True, resumes training from the last checkpoint if available.
    """
    # Load training configuration
    cfg = get_train_cfg(config_file_path, checkpoint_path, train_dataset_name, 
                        test_dataset_name, num_classes, device, output_dir, eval_period=500)

    # Ensure checkpoints are saved periodically
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Save checkpoint every 1000 iterations

    # Save the configuration to a file for reference
    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Create the output directory if it doesn't exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Initialize and train the model
    trainer = TrainerWithAugmentation(cfg)
    trainer.resume_or_load(resume=resume)  # Resume training if a checkpoint exists
    trainer.train()

if __name__ == '__main__':
    import argparse

    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Train a Detectron2 model with optional resume.")
    parser.add_argument('--resume', action='store_true', help="Resume training from last checkpoint if available")
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(resume=args.resume)
