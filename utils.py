from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode

import cv2
import matplotlib.pyplot as plt 
import os

# Function to configure training settings
def get_train_cfg(config_file_path, checkpoint_path, train_dataset_name, test_dataset_name, num_classes, device, output_dir, eval_period=500):
    """
    Generates a training configuration based on input parameters.

    Parameters:
        config_file_path (str): Path to the configuration file (.yaml) from model_zoo.
        checkpoint_path (str): Path to the model weights (.pth) or configuration file (.yaml).
        train_dataset_name (str): Name of the training dataset registered in Detectron2.
        test_dataset_name (str): Name of the test dataset registered in Detectron2.
        num_classes (int): Number of classes for the model.
        device (str): Device to use for training (e.g., "cuda" or "cpu").
        output_dir (str): Directory to save the model and logs.
        eval_period (int): Evaluation period during training. Default is 500.

    Returns:
        cfg: A Detectron2 configuration object.
    """
    cfg = get_cfg()

    # Load the base configuration file from model_zoo
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))

    # Determine weights based on checkpoint_path extension
    if checkpoint_path is not None:
        file_extension = os.path.splitext(checkpoint_path)[1]
        if file_extension == ".pth":
            cfg.MODEL.WEIGHTS = checkpoint_path  # Use the provided .pth file as weights
        elif file_extension == ".yaml":
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_path)  # Fetch weights from model_zoo
        else:
            raise ValueError("Unsupported file type for checkpoint_path. Use either .pth or .yaml.")
    else:
        cfg.MODEL.WEIGHTS = ""  # Start training from scratch

    # Set dataset names
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    # Set training parameters
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000  # Maximum iterations
    cfg.SOLVER.STEPS = []
    cfg.TEST.EVAL_PERIOD = eval_period  # Validation test period

    # Model-specific settings
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir

    return cfg

# Function to visualize predictions on an image
def on_image(image_path, predictor):
    """
    Displays an image with predictions from a given predictor.

    Parameters:
        image_path (str): Path to the input image.
        predictor: A Detectron2 predictor object for making predictions.
    """
    # Read the input image
    im = cv2.imread(image_path)

    # Get metadata if available
    metadata = MetadataCatalog.get(predictor.cfg.DATASETS.TEST[0]) if predictor.cfg.DATASETS.TEST else {}

    # Get predictions from the predictor
    outputs = predictor(im)

    # Create a visualizer for the image with predictions
    v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Prepare the figure to display side by side
    plt.figure(figsize=(16, 10))

    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    # Display the image with predictions
    plt.subplot(1, 2, 2)
    plt.imshow(v.get_image())
    plt.title("Image with Predictions")
    plt.axis("off")

    # Show both images
    plt.show()
