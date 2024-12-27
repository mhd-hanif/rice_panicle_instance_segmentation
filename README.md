# Rice Panicle Instance Segmentation

## Project Introduction
This project focuses on **rice panicle instance segmentation**, aiming to detect and segment the boundaries of rice panicles from RGB image inputs. Using the **Detectron2** library with the Mask R-CNN architecture, this repository provides tools for training and inference to achieve precise segmentation.

For a detailed explanation of the methods and implementation, please refer to this [link](https://github.com/mhd-hanif/rice_panicle_instance_segmentation).

---

## How to Install & Setup

### Prerequisites
This tutorial is beginner-friendly and offers useful resources for setting up the Detectron2 library:

- [Detectron2 Setup Guide - Tutorial 1](https://www.youtube.com/watch?v=Pb3opEFP94U)
- [Detectron2 Basics - Tutorial 2](https://www.youtube.com/watch?v=ffTURA0JM1Q)
- [Detectron2 Training Workflow - Tutorial 3](https://www.youtube.com/watch?v=GoItxr16ae8)

### Steps to Install and Setup

1. **Install Detectron2**
   - Follow the instructions from the official Detectron2 GitHub repository: [Detectron2 GitHub](https://github.com/facebookresearch/detectron2).

2. **Clone This Repository**
   ```bash
   git clone https://github.com/mhd-hanif/rice_panicle_instance_segmentation.git
   cd rice_panicle_instance_segmentation
   ```

3. **Download the Dataset**
   - Download the datasets from [this link](https://drive.google.com/drive/folders/1YdZE4nLoY9pw7kZU8QaJJ4EPuR8n_HMo?usp=sharing).
   - Place the `wheat_head_dataset` and `rice_panicle_dataset` folders inside the `dataset` folder in this repository.

4. **Download Trained Model Weights**
   - Download the pretrained models from [this link](https://drive.google.com/drive/folders/1YdZE4nLoY9pw7kZU8QaJJ4EPuR8n_HMo?usp=sharing).
   - Place the `M1`, `M3`, and `M4` model files into the `output` folder in this repository.

   #### Model Descriptions:
   - **M1 Model**: Trained exclusively on the `wheat_head_dataset`.
   - **M3 Model**: Trained exclusively on the `rice_panicle_dataset`.
   - **M4 Model**: First trained on the `wheat_head_dataset` and then fine-tuned on the `rice_panicle_dataset`.

### Directory Structure Example
After setting up the datasets and downloading the trained models, your directory should look like this:
```
rice_panicle_instance_segmentation/
│
├── dataset/
│   ├── wheat_head_dataset/
│   ├── rice_panicle_dataset/
│
├── output/
│   ├── M1.pth
│   ├── M3.pth
│   ├── M4.pth
│
├── configs/
├── scripts/
├── README.md
└── ...
```
Your setup is now complete. Please refer to the next section to learn how to use the repository.

---

## How to Use

### 1. Training the Model
To train the model from scratch, run the following command:
```bash
python train.py --config-file configs/rice_panicle_train_config.yaml
```

### 2. Running Inference
To perform inference using a trained model, use the following command:
```bash
python inference.py --input input_image.jpg --output output_directory --model models/M3.pth
```

### Pretrained Models
Three pretrained models are provided in the `output` folder:
- **M1 Model**: Trained exclusively on the `wheat_head_dataset`.
- **M3 Model**: Trained exclusively on the `rice_panicle_dataset`.
- **M4 Model**: First trained on the `wheat_head_dataset` and then fine-tuned on the `rice_panicle_dataset`.

---

## Model Evaluation
The model evaluation includes training and validation accuracy metrics:

| Model | Dataset             | Training Accuracy | Validation Accuracy |
|-------|---------------------|-------------------|---------------------|
| M1    | Wheat Head         | 90.1%             | 87.5%               |
| M3    | Rice Panicle        | 92.3%             | 89.7%               |
| M4    | Wheat + Rice Panicle| 94.5%             | 91.2%               |

---

For additional details or to contribute to the project, feel free to explore or raise issues in this repository.
