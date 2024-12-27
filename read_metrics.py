import json
import matplotlib.pyplot as plt

# Load the metrics from the JSON file
metrics_path = "../output/M4/metrics.json"  # Path to the metrics file
metrics = []
with open(metrics_path, "r") as f:
    for line in f:
        metrics.append(json.loads(line))

# Initialize data containers for metrics
iterations = []
train_loss = []
mask_rcnn_accuracy = []
bbox_iterations = []
bbox_ap = []
bbox_ap50 = []
bbox_ap75 = []
segm_iterations = []
segm_ap = []
segm_ap50 = []
segm_ap75 = []

# Extract metrics from the JSON entries
for entry in metrics:
    # Collect iteration count
    if "iteration" in entry:
        iterations.append(entry["iteration"])
    # Collect training loss
    if "total_loss" in entry:
        train_loss.append(entry["total_loss"])
    # Collect Mask RCNN accuracy
    if "mask_rcnn/accuracy" in entry:
        mask_rcnn_accuracy.append(entry["mask_rcnn/accuracy"])
    # Collect bounding box metrics
    if "bbox/AP" in entry:
        bbox_iterations.append(entry["iteration"])
        bbox_ap.append(entry["bbox/AP"])
        bbox_ap50.append(entry["bbox/AP50"])
        bbox_ap75.append(entry["bbox/AP75"])
    # Collect segmentation metrics
    if "segm/AP" in entry:
        segm_iterations.append(entry["iteration"])
        segm_ap.append(entry["segm/AP"])
        segm_ap50.append(entry["segm/AP50"])
        segm_ap75.append(entry["segm/AP75"])

# Plot training loss and Mask RCNN accuracy
plt.figure(figsize=(10, 6))
plt.plot(iterations[:len(train_loss)], train_loss, label="Total Loss")
plt.plot(iterations[:len(mask_rcnn_accuracy)], mask_rcnn_accuracy, label="Mask RCNN Accuracy")
plt.xlabel("Iterations")
plt.ylabel("Metrics")
plt.title("Training Loss and Mask Accuracy")
plt.legend()
plt.grid()
plt.show()

# Plot bounding box Average Precision (AP) metrics
plt.figure(figsize=(10, 6))
plt.plot(bbox_iterations, bbox_ap, label="BBox AP")
plt.plot(bbox_iterations, bbox_ap50, label="BBox AP50")
plt.plot(bbox_iterations, bbox_ap75, label="BBox AP75")
plt.xlabel("Iterations")
plt.ylabel("Bounding Box AP Metrics")
plt.title("Bounding Box AP Metrics")
plt.ylim(0, 100)  # Set y-axis range from 0 to 100
plt.legend()
plt.grid()
plt.show()

# Plot segmentation Average Precision (AP) metrics
plt.figure(figsize=(10, 6))
plt.plot(segm_iterations, segm_ap, label="Segm AP")
plt.plot(segm_iterations, segm_ap50, label="Segm AP50")
plt.plot(segm_iterations, segm_ap75, label="Segm AP75")
plt.xlabel("Iterations")
plt.ylabel("Segmentation AP Metrics")
plt.title("Segmentation AP Metrics")
plt.ylim(0, 100)  # Set y-axis range from 0 to 100
plt.legend()
plt.grid()
plt.show()
