import json
import matplotlib.pyplot as plt

# Path to the metrics.json file
metrics_path = "../output/M4/metrics.json"

# Initialize dictionaries to store unique metrics
training_data = {}
validation_data = {}

# Read the JSON file and extract relevant metrics
with open(metrics_path, "r") as f:
    for line in f:
        data = json.loads(line)
        iteration = data.get("iteration", None)
        if iteration is not None:
            # Store training metrics by iteration (overwrite duplicates)
            training_data[iteration] = {
                "total_loss": data.get("total_loss", None),
                "accuracy": data.get("mask_rcnn/accuracy", None),
            }
            # Store validation metrics only when available
            if "bbox/AP" in data:
                validation_data[iteration] = {
                    "AP": data["bbox/AP"],
                    "AP50": data["bbox/AP50"],
                    "AP75": data["bbox/AP75"],
                }

# Extract sorted training metrics
iterations = sorted(training_data.keys())
training_loss = [training_data[i]["total_loss"] for i in iterations]
training_accuracy = [training_data[i]["accuracy"] for i in iterations]

# Extract sorted validation metrics
validation_iterations = sorted(validation_data.keys())
validation_AP = [validation_data[i]["AP"] for i in validation_iterations]
validation_AP50 = [validation_data[i]["AP50"] for i in validation_iterations]
validation_AP75 = [validation_data[i]["AP75"] for i in validation_iterations]

# Get Final AP Evaluation
if validation_iterations:
    final_iteration = validation_iterations[-1]
    final_AP = validation_AP[-1]
    final_AP50 = validation_AP50[-1]
    final_AP75 = validation_AP75[-1]
    print(f"Final AP Evaluation at Iteration {final_iteration}:")
    print(f"  AP (IoU=0.50:0.95): {final_AP:.3f}")
    print(f"  AP (IoU=0.50): {final_AP50:.3f}")
    print(f"  AP (IoU=0.75): {final_AP75:.3f}")

# Plot Training Metrics (Loss and Accuracy) on the Same Plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, training_loss, label="Training Loss", color="blue")
plt.plot(iterations, training_accuracy, label="Training Accuracy", color="green")
plt.xlabel("Iterations")
plt.ylabel("Metrics")
plt.title("Training Loss and Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Validation Metrics (AP)
plt.figure(figsize=(10, 6))
plt.plot(validation_iterations, validation_AP, label="AP @ IoU=0.50:0.95", color="orange")
plt.plot(validation_iterations, validation_AP50, label="AP @ IoU=0.50", color="red")
plt.plot(validation_iterations, validation_AP75, label="AP @ IoU=0.75", color="purple")
plt.xlabel("Iterations")
plt.ylabel("Average Precision (AP)")
plt.title("Validation Average Precision (AP)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
