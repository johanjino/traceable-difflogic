import os
import subprocess
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Experiment settings -----
# List of output neuron counts to try (for example, for MNIST num_classes = 10)
K_NEURONS = [2000, 4000, 6000, 8000, 10000, 12000]
TAU_VALUES = [4, 6, 8, 10, 12, 14]
NUM_CLASSES = 10

# Fixed network parameters
NUM_LAYERS = 6
DATASET = "mnist20x20"
BATCH_SIZE = 100
NUM_ITERATIONS = 50000
EVAL_FREQ = 5000

# Base experiment ID (for uniqueness)
base_experiment_id = 525000

# Directory for logs and results
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Prepare a results matrix:
# rows: tau values, columns: output_neurons/num_classes (ratio)
results = np.zeros((len(TAU_VALUES), len(K_NEURONS)))

# Loop over each output neuron configuration
for col, num_neurons in enumerate(K_NEURONS):
    # Compute the output ratio for plotting: num_neurons / NUM_CLASSES
    output_ratio = num_neurons / NUM_CLASSES

    # Loop over each tau value in our fixed list
    for row, tau in enumerate(TAU_VALUES):
        # Create a unique experiment ID for this configuration
        experiment_id = base_experiment_id + col * len(TAU_VALUES) + row

        # Construct a model filename for logging (you can modify this if needed)
        model_filename = f"{DATASET}_k{num_neurons}_l{NUM_LAYERS}_tau{tau:.2f}"

        # Build the command to run the experiment:
        cmd = [
            "python", "main.py",
            "-bs", str(BATCH_SIZE),
            "-t", str(tau),
            "--dataset", DATASET,
            "-ni", str(NUM_ITERATIONS),
            "-ef", str(EVAL_FREQ),
            "-k", str(num_neurons),
            "-l", str(NUM_LAYERS),
            "--experiment_id", str(experiment_id)
        ]

        print(f"\nRunning experiment: Output neurons = {num_neurons} (ratio={output_ratio:.1f}), tau = {tau}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="")  # also print to terminal
        process.wait()

        # After the experiment, load the JSON log file.
        # We assume that main.py writes a JSON log file named "{experiment_id}.json" in RESULTS_DIR.
        json_file_path = os.path.join(RESULTS_DIR, f"00{experiment_id}.json")
        try:
            with open(json_file_path, "r") as jf:
                log_data = json.load(jf)
            final_accuracy = float(log_data["test_acc_eval_mode_"])
        except Exception as e:
            print(f"Error loading or parsing {json_file_path}: {e}")
            final_accuracy = 0.0

        results[row, col] = final_accuracy

# After all experiments complete, plot a heatmap.
# x-axis: (output_neurons / num_classes) and y-axis: tau values.
x_labels = [f"{k / NUM_CLASSES:.1f}" for k in K_NEURONS]
y_labels = [str(tau) for tau in TAU_VALUES]

plt.figure(figsize=(10, 6))
ax = sns.heatmap(results, annot=True, fmt=".1f", xticklabels=x_labels, yticklabels=y_labels, cmap="viridis")
ax.set_xlabel("Output Neurons / Num Classes")
ax.set_ylabel("Tau")
ax.set_title("Grid Search: Accuracy vs Output Ratio and Tau")
heatmap_path = os.path.join(RESULTS_DIR, "grid_search_heatmap.png")
plt.savefig(heatmap_path)
plt.show()

print(f"Grid search complete. Heatmap saved to: {heatmap_path}")