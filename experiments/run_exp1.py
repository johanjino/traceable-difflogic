import main
import subprocess
import os
import shutil

if __name__ == "__main__":
    for i, NUM_NEURONS in enumerate([2000,4000,6000,8000,10000,12000]):
        # Experiment parameters
        BATCH_SIZE = 100
        TAU = 10
        DATASET = "mnist20x20"
        NUM_ITERATIONS = 200000
        EVAL_FREQ = 5000
        NUM_LAYERS = 6
        EXPERIMENT_ID = 520000+i  # Ensure this is in the expected range
        
        # Define results directory
        RESULTS_DIR = "./results/1"
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        
        print("Starting Experiment 1:")
        print(f"Dataset: {DATASET}")
        print(f"Batch Size: {BATCH_SIZE}, Tau: {TAU}")
        print(f"Iterations: {NUM_ITERATIONS}, Eval Frequency: {EVAL_FREQ}")
        print(f"Neurons: {NUM_NEURONS}, Layers: {NUM_LAYERS}")
        print(f"Experiment ID: {EXPERIMENT_ID}")
        
        # Build the command as a list of arguments
        cmd = [
            "python", "main.py",
            "-bs", str(BATCH_SIZE),
            "-t", str(TAU),
            "--dataset", DATASET,
            "-ni", str(NUM_ITERATIONS),
            "-ef", str(EVAL_FREQ),
            "-k", str(NUM_NEURONS),
            "-l", str(NUM_LAYERS),
            "-vss", str(0.2),
            "--experiment_id", str(EXPERIMENT_ID)
        ]
        
        # Run the experiment and log the output to a file
        log_file_path = os.path.join(RESULTS_DIR, "experiment1.log")
        with open(log_file_path, "w") as logfile:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # Read and log the output line-by-line.
            for line in process.stdout:
                print(line, end="")  # Also print to terminal
                logfile.write(line)
            process.wait()