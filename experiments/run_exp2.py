import main
import subprocess
import os
import shutil

if __name__ == "__main__":
    NUM_NEURONS_LISTS = [
        # standard sparse architecture (baseline)
        # [8000, 8000, 8000, 8000, 8000, 8000], # 48k

        # very sparse architecture (expand-contract)
        [24000, 12000, 6000, 3000, 1500, 750], # 47250, ratio=2
        [17466, 11644, 7763, 5175, 3450, 2300], # 47796, ratio=1.5
        [11944, 9953, 8294, 6912, 5760, 4800], # 47663, ratio=1.2
        
        # # very sparse architecture (expand-contract)
        # [24000, 12000, 6000, 3000, 1500], # 46500, ratio=2
        # [11719, 11813, 7875, 5250, 3500], # 46156, ratio=1.5
        # [12442, 10368, 8640, 7200, 6000], # 44649, ratio=1.2
    ]
    
    for i, num_neorons_list in enumerate(NUM_NEURONS_LISTS):
        # Experiment parameters
        BATCH_SIZE = 100
        TAU = 10
        DATASET = "mnist20x20"
        NUM_ITERATIONS = 200000
        EVAL_FREQ = 5000
        NUM_LAYERS = 6
        ARCHITECTURE = "randomly_connected_list"
        EXPERIMENT_ID = 521000+i  # Ensure this is in the expected range

        # Define expected filenames based on your naming convention
        model_file = f"{DATASET}_arch{i}_l{NUM_LAYERS}"
        
        # Define results directory
        RESULTS_DIR = "./results/2"
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        
        print("Starting Experiment 2:")
        print(f"Dataset: {DATASET}")
        print(f"Batch Size: {BATCH_SIZE}, Tau: {TAU}")
        print(f"Iterations: {NUM_ITERATIONS}, Eval Frequency: {EVAL_FREQ}")
        print(f"Neurons: {num_neorons_list}, Layers: {NUM_LAYERS}")
        print(f"Experiment ID: {EXPERIMENT_ID}")
        
        # Build the command as a list of arguments
        cmd = [
            "python", "main.py",
            "-bs", str(BATCH_SIZE),
            "-t", str(TAU),
            "--dataset", DATASET,
            "-ni", str(NUM_ITERATIONS),
            "-ef", str(EVAL_FREQ),
            "-a", str(ARCHITECTURE),
            "-K", *map(lambda x:str(x), num_neorons_list),
            "-l", str(NUM_LAYERS),
            "--experiment_id", str(EXPERIMENT_ID),
            "--name", model_file,
        ]
        # print(" ".join(map(lambda x:str(x), num_neorons_list)))
        # exit()
        
        # Run the experiment and log the output to a file
        log_file_path = os.path.join(RESULTS_DIR, "experiment2.log")
        with open(log_file_path, "w") as logfile:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # Read and log the output line-by-line.
            for line in process.stdout:
                print(line, end="")  # Also print to terminal
                logfile.write(line)
            process.wait()