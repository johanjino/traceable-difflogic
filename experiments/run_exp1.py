import main
import subprocess
import os
import shutil
import math

if __name__ == "__main__":
    ''' from ConvDLG paper: 
    A good rule of thumb during scaling is that the optimal temperature is 
    proportional to the square-root of the number of output gates
    k_out | tau
    8000  | 10  => linear scaler: k = 10 / sqrt(8000/10) = 0.3536
    4800  | 7.7 => 0.3536 * sqrt(4800/10)
    2300  | 5.4
    750   | 3
    '''
    LINEAR_TAUS = True
    K_NEURONS = [2000,4000,6000,8000,10000,12000]
    GROUP_SUM_TAUS = [0.3536 * math.sqrt(k / 10) for k in K_NEURONS]
    
    for i, NUM_NEURONS in enumerate(K_NEURONS):
        # !!! quick fix !!!
        if NUM_NEURONS < 12000:
            continue
        
        # Experiment parameters
        BATCH_SIZE = 100
        TAU = 10 if not LINEAR_TAUS else GROUP_SUM_TAUS[i]
        DATASET = "mnist20x20"
        NUM_ITERATIONS = 100000
        EVAL_FREQ = 5000
        NUM_LAYERS = 6
        EXPERIMENT_ID = 520000+i  # Ensure this is in the expected range
        
        # Define expected filenames based on your naming convention
        model_file = f"{DATASET}_k{NUM_NEURONS}_l{NUM_LAYERS}"
        model_file += f"_t{TAU}" if LINEAR_TAUS else ""
        
        # Define results directory
        RESULTS_DIR = "./results"
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        
        print("Starting Experiment 1:")
        print(f"Dataset: {DATASET}")
        print(f"Batch Size: {BATCH_SIZE}, Tau: {TAU}")
        print(f"Iterations: {NUM_ITERATIONS}, Eval Frequency: {EVAL_FREQ}")
        print(f"Neurons: {NUM_NEURONS}, Layers: {NUM_LAYERS}")
        
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
        log_file_path = os.path.join(RESULTS_DIR, f"{model_file}.log")
        with open(log_file_path, "w") as logfile:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # Read and log the output line-by-line.
            for line in process.stdout:
                print(line, end="")  # Also print to terminal
                logfile.write(line)
            process.wait()