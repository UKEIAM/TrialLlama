import os
import subprocess

""""Since with PEFT methods only one GPU is usable, we simply select the one currently least used."""


def get_free_cuda_device():
    try:
        # Run nvidia-smi to get GPU information
        nvidia_smi_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.total,memory.used,temperature.gpu,name",
                "--format=csv,noheader,nounits",
            ]
        )

        # Split the output into lines
        lines = nvidia_smi_output.decode("utf-8").strip().split("\n")
        gpu_info = [line.strip().split(", ") for line in lines]

        # Initialize variables to keep track of the least busy GPU
        least_busy_gpu_index = None
        least_busy_utilization = 50  # Initialize with a high value
        rel_used_memory = []
        for info in gpu_info:
            index, utilization, total_memory, used_memory, temperature, name = info
            utilization = float(utilization)
            used_memory = int(used_memory)

            rel_used_memory.append(used_memory / int(total_memory))

        least_busy_gpu_index = rel_used_memory.index(min(rel_used_memory))
        least_busy_memory = rel_used_memory[least_busy_gpu_index]
        if least_busy_memory > 0.05:
            # If the least busy GPU is more than 5% busy, return None
            return False

        os.environ["CUDA_VISIBLE_DEVICES"] = str(least_busy_gpu_index)
        print(least_busy_gpu_index)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return None
