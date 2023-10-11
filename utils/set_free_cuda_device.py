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
        least_busy_utilization = 100  # Initialize with a high value

        for info in gpu_info:
            index, utilization, total_memory, used_memory, temperature, name = info
            utilization = float(utilization)
            used_memory = int(used_memory)

            # Check if the GPU is running a process
            if utilization == 0 and used_memory < 10:
                least_busy_gpu_index = index
                break

            # Check if the GPU has lower utilization and lower memory usage
            if utilization < least_busy_utilization:
                least_busy_gpu_index = index
                least_busy_utilization = utilization

        os.environ["CUDA_VISIBLE_DEVICES"] = least_busy_gpu_index
        print(least_busy_gpu_index)
        return least_busy_gpu_index
    except Exception as e:
        print(f"Error: {e}")
        return None
