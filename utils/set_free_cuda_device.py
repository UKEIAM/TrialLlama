import os
import subprocess

""""Since with PEFT methods only one GPU is usable, we simply select the one currently least used."""


def get_free_cuda_device():
    try:
        # Run nvidia-smi to get GPU information
        nvidia_smi_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )

        # Split the output into lines
        lines = nvidia_smi_output.decode("utf-8").strip().split("\n")

        # Parse the GPU utilization data
        gpu_utilization = [line.strip().split(", ") for line in lines]

        # Find the first available GPU (lowest utilization)
        free_gpu = min(gpu_utilization, key=lambda x: float(x[1]))

        # Set an environment variable with the free GPU index
        os.environ["CUDA_VISIBLE_DEVICES"] = free_gpu[0]

        return int(free_gpu[0])
    except Exception as e:
        print(f"Error: {e}")
        return None
