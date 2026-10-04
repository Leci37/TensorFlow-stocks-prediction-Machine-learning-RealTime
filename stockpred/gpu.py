import tensorflow as tf
from stockpred import config

def compute_resource():
    if config.USE_GPU.lower() == 'yes' or config.USE_GPU.lower() == 'y':
        # Allow memory growth for the GPU
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=config.PER_PROCESS_GPU_MEMORY_FRACTION)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    else:
        print("Going to use CPU ")
        physical_devices = tf.config.experimental.list_physical_devices('CPU')
