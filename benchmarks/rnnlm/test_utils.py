import tensorflow as tf


def get_config():
    config = tf.compat.v1.ConfigProto()

    config.log_device_placement = False
    config.allow_soft_placement = True

    config.gpu_options.allow_growth = True

    config.intra_op_parallelism_threads = 0
    config.inter_op_parallelism_threads = 56
    return config


def device():
    return "/device:GPU:0" if tf.test.is_gpu_available(
        cuda_only=True) else "/device:CPU:0"


def force_gpu_sync():
    tf.constant(1).gpu().cpu()
