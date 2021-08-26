import tensorflow as tf
import numpy as np

def gpu_setting(memory):
    ################### Limit GPU Memory ###################
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("########################################")
    print('{} GPU(s) is(are) available'.format(len(gpus)))
    print("########################################")

    # set the only one GPU and memort limit
    memory_limit = memory * 1024

    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            print("Use only one GPU{} limited {}MB memory".format(gpus[0], memory_limit))
        except RuntimeError as e:
            print(e)

    else:
        print('GPU is not available')
    ##########################################################


def slice_data(dataframe, window):
    for column in dataframe.columns:
        if "strkdpthpcval" in column.lower():

            brake_sig = column
            dataframe.drop([brake_sig], axis=1, inplace=True)

    features = len(dataframe.columns)
    sliced_size = len(dataframe) - window + 1

    result = np.zeros(shape=(sliced_size, window, features))

    for i in range(sliced_size):
        result[i] = dataframe.values[i:i + window]

    return result

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)