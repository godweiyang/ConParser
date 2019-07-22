import time

import numpy as np
import torch

def torch_load(load_path):
    if torch.cuda.is_available():
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)


def split_batch(batch_data, subbatch_max_tokens=2000):
    lens = [len(data['w']) for data in batch_data]

    lens = np.asarray(lens, dtype=int)
    lens_argsort = np.argsort(lens).tolist()

    num_subbatches = 0
    subbatch_size = 1
    while lens_argsort:
        if (subbatch_size == len(lens_argsort)) or (
                subbatch_size * lens[lens_argsort[subbatch_size]] >
                subbatch_max_tokens):
            yield [sentences[i] for i in lens_argsort[:subbatch_size]
                   ], [golds[i] for i in lens_argsort[:subbatch_size]]
            lens_argsort = lens_argsort[subbatch_size:]
            num_subbatches += 1
            subbatch_size = 1
        else:
            subbatch_size += 1


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string
