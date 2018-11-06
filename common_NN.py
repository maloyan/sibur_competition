import pandas as pd
import numpy as np


def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


lookback = 10
step = 1
delay = 5
batch_size = 5

a = np.arange(50)
b = np.arange(100, 150)
data = pd.DataFrame({'a':a, 'b':b})

train_gen = generator(data.values, lookback=lookback, delay=delay, min_index=0, max_index= len(data)-1,
                      shuffle=True, step=step, batch_size=batch_size)

for i in train_gen:
    print(i)
