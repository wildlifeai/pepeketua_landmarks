from io import BytesIO
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


class ServerImageDataGenerator:
    """Replaces ImageDataGenerator with one that loads images from a server"""

    def __init__(self, rescale: float = 1 / 255):
        self.rescale = rescale

    def flow_from_dataframe(
        self,
        dataframe: pd.DataFrame,
        x_col: str,
        y_col: List[str],
        target_size: Optional[Tuple[int]] = None,
        batch_size: int = 3,
    ):
        return ServerFlowIterator(
            dataframe,
            x_col,
            y_col,
            target_size,
            batch_size,
            self.rescale,
        )


class ServerFlowIterator:
    def __init__(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: Sequence[str],
        target_size: Tuple[int],
        batch_size: int,
        rescale: float,
    ):
        self.df = df
        self.n = len(df)
        self.x_col = x_col
        self.labels = df[y_col].values
        self.target_size = target_size
        self.batch_size = batch_size
        self.rescale = rescale
        self.batch_index = 0
        self.total_batches_seen = 0
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)

    def __getitem__(self, idx: int) -> np.array:
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[
            self.batch_size * idx : self.batch_size * (idx + 1)
        ]
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array: np.array):
        # generate batch array including color depth channel
        batch_x = np.zeros((len(index_array),) + self.target_size + (3,), dtype=None)
        for i, j in enumerate(index_array):
            # load, resize and rescale image to target size and scale
            with Image.open(BytesIO(self.df.iloc[j][self.x_col])) as image:
                image = image.resize(self.target_size, Image.Resampling.NEAREST)
                image_arr = tf.keras.utils.img_to_array(image)
                image_arr = np.array([image_arr])  # Convert single image to a batch.
                image_arr *= self.rescale  # rescale image by factor

            batch_x[i] = image_arr

        batch_y = self.labels[index_array]

        return batch_x, batch_y

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        self.reset()
        while 1:
            if self.batch_index == 0:
                self._set_index_array()

            if self.n == 0:
                current_index = 0
            else:
                current_index = (self.batch_index * self.batch_size) % self.n

            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0

            yield self.index_array[current_index : current_index + self.batch_size]

    def on_epoch_end(self):
        self._set_index_array()

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        return self

    def next(self):
        index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

    def __next__(self):
        return self.next()
