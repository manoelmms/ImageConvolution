import numpy as np
import cv2
from kernel import Kernel2D
import multiprocessing as mp
from multiprocessing import shared_memory
import threading as th
from concurrent.futures import ThreadPoolExecutor
from numba import jit


class Image:

    def __init__(self, filepath: str):
        """
        Initialize the image
        :param filepath: The path to the image
        """
        self.data = cv2.imread(filepath)
        self.height, self.width, self.channels = self.data.shape
        self.filepath = filepath
        self.name = filepath.split('/')[-1].split('.')[0]
        self.pitch = self.width  # Can be used to optimize the padding

    @classmethod
    def from_data(cls, data: np.ndarray):
        """
        Initialize the image from data
        :param data: The image data
        :return: The image
        """
        image = cls.__new__(cls)
        image.data = data
        image.height, image.width, image.channels = data.shape
        image.filepath = None
        image.name = "Image"
        image.pitch = image.width
        return image

    def padding(self, padding_size_x: int, padding_size_y: int):
        """
        Pad the image with zeros
        :param padding_size_x: The padding size in the x-direction
        :param padding_size_y: The padding size in the y-direction
        :return: The padded image
        """
        self.data = cv2.copyMakeBorder(self.data, padding_size_x, padding_size_x, padding_size_y, padding_size_y,
                                       cv2.BORDER_REFLECT)
        self.pitch = self.width + 2 * padding_size_x

    def save_image(self, output_path: str):
        """
        Save the image to a file
        :param output_path: The path to save the image
        """
        cv2.imwrite(output_path, self.data)

    def show(self):
        """
        Display the image
        """
        cv2.imshow(self.name, self.data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
