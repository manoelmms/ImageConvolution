import numpy as np
import cv2
from kernel import Kernel2D


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

    def padding(self, padding_size: int):
        """
        Pad the image with zeros
        :param padding_size: The size of the padding
        :return: The padded image
        """
        if self.channels == 1:
            self.pitch = self.width + 2 * padding_size  # Update the pitch
            self.data = cv2.copyMakeBorder(self.data, padding_size, padding_size, padding_size, padding_size,
                                           cv2.BORDER_REFLECT)  # Pad the image with reflection
        else:
            self.data = cv2.copyMakeBorder(self.data, padding_size, padding_size, padding_size, padding_size,
                                           cv2.BORDER_REFLECT)  # Pad the image with reflection

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

    def convolution(self, kernel: Kernel2D):
        """
        Perform a convolution operation on the image
        :param kernel: The kernel to perform the convolution
        :return: The convoluted image
        """

        # Initialize the new image
        new_image = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)

        # TODO: Optimize padding (without using memory!)
        # Pad the image
        padding_size = kernel.height // 2
        self.padding(padding_size)

        # Perform the convolution operation
        for y in range(self.height):
            for x in range(self.width):
                for c in range(self.channels):
                    new_image[y, x, c] = np.sum(
                        self.data[y:y + kernel.height, x:x + kernel.width, c] * kernel.kernel)

        return Image.from_data(new_image)
