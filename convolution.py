from numba import jit, prange, set_num_threads
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor


def calculate_convolution(image: np.ndarray, x, y, c, kernel: np.ndarray, new_image: np.ndarray = None):
    """
    Calculate the convolution of a pixel
    :param image: The image data
    :param x: The x-coordinate of the pixel
    :param y: The y-coordinate of the pixel
    :param c: The channel of the pixel
    :param kernel: The kernel to perform the convolution
    :param new_image: The new image to store the result
    :return: The convoluted pixel
    """

    '''
    Kernel example:

    -------------
    | 1 | 2 | 3 | 
    | 4 | 5 | 6 |
    | 7 | 8 | 9 |
    -------------
    x = 2
    y = 2

    x -> 1, 2, 3 := x - kernel.center[0] + i
    y -> 1, 2, 3 := y - kernel.center[1] + j
    '''
    kernel_height, kernel_width = kernel.shape
    kernel_center = (kernel_height // 2, kernel_width // 2)

    # Initialize the result
    result = 0

    # Iterate over the kernel
    for i in range(kernel_height):
        for j in range(kernel_width):
            x_k = x - kernel_center[0] + i
            y_k = y - kernel_center[1] + j
            result += image[x_k, y_k, c] * kernel[i, j]

    if new_image is not None:
        new_image[x, y, c] = result
    return result


def calculate_convolution_padless(image: np.ndarray, x, y, c, kernel: np.ndarray, new_image: np.ndarray = None):
    """
    Calculate the convolution of a pixel without padding
    :param image: The image data
    :param x: The x-coordinate of the pixel
    :param y: The y-coordinate of the pixel
    :param c: The channel of the pixel
    :param kernel: The kernel to perform the convolution
    :param new_image: The new image to store the result
    :return: The convoluted pixel
    """

    kernel_height, kernel_width = kernel.shape
    kernel_center = (kernel_height // 2, kernel_width // 2)
    image_height, image_width, _ = image.shape

    # Initialize the result
    result = 0

    # Iterate over the kernel
    for i in range(kernel_height):
        for j in range(kernel_width):
            x_k = x - kernel_center[0] + i
            y_k = y - kernel_center[1] + j
            if x_k < 0 or y_k < 0 or x_k >= image_height or y_k >= image_width:
                continue
            result += image[x_k, y_k, c] * kernel[i, j]

    if new_image is not None:
        new_image[x, y, c] = result
    return result


def convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform a convolution operation on the image
    :param image:  data
    :param kernel: The kernel to perform the convolution
    :return: The convoluted image
    """
    image_height, image_width, image_channels = image.shape

    # Initialize the new image
    new_image = np.zeros(image.shape, dtype=np.uint8)

    # Perform the convolution operation
    for c in range(image_channels):
        for x in range(image_height):
            for y in range(image_width):
                calculate_convolution_padless(image, x, y, c, kernel, new_image)
    return new_image


def convolution_pool(image: np.ndarray, kernel: np.ndarray, num_processes: int) -> np.ndarray:
    """
    Perform a convolution operation on the image using multiple processes
    Strategy #1: using the Pool class
    :param image: The image data
    :param kernel: The kernel to perform the convolution
    :param num_processes: The number of processes to use
    :return: The convoluted image
    """
    image_height, image_width, image_channels = image.shape

    # Initialize the new image
    new_image = np.zeros(image.shape, dtype=np.uint8)

    # Perform the convolution operation in parallel using map
    with (mp.Pool(num_processes) as pool):
        for c in range(image_channels):
            new_image[:, :, c] = np.array(pool.starmap(calculate_convolution_padless,
                                                       [(image, x, y, c, kernel) for x in range(image_height)
                                                        for y in range(image_width)])).reshape(image_height,
                                                                                               image_width)
    return new_image


def calculate_convolution_block(image: np.ndarray, x_i, y_i, x_f, y_f, c, kernel: np.ndarray) -> np.ndarray:
    """
    Calculate the convolution of a pixel, return the block result instead of a single value
    :param image: The image data
    :param x_i: The initial x-coordinate of the block
    :param y_i: The initial y-coordinate of the block
    :param x_f: The final x-coordinate of the block
    :param y_f: The final y-coordinate of the block
    :param c: The channel of the pixel
    :param kernel: The kernel to perform the convolution
    :return: The convoluted block
    """

    kernel_height, kernel_width = kernel.shape
    kernel_center = (kernel_height // 2, kernel_width // 2)
    image_height, image_width, _ = image.shape

    # Initialize the result
    result = np.zeros((x_f - x_i, y_f - y_i))

    # Iterate over the kernel
    for i in range(x_f - x_i):
        for j in range(y_f - y_i):
            for k in range(kernel_height):
                for l in range(kernel_width):
                    x_k = x_i + i - kernel_center[0] + k
                    y_k = y_i + j - kernel_center[1] + l
                    if x_k < 0 or y_k < 0 or x_k >= image_height or y_k >= image_width:
                        continue
                    result[i, j] += image[x_k, y_k, c] * kernel[k, l]
    return result


def convolution_block(image: np.ndarray, kernel: np.ndarray, num_processes: int) -> np.ndarray:
    """
    Perform a convolution operation on the image using multiple processes
    Strategy #2: divide the image into chunks and process them in parallel, then merge the results
    :param image: The image data
    :param kernel: The kernel to perform the convolution
    :param num_processes: The number of processes to use
    :return: The convoluted image
    """
    image_height, image_width, image_channels = image.shape

    result = []
    new_image = np.empty(image.shape, dtype=np.uint8)

    # Divide the image into chunks and the rest of the division into the last chunk
    chunk_size = image_height // num_processes
    chunks = []
    for p in range(num_processes):
        chunks.append((p * chunk_size, (p + 1) * chunk_size if p != num_processes - 1 else image_height))

    # print(chunks)

    # Perform the convolution operation in parallel using map
    with (mp.Pool(num_processes) as pool):
        for c in range(image_channels):
            result.append(pool.starmap(calculate_convolution_block,
                                       [(image, x_i, 0, x_f, image_width, c, kernel) for x_i, x_f in chunks]))
            # concatenate the results
            # concatenate the results
            for i, chunk in enumerate(result):
                new_image[:, :, c] = np.vstack(chunk)
    return new_image


@jit(nopython=True, parallel=True)
def convolution_numba(image: np.ndarray, kernel: np.ndarray, num_processes: int) -> np.ndarray:
    """
    Perform a convolution operation on the image using multiple processes
    Strategy #5 : using numba.jit and parallel=True decorator and prange for parallelization
    :param image: The image data
    :param kernel: The kernel to perform the convolution
    :param num_processes: The number of processes to use
    :return: The convoluted image
    """
    set_num_threads(num_processes)
    image_height, image_width, image_channels = image.shape

    # Initialize the new image
    new_image = np.zeros(image.shape, dtype=np.uint8)

    # Perform the convolution operation in parallel using numba.jit
    for c in range(image_channels):
        for x in prange(image_height):
            for y in prange(image_width):
                for j in range(kernel.shape[0]):
                    for k in range(kernel.shape[1]):
                        x_k = x - j + kernel.shape[0] // 2
                        y_k = y - k + kernel.shape[1] // 2
                        if x_k < 0 or y_k < 0 or x_k >= image_height or y_k >= image_width:
                            continue
                        new_image[x, y, c] += image[x_k, y_k, c] * kernel[j, k]
    return new_image


def convolution_thread(image: np.ndarray, kernel: np.ndarray, num_processes: int) -> np.ndarray:
    """
    Perform a convolution operation on the image using multiple threads
    Strategy #1: using the ThreadPoolExecutor and saving directly to the new image
    :param image: The image data
    :param kernel: The kernel to perform the convolution
    :param num_processes: The number of processes to use
    :return: The convoluted image
    """
    image_height, image_width, image_channels = image.shape

    # Initialize the new image
    new_image = np.zeros(image.shape, dtype=np.uint8)

    # Perform the convolution operation in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(num_processes) as executor:
        for c in range(image_channels):
            executor.map(calculate_convolution_padless,
                         [(image, x, y, c, kernel, new_image)
                          for x in range(image_height) for y in range(image_width)])
    return new_image
