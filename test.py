import unittest
import cv2
import numpy as np
import convolution
import kernel
import image
import time


def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.uint8)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result


def create_bw_image(width, height, path=None):
    data = np.random.randint(0, 255, (height, width, 1), np.uint8)
    img = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    if path:
        cv2.imwrite(path, img)
    return data


def create_color_image(width, height, path=None):
    data = np.random.randint(0, 255, (height, width, 3), np.uint8)
    img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)  # Convert to RGB
    if path:
        cv2.imwrite(path, img)
    return data


def create_test_images(width: list, height: list, path: str, is_bw: bool):
    images = []
    for w, h in zip(width, height):
        path_img = f"{path}_{is_bw}_{w}_{h}.jpg"
        if is_bw:
            img = create_bw_image(w, h, path_img)
        else:
            img = create_color_image(w, h, path_img)
        images.append(img)
    return images


class Corretude(unittest.TestCase):

    def test_concurrent_color(self):
        width = [333, 500, 1024]
        height = [333, 500, 768]
        threads = [1, 2, 4, 8]
        images = []
        for w, h in zip(width, height):
            images.append(image.Image(f"./images/corretude_False_{w}_{h}.jpg").data)

        print("Começando testes de corretude")
        print("Testando imagens de tamanho", [(img.shape[0], img.shape[1]) for img in images])

        for thread in threads:
            for img in images:
                print("Testando imagem de tamanho", img.shape)
                print("Testando com thread = ", thread)

                sequential = convolution.convolution(img, kernel.edge_detection_kernel(3))
                self.assertEqual(sequential.all(),
                                 convolution.convolution_pool(img, kernel.edge_detection_kernel(3), thread).all())
                print(" -pool OK")

                self.assertEqual(sequential.all(),
                                 convolution.convolution_block(img, kernel.edge_detection_kernel(3), thread).all())
                print(" -block OK")

                self.assertEqual(sequential.all(),
                                 convolution.convolution_thread(img, kernel.edge_detection_kernel(3), thread).all())
                print(" -thread OK")

                self.assertEqual(sequential.all(),
                                 convolution.convolution_numba(img, kernel.edge_detection_kernel(3), thread).all())
                print(" -numba OK")

    def test_concurrent_bw(self):
        width = [333, 500, 1024]
        height = [333, 500, 768]
        threads = [1, 2, 4, 8]
        images = []
        for w, h in zip(width, height):
            images.append(image.Image(f"./images/corretude_True_{w}_{h}.jpg").data)

        print("Começando testes de corretude")
        print("Testando imagens de tamanho", [(img.shape[0], img.shape[1]) for img in images])

        for thread in threads:
            for img in images:
                print("Testando imagem de tamanho", img.shape)
                print("Testando com thread = ", thread)

                sequential = convolution.convolution(img, kernel.edge_detection_kernel(3))
                self.assertEqual(sequential.all(),
                                 convolution.convolution_pool(img, kernel.edge_detection_kernel(3), thread).all())
                print(" -pool OK")

                self.assertEqual(sequential.all(),
                                 convolution.convolution_block(img, kernel.edge_detection_kernel(3), thread).all())
                print(" -block OK")

                self.assertEqual(sequential.all(),
                                 convolution.convolution_thread(img, kernel.edge_detection_kernel(3), thread).all())
                print(" -thread OK")

                self.assertEqual(sequential.all(),
                                 convolution.convolution_numba(img, kernel.edge_detection_kernel(3), thread).all())
                print(" -numba OK")


class Performance(unittest.TestCase):

    def test_performance_bw(self):
        width = [500, 1000, 5000]
        height = [500, 1000, 5000]
        threads = [1, 2, 4, 8]
        images = []
        for w, h in zip(width, height):
            images.append(image.Image(f"./images/performance_True_{w}_{h}.jpg").data)

        print("Começando testes de performance")
        print("Testando imagens de tamanho", [(img.shape[0], img.shape[1]) for img in images])

        for thread in threads:
            for img in images:
                print("Testando imagem de tamanho", img.shape)
                print("Testando com thread = ", thread)

                start = time.time()
                convolution.convolution(img, kernel.edge_detection_kernel(3))
                print("Sequential:", time.time() - start)

                start = time.time()
                convolution.convolution_pool(img, kernel.edge_detection_kernel(3), thread)
                print("Pool:", time.time() - start)

                start = time.time()
                convolution.convolution_block(img, kernel.edge_detection_kernel(3), thread)
                print("Block:", time.time() - start)

                start = time.time()
                convolution.convolution_thread(img, kernel.edge_detection_kernel(3), thread)
                print("Thread:", time.time() - start)

                start = time.time()
                convolution.convolution_numba(img, kernel.edge_detection_kernel(3), thread)
                print("Numba:", time.time() - start)


if __name__ == '__main__':
    create_test_images([333, 500, 1024], [333, 500, 768], "./images/corretude", True)
    create_test_images([333, 500, 1024], [333, 500, 768], "./images/corretude", False)
    create_test_images([500, 1000, 5000], [500, 1000, 5000], "./images/performance", True)
    # unittest.main()
