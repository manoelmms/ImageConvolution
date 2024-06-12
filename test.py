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


# TODO: Implement the tests here!
class Corretude(unittest.TestCase):
    def test_concurrent_color(self):
        width, height = range(256, 1333, 256), range(128, 777, 128)
        # start_list = (0, 0, 192)
        # stop_list = (255, 255, 64)
        # is_horizontal_list = [True, False, False]
        images = []
        for w, h in zip(width, height):
            images.append(create_color_image(w, h))

        print("Começando testes de corretude")
        print("Testando imagens de tamanho", [(img.shape[0], img.shape[1]) for img in images])

        for img in images:
            print("Testando imagem de tamanho", img.shape)
            self.assertEqual(convolution.convolution(img, kernel.edge_detection_kernel(3)).all(),
                             convolution.convolution_mp_pool_01(img, kernel.edge_detection_kernel(3)).all())
            print(" -mp_pool_01 OK")
            # self.assertEqual(convolution.convolution(img, kernel.edge_detection_kernel(3)).all(),
            #                  convolution.convolution_mp_pool_02(img, kernel.edge_detection_kernel(3)).all())
            # print(" -mp_pool_02 OK")
            self.assertEqual(convolution.convolution(img, kernel.edge_detection_kernel(3)).all(),
                             convolution.convolution_mp_pool_03(img, kernel.edge_detection_kernel(3)).all())
            print(" -mp_pool_03 OK")
            self.assertEqual(convolution.convolution(img, kernel.edge_detection_kernel(3)).all(),
                             convolution.convolution_mp_pool_04(img, kernel.edge_detection_kernel(3)).all())
            print(" -mp_pool_04 OK")
            self.assertEqual(convolution.convolution(img, kernel.edge_detection_kernel(3)).all(),
                             convolution.convolution_numba(img, kernel.edge_detection_kernel(3)).all())
            print(" -numba OK")
            self.assertEqual(convolution.convolution(img, kernel.edge_detection_kernel(3)).all(),
                             convolution.convolution_th_pool_01(img, kernel.edge_detection_kernel(3)).all())
            print(" -th_pool_01 OK")

    def test_concurrent_bw(self):
        width, height = range(500, 1333, 256), range(500, 1000, 128)
        images = []
        for w, h in zip(width, height):
            images.append(create_bw_image(w, h))

        print("Começando testes de corretude")
        print("Testando imagens de tamanho", [(img.shape[0], img.shape[1]) for img in images])

        for img in images:
            print("Testando imagem de tamanho", img.shape)
            self.assertEqual(convolution.convolution(img, kernel.edge_detection_kernel(3)).all(),
                             convolution.convolution_mp_pool_01(img, kernel.edge_detection_kernel(3)).all())
            print(" -mp_pool_01 OK")
            # self.assertEqual(convolution.convolution(img, kernel.edge_detection_kernel(3)).all(),
            #                  convolution.convolution_mp_pool_02(img, kernel.edge_detection_kernel(3)).all())
            # print(" -mp_pool_02 OK")
            self.assertEqual(convolution.convolution(img, kernel.edge_detection_kernel(3)).all(),
                             convolution.convolution_mp_pool_03(img, kernel.edge_detection_kernel(3)).all())
            print(" -mp_pool_03 OK")
            self.assertEqual(convolution.convolution(img, kernel.edge_detection_kernel(3)).all(),
                             convolution.convolution_mp_pool_04(img, kernel.edge_detection_kernel(3)).all())
            print(" -mp_pool_04 OK")
            self.assertEqual(convolution.convolution(img, kernel.edge_detection_kernel(3)).all(),
                             convolution.convolution_numba(img, kernel.edge_detection_kernel(3)).all())
            print(" -numba OK")
            self.assertEqual(convolution.convolution(img, kernel.edge_detection_kernel(3)).all(),
                             convolution.convolution_th_pool_01(img, kernel.edge_detection_kernel(3)).all())
            print(" -th_pool_01 OK")


class Performance(unittest.TestCase):

    def test_performance_bw(self):
        width, height = range(500, 10000, 2500), range(500, 10000, 2500)
        images = []
        for w, h in zip(width, height):
            images.append(create_bw_image(w, h))

        print("Começando testes de performance")
        print("Testando imagens de tamanho", [(img.shape[0], img.shape[1]) for img in images])

        for img in images:
            print("Testando imagem de tamanho", img.shape)
            start = time.time()
            convolution.convolution(img, kernel.edge_detection_kernel(3))
            print(" -sequential:", time.time() - start)

            start = time.time()
            convolution.convolution_mp_pool_01(img, kernel.edge_detection_kernel(3))
            print(" -mp_pool_01:", time.time() - start)

            # start = time.time()
            # convolution.convolution_mp_pool_02(img, kernel.edge_detection_kernel(3))
            # print(" -mp_pool_02:", time.time() - start)

            start = time.time()
            convolution.convolution_mp_pool_03(img, kernel.edge_detection_kernel(3))
            print(" -mp_pool_03:", time.time() - start)

            start = time.time()
            convolution.convolution_mp_pool_04(img, kernel.edge_detection_kernel(3))
            print(" -mp_pool_04:", time.time() - start)

            start = time.time()
            convolution.convolution_numba(img, kernel.edge_detection_kernel(3))
            print(" -numba:", time.time() - start)

            # start = time.time()
            # convolution.convolution_th_pool_01(img, kernel.edge_detection_kernel(3)) # HIGH MEMORY USAGE
            # print(" -th_pool_01:", time.time() - start)


if __name__ == '__main__':
    unittest.main()
