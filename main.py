import sys

import convolution
from image import Image
import kernel
import time


def main(argc, argv):
    if argc != 3:
        print("Usage: python main.py <input_image> <output_image>")
        raise ValueError("Invalid arguments")

    # Load the image
    image = Image(argv[1])

    print("Image loaded successfully from the path: ", argv[1])
    print("Image dimensions: ", image.height, "x", image.width, "x", " - ", image.channels)
    print("\nSelect the kernel you want to apply: ")
    print("1. Sharpen")
    print("2. Edge Detection")
    print("3. Emboss")
    print("4. Gaussian")
    kernel_choice = int(input("Enter the kernel number: "))

    # Create a kernel based on the user's choice
    match kernel_choice:
        case 1:
            selected_kernel = kernel.sharpen_kernel
        case 2:
            kernel_size = int(input("Enter the size of the kernel: "))
            selected_kernel = kernel.Kernel2D(kernel.edge_detection_kernel(kernel_size))
            # print(selected_kernel.kernel)
        case 3:
            selected_kernel = kernel.emboss_kernel
        case 4:
            kernel_size = int(input("Enter the size of the kernel: "))
            sigma = float(input("Enter the standard deviation of the Gaussian distribution: "))
            selected_kernel = kernel.Kernel2D(kernel.gaussian_kernel(kernel_size, sigma))
            # print(selected_kernel.kernel)
        case _:
            # selected_kernel = kernel.identity_kernel
            raise ValueError("Invalid kernel choice")

    # Apply the selected kernel and measure the time
    print("\nSelect the model you want to use: ")
    print("1. Sequential")
    print("2. Pool")
    print("3. Block")
    print("4. Numba")
    model_choice = int(input("Enter the model number: "))

    if model_choice == 1:
        start_time = time.time()
        print("Applying the kernel...")
        result_image = convolution.convolution(image.data, selected_kernel.kernel)
        end_time = time.time()
        result_image = Image.from_data(result_image)

    else:
        print("\nSelect the number of threads/processes you want to use: ")
        num_threads = int(input("Enter the number of threads/processes: "))
        if num_threads < 1:
            raise ValueError("Invalid number of threads/processes")

        start_time = time.time()
        print("Applying the kernel...")
        match model_choice:
            case 2:
                result_image = convolution.convolution_pool(image.data, selected_kernel.kernel, num_threads)
                end_time = time.time()
                result_image = Image.from_data(result_image)
            case 3:
                result_image = convolution.convolution_block(image.data, selected_kernel.kernel, num_threads)
                end_time = time.time()
                result_image = Image.from_data(result_image)
            case 4:
                result_image = convolution.convolution_numba(image.data, selected_kernel.kernel, num_threads)
                end_time = time.time()
                result_image = Image.from_data(result_image)
            case _:
                raise ValueError("Invalid model choice")

    print("Time taken to apply the kernel: ", end_time - start_time, " seconds")

    # Save the image
    result_image.save_image(argv[2])
    print("Image saved successfully to the path: ", argv[2])

    # Display the image
    result_image.show()


if __name__ == "__main__":

    main(len(sys.argv), sys.argv)
    # main(3, ["main.py", "images/dog.jpg", "output.jpg"])
