import numpy as np
import matplotlib.pyplot as plt
import shutil


def plot_cnn_weights(input, output, w, h):
    with open(input, 'r') as f:
        contents = f.readlines()
    cc2 = 0
    matrices = []
    for i in range(len(contents)):
        if contents[i][0] == 'M':
            matrix_str = contents[i + 1:i + 6]
            n = matrix_str[0].count(',')
            matrix = np.zeros((n, n))
            for k in range(n):
                commas = [j for j in range(len(matrix_str[k])) if matrix_str[k].startswith(",", j)]
                for q in range(len(commas) - 1):
                    if q == 0:
                        matrix[k][q] = float(matrix_str[k][0:commas[0]])
                    matrix[k][q + 1] = float(matrix_str[k][(commas[q] + 2):commas[q + 1]])
            # normalizing the matrix
            matrix -= np.min(matrix)
            matrix /= np.max(matrix)

            cc2 += 1
            matrices.append(matrix)
    fix, axs = plt.subplots(w, h)
    for i in range(w):
        for j in range(h):
            axs[i, j].imshow(matrices[i + j * 6])
            axs[i, j].get_xaxis().set_visible(False)
            axs[i, j].get_yaxis().set_visible(False)
    plt.savefig(output, bbox_inches='tight', pad_inches=0)


def get_nearest_neighbours(image_loc, input, output):
    no_images = 6
    with open(input, 'r') as f:
        contents = f.readlines()
    for i in range(len(contents)):
        if contents[i][0] == 'N':
            img_num = contents[i][-2]
            if int(img_num) >= no_images:
                break
            matrix_str = contents[i + 1:i + no_images + 1]
            neighbours = []
            for j in range(no_images):
                colon = matrix_str[j].find(':')
                newl = matrix_str[j].find('\n')
                neighbours.append(str(int(matrix_str[j][(colon + 2):newl]) + 1).zfill(
                    5) + ".png")  # need to add one because names of files starts from 1
            cc = 0
            for n in neighbours:
                shutil.copy(image_loc + n, output + img_num + "/" + str(cc) + ".png")
                cc += 1
            shutil.copy(image_loc + str(int(img_num) + 1).zfill(5) + ".png", output + img_num + "/" + "img.png")
