import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


IMAGE_PATH = 'image.jpeg'
D0_VALUES = [5, 10, 50, 250]
BUTTERWORTH_N = 10

SAVE_FOLDER = 'results'
os.makedirs(SAVE_FOLDER, exist_ok=True)

#dop funcs
def normalize_to_uint8(img):
    img = np.abs(img)
    img = img - img.min()
    if img.max() != 0:
        img = img / img.max() * 255
    return img.astype(np.uint8)


def save_gray(path, img):
    cv2.imwrite(path, normalize_to_uint8(img))

#image f-spectrum
def get_spectrum_image(fshift):
    spectrum = 20 * np.log(np.abs(fshift) + 1)
    return normalize_to_uint8(spectrum)


def get_distance_matrix(rows, cols):
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    center_row = rows // 2
    center_col = cols // 2
    D = np.sqrt((U - center_row) ** 2 + (V - center_col) ** 2)
    return D


#filters
def ideal_low_pass(D, D0):
    H = np.zeros_like(D, dtype=np.float32)
    H[D <= D0] = 1
    return H


def butterworth_low_pass(D, D0, n=2):
    H = 1 / (1 + (D / D0) ** (2 * n))
    return H.astype(np.float32)


def gaussian_low_pass(D, D0):
    H = np.exp(-(D ** 2) / (2 * (D0 ** 2)))
    return H.astype(np.float32)


def ideal_high_pass(D, D0):
    H = np.ones_like(D, dtype=np.float32)
    H[D <= D0] = 0
    return H


#Почему происходит наложение
#Почему и зачем дополняем нулями.
#Как выглядят все дискретные функции
#Почему низкие частоты слева, зачем мы делаем отступы

#результат идеальный подобрать n (с такими же арт.)
def butterworth_high_pass(D, D0, n=2):
    eps = 1e-6
    H = 1 / (1 + (D0 / (D + eps)) ** (2 * n))
    return H.astype(np.float32)


def gaussian_high_pass(D, D0):
    H = 1 - np.exp(-(D ** 2) / (2 * (D0 ** 2)))
    return H.astype(np.float32)


# 3d graph
def save_filter_3d(H, title, save_path):
    rows, cols = H.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, H, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('v')
    ax.set_ylabel('u')
    ax.set_zlabel('H(u, v)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


#primen
def apply_filter(fshift, H):
    filtered_shift = fshift * H
    #(sn shift, obrpfur)
    filtered_img = np.fft.ifft2(np.fft.ifftshift(filtered_shift))
    filtered_img = np.abs(filtered_img)
    return filtered_shift, filtered_img


#download
image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f'Не найден файл: {IMAGE_PATH}')

rows, cols = image.shape
D = get_distance_matrix(rows, cols)


# f - spectrum
fft = np.fft.fft2(image)
fshift = np.fft.fftshift(fft)
spectrum_image = get_spectrum_image(fshift)

cv2.imwrite(os.path.join(SAVE_FOLDER, '00_original_image.png'), image)
cv2.imwrite(os.path.join(SAVE_FOLDER, '01_fourier_spectrum.png'), spectrum_image)


filters = [
    ('ideal_low', ideal_low_pass),
    ('butterworth_low', lambda D, D0: butterworth_low_pass(D, D0, BUTTERWORTH_N)),
    ('gaussian_low', gaussian_low_pass),
    ('ideal_high', ideal_high_pass),
    ('butterworth_high', lambda D, D0: butterworth_high_pass(D, D0, BUTTERWORTH_N)),
    ('gaussian_high', gaussian_high_pass),
]


#main
for filter_name, filter_func in filters:
    filter_folder = os.path.join(SAVE_FOLDER, filter_name)
    os.makedirs(filter_folder, exist_ok=True)

    for D0 in D0_VALUES:
        H = filter_func(D, D0)

        #spectr filter and obrfur
        filtered_shift, filtered_img = apply_filter(fshift, H)

        #images
        kernel_image = normalize_to_uint8(H)
        filtered_spectrum_image = get_spectrum_image(filtered_shift)
        restored_image = normalize_to_uint8(filtered_img)

        cv2.imwrite(os.path.join(filter_folder, f'D0_{D0}_kernel.png'), kernel_image)
        cv2.imwrite(os.path.join(filter_folder, f'D0_{D0}_filtered_spectrum.png'), filtered_spectrum_image)
        cv2.imwrite(os.path.join(filter_folder, f'D0_{D0}_restored.png'), restored_image)

        # 3d
        title = f'{filter_name}, D0 = {D0}'
        plot_path = os.path.join(filter_folder, f'D0_{D0}_3d.png')
        save_filter_3d(H, title, plot_path)

print('Готово. Все результаты сохранены в папку:', SAVE_FOLDER)