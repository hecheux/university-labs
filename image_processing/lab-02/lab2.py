import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 2 (Темные участки)
def log_transform(img, c):
    img = img.astype(np.float32)
    result = c * np.log(img + 1)
    result = (result - result.min()) / (result.max() - result.min()) * 255
    return result.astype(np.uint8)

log_image = log_transform(image, 20)
cv2.imwrite("2_LOG.png", log_image)


# 3 (Яркость)
def power_transform(img, gamma):
    img = img.astype(np.float32) / 255
    result = img ** gamma
    result = result * 255
    return result.astype(np.uint8)

gamma_values = [0.1, 0.45, 5.0]
for gamma in gamma_values:
    power_image = power_transform(image, gamma)
    cv2.imwrite(f"3_GAMMA_{gamma}.png", power_image)


# 4
def erlang_noise(img, k, theta):
    noise = np.random.gamma(shape=k, scale=theta, size=img.shape)
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255)
    return noise.astype(np.uint8), noisy_img.astype(np.uint8)

noise_image, noisy_image = erlang_noise(gray, 2, 10)
cv2.imwrite("4_ERLANG.png", noise_image)
cv2.imwrite("4_NOISE.png", noisy_image)

plt.figure(figsize=(8, 5))
plt.hist(noise_image.ravel(), bins=256, range=(0, 255))
plt.title("Histogram")
plt.xlabel("Brightness")
plt.ylabel("Number of pixels")
plt.savefig("4_HISTOGRAM.png")
plt.close()


#переписать блюр
# 5 блюр
def mean_filter(img, size):
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    result = cv2.filter2D(img, -1, kernel)
    return result


mask_sizes = [3, 9, 15]

for size in mask_sizes:
    filtered = mean_filter(noisy_image, size)
    cv2.imwrite(f"5_FILTER_{size}x{size}.png", filtered)


# 6 Повышение резкости
def sharpen_image(img):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    result = cv2.filter2D(img, -1, kernel)
    return result
sharp_image = sharpen_image(gray)
cv2.imwrite("6_SHARPEN.png", sharp_image)


# 7 Выделение границ
# Робертс
def roberts_edge(img):
    kernel_x = np.array([
        [1, 0],
        [0, -1]
    ], dtype=np.float32)

    kernel_y = np.array([
        [0, 1],
        [-1, 0]
    ], dtype=np.float32)

    gx = cv2.filter2D(img.astype(np.float32), -1, kernel_x)
    gy = cv2.filter2D(img.astype(np.float32), -1, kernel_y)

    result = np.abs(gx) + np.abs(gy)
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

# Превитт
def prewitt_edge(img):
    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    kernel_y = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ], dtype=np.float32)

    gx = cv2.filter2D(img.astype(np.float32), -1, kernel_x)
    gy = cv2.filter2D(img.astype(np.float32), -1, kernel_y)

    result = np.abs(gx) + np.abs(gy)
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

#переписать собел
# Собель
def sobel_edge(img):
    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    kernel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    gx = cv2.filter2D(img.astype(np.float32), -1, kernel_x)
    gy = cv2.filter2D(img.astype(np.float32), -1, kernel_y)

    result = np.abs(gx) + np.abs(gy)
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


roberts_image = roberts_edge(gray)
prewitt_image = prewitt_edge(gray)
sobel_image = sobel_edge(gray)

cv2.imwrite("7_ROBERTS.png", roberts_image)
cv2.imwrite("7_PREWITT.png", prewitt_image)
cv2.imwrite("7_SOBEL.png", sobel_image)