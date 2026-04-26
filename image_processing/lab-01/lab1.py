from os.path import getsize
import numpy as np
import cv2

# 1
impath = "image.jpg"
image = cv2.imread(impath)

# 2
cv2.imwrite("1_image.bmp", image)
cv2.imwrite("2_image.png", image)
print(f'Изображения сохранены')

# 3
filesize = getsize(impath)
filesize2 = getsize("1_image.bmp")
h, w, _ = image.shape
bit = 24
k1 = ((w * h * bit) / 8) / filesize
k2 = ((w * h * bit) / 8) / filesize2
print(h, w, k1, k2)
print(filesize, filesize2)
print(f'(3)\nСтепень сжатия JPG -> {k1}')
print(f'Степень сжатия BMP -> {k2}')

# 4
b, g, r = cv2.split(image)
cv2.imwrite("4_R_image.png", r)
cv2.imwrite("4_G_image.png", g)
cv2.imwrite("4_B_image.png", b)

# 5
image_g = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
cv2.imwrite("5_GREY_image.png", image_g)

# 6
_, binary = cv2.threshold(image_g, 64, 255, cv2.THRESH_BINARY)
cv2.imwrite("6_64_THRESH_image.png", binary)
_, binary = cv2.threshold(image_g, 128, 255, cv2.THRESH_BINARY)
cv2.imwrite("6_128_THRESH_image.png", binary)
_, binary = cv2.threshold(image_g, 192, 255, cv2.THRESH_BINARY)
cv2.imwrite("6_192_THRESH_image.png", binary)

# 7
mirror = cv2.flip(image, 1)
cv2.imwrite("7_MIRROR_image.png", mirror)

# 8
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite("8_ROTATED.png", rotated)


# 9
def block_img(img, size):
    h, w = img.shape
    img_trim = img[:h // size * size, :w // size * size]

    blocks = img_trim.reshape(h // size, size, w // size, size)
    blocks = blocks.swapaxes(1, 2)

    block_means = blocks.mean(axis=(2, 3))

    result = np.repeat(np.repeat(block_means, size, axis=0), size, axis=1)

    return result.astype(np.uint8)


blocks = [5, 10, 20, 50]
for bl in blocks:
    disc = block_img(image_g, bl)
    cv2.imwrite(f"9_DISCRET{bl}x{bl}.png", disc)

#10
def quant_img(img, levels):
    step = 256 // levels
    quantized = (np.floor(img.astype(np.float32) / step) * step).astype(np.uint8)
    return quantized

levels = [4, 16, 32, 64, 128]
for lev in levels:
    quantized = quant_img(image_g, lev)
    cv2.imwrite(f"10_QUANT_{lev}.png", quantized)


#11
h, w = image_g.shape
start_h, start_w = (h-100)//2, (w-100)//2
crop = image_g[start_h:start_h+100, start_w:start_w+100]
cv2.imwrite("11_CENTER_100.png", crop)

#12
resized_crop = cv2.resize(crop, (300, 300), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("12_CENTER_300.png", resized_crop)

#12345
h, w = image.shape[:2]
start_h, start_w = (h-600)//2, (w-600)//2
image[start_h:start_h+600, start_w:start_w+600] = (0, 0, 255)
cv2.imwrite("NEW.png", image)

#dop
size = 100
im = cv2.rectangle(image, pt1=(0, 0), pt2=(size, size), color=(255, 0, 0), thickness=-1)
im = cv2.rectangle(image, pt1=(w - size, 0), pt2=(w, size), color=(255, 0, 0), thickness=-1)
im = cv2.rectangle(image, pt1=(0, h - size), pt2=(size, h), color=(255, 0, 0), thickness=-1)
im = cv2.rectangle(image, pt1=(w - size, h - size), pt2=(w, h), color=(255, 0, 0), thickness=-1)
cv2.imwrite('13_BLUE.png', im)

imge = image[0:50, ]


