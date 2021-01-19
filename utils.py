import numpy as np
from tifffile import imread, imshow
from skimage.morphology import dilation, disk, erosion, closing


# def create_mask(img, stride=27, threshold=0.7):
#     mask = img == 0
#     for i in range(0, mask.shape[0], stride):
#         for j in range(0, mask.shape[1], stride):
#             if np.mean(mask[i:i+stride,j:j+stride]) > threshold:
#                 mask[i:i+stride,j:j+stride] = 0
# #             mask[i:i+stride,j:j+stride] = (1 - np.min(mask[i:i+stride,j:j+stride])) * mask[i:i+stride,j:j+stride]
#     return mask


def create_mask(img, alpha=13):
    mask = img != 0
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    # mask1 = dilation(mask, disk(alpha))
    # mask1 = erosion(mask1, disk(alpha + 1))
    mask1 = closing(mask, disk(alpha))
    return mask1 * ~mask


def create_patches(img, resolution, coverage=0.5):
    patches = []
    for i in range(0, img.shape[1] - resolution, int(resolution * coverage)):
        for j in range(0, img.shape[2] - resolution, int(resolution * coverage)):
            challenger = img[:, i:i + resolution, j:j + resolution]
            # if np.count_nonzero(challenger == 0) < resolution ** 2 * coverage:
            patches.append(challenger)
        patches.append(img[:, i:i + resolution, -resolution:])

    for j in range(0, img.shape[2] - resolution, int(resolution * coverage)):
        challenger = img[:, -resolution:, j:j + resolution]
        patches.append(challenger)
    patches.append(img[:, -resolution:, -resolution:])
    return patches


def parse_image(path, all_channels_last=False):
    im = np.load(path)
    img = im['arr_0']
    if all_channels_last:
        img = np.moveaxis(img, 0, 2)
        img = img.reshape(img.shape[0], img.shape[1], -1)
    return img


def main():
    print('Nothing here!')


if __name__ == '__main__':
    main()
