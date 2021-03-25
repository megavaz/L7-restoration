import numpy as np
from tqdm import tqdm
import argparse


def parse_image(path):
    extension = path.split('.')[-1]
    if extension == 'npz':
        im = np.load(path)
        img = im['arr_0']
    else:
        from tifffile import imread
        img = imread(path)
    return img, extension


def save_image(img, extension):
    if extension == 'npz':
        np.savez('mask.' + extension, img)
    else:
        from tifffile import imsave
        imsave('mask.' + extension, img)


def _create_mask(img, alpha=13):
    from skimage.morphology import disk, closing
    mask_sup = img != 0
    mask = np.ones(mask_sup.shape[:-1], dtype=bool)
    if len(mask_sup.shape) == 3:
        for channel in range(mask_sup.shape[2]):
            mask = mask * mask_sup[:, :, channel]
    else:
        mask = mask_sup
    mask1 = closing(mask, disk(alpha))
    return mask1 * ~mask


def create_mask(img_volume, full=False, silent=False):
    mask_equalize = img_volume[:, :, :, :] != 0
    mask_volume = np.ones(img_volume.shape[:-1], dtype=bool)
    for i in range(mask_equalize.shape[-1]):
        mask_volume = mask_volume * mask_equalize[:, :, :, i]
    if full:
        for i in tqdm(range(0, len(mask_volume)), disable=silent):
            mask_volume[i] = _create_mask(mask_volume[i])
    else:
        mask_volume = ~mask_volume
    return mask_volume


def main():
    parser = argparse.ArgumentParser(
        description='This script creates proper mask for "l7_restoration.py". You need to provide'
                    'unaltered images in order for this script to work properly. If you provide it indexes'
                    'like NDVI created masks will be not perfect!'
    )
    parser.add_argument('path_to_file', type=str, help='Path to the file you want to restore. Currently supported '
                                                       'formats are .npz and .tif. Note that your file should have'
                                                       'shape (num_images, height, width, num_channels).')
    parser.add_argument('--full', action='store_true', help='This method works orders of magnitude slower.'
                                                            'Use it when there are lost pixels in the corner'
                                                            'of some images.')
    parser.add_argument('--silent', action='store_true')
    args = parser.parse_args()

    img_volume, extension = parse_image(args.path_to_file)
    mask_volume = create_mask(img_volume, full=args.full, silent=args.silent)
    save_image(mask_volume, extension)
    print('Mask is saved in file "mask.{}"'.format(extension))


if __name__ == '__main__':
    main()
