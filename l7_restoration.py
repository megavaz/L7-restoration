import numpy as np
from sklearn.cluster import DBSCAN
from tifffile import imsave, imread
from tqdm import tqdm
from skimage.morphology import dilation, disk
from catboost import Pool, CatBoostRegressor, CatBoostError
import argparse


def parse_image(path):
    extension = path.split('.')[-1]
    if extension == 'npz':
        im = np.load(path)
        img = im['arr_0']
    else:
        img = imread(path)
    return img, extension


def save_image(img, extension):
    if extension == 'npz':
        np.savez('res.' + extension, img)
    else:
        imsave('res.' + extension, img)


def check_intersection(img):
    mask = np.ones(img.shape[1:3], dtype=bool)
    for i in range(len(img)):
        msk = img[i] == 0
        mask = mask * msk
    return np.count_nonzero(mask)


def create_train_data(target_image, training_images, target_mask, train_mask, j, window_size, geospatial=True):
    target_image_window = target_image[:, j: j + window_size]
    target_mask_window = target_mask[:, j: j + window_size]
    if np.count_nonzero(target_mask_window) == 0:
        return 0, 0, 0, 0, 0
    train_mask_window = train_mask[:, j: j + window_size]
    training_images_window = training_images[:, :, j: j + window_size]
    train_data = np.array([training_images_window[m][train_mask_window] for m in range(len(training_images_window))])
    train_label = target_image_window[train_mask_window]
    train_data = np.moveaxis(train_data, 0, 1)
    if geospatial:
        coordinates = np.array(np.where(train_mask_window))
        coordinates = np.moveaxis(coordinates, 0, 1)
        train_data = np.concatenate([train_data, coordinates], axis=1)
    restore_data = np.array([training_images_window[m][target_mask_window] for m in range(len(training_images_window))])
    restore_data = np.moveaxis(restore_data, 0, 1)
    if geospatial:
        coordinates = np.array(np.where(target_mask_window))
        coordinates = np.moveaxis(coordinates, 0, 1)
        restore_data = np.concatenate([restore_data, coordinates], axis=1)
    return train_data, train_label, restore_data, target_mask_window, target_image_window


def create_train_data_quick(target_image, training_images, target_mask, train_mask, i, j, window_size, geospatial=True):
    target_image_window = target_image[i: i + window_size, j: j + window_size]
    target_mask_window = target_mask[i: i + window_size, j: j + window_size]
    if np.count_nonzero(target_mask_window) == 0:
        return 0, 0, 0, 0, 0
    train_mask_window = train_mask[i: i + window_size, j: j + window_size]
    training_images_window = training_images[:, i: i + window_size, j: j + window_size]
    train_data = np.array([training_images_window[m][train_mask_window] for m in range(len(training_images_window))])
    train_label = target_image_window[train_mask_window]
    train_data = np.moveaxis(train_data, 0, 1)
    if geospatial:
        coordinates = np.array(np.where(train_mask_window))
        coordinates = np.moveaxis(coordinates, 0, 1)
        train_data = np.concatenate([train_data, coordinates], axis=1)
    restore_data = np.array([training_images_window[m][target_mask_window] for m in range(len(training_images_window))])
    restore_data = np.moveaxis(restore_data, 0, 1)
    if geospatial:
        coordinates = np.array(np.where(target_mask_window))
        coordinates = np.moveaxis(coordinates, 0, 1)
        restore_data = np.concatenate([restore_data, coordinates], axis=1)
    return train_data, train_label, restore_data, target_mask_window, target_image_window


def create_order(img):
    number_of_skips = [np.count_nonzero(img[i] == 0) for i in range(img.shape[0])]
    order = [i for i in range(img.shape[0])]
    order = [x for x, _ in sorted(zip(order, number_of_skips), key=lambda pair: pair[1])]
    return order


def create_horizontal_iteration(img, mask, epsilon=1, n=5, total_max_segments=65, max_covered_length=35):
    total_max_segments -= 2
    horizontal_iteration = [0]
    sliding_average = img[mask][0]
    threshold = epsilon * np.std(img[mask])
    for j in range(0, img.shape[-1]):
        current_value = img[:, j][mask[:, j]]
        if current_value.size == 0:
            continue
        current_value = np.average(current_value)
        if np.abs(sliding_average - current_value) >= threshold:
            horizontal_iteration.append(j)
            sliding_average = current_value
        else:
            sliding_average = (sliding_average * (n - 1) + current_value) / n
        if j - horizontal_iteration[-1] == max_covered_length:
            horizontal_iteration.append(j)
    try:
        ratio = len(horizontal_iteration) / total_max_segments
    except ZeroDivisionError:
        return [0, 955]
    if ratio > 1:
        horizontal_iteration_temp = [0]
        main_number, lagging_number = 0, 0
        for elem in horizontal_iteration:
            main_number += 1
            if main_number // ratio > lagging_number:
                lagging_number += 1
                horizontal_iteration_temp.append(elem)
        horizontal_iteration = horizontal_iteration_temp
    horizontal_iteration.append(img.shape[-1])
    return horizontal_iteration


def _dummy_restore(img_volume, mask_volume):
    def roll_down(mask, i):
        for m in range(i, i + 30):
            if not mask[m]:
                return m
            return m

    for channel in range(img_volume.shape[3]):
        mask = img_volume[:, :, :, channel] == 0
        mask = mask * mask_volume
        if np.count_nonzero(mask) != 0:
            indexes = np.where(mask)
            (numbers, i_s, j_s) = indexes
            for number, i, j in zip(numbers.tolist(), i_s.tolist(), j_s.tolist()):
                img_volume[number, i, j, channel] = img_volume[number, i - 1, j, channel] if i - 1 > 0 else img_volume[
                    number, roll_down(mask_volume[number, :, j], i), j, channel]
    return img_volume


def _quick_restore(
        img_volume,
        mask_volume,
        restoration_order,
        window_size=479,
        min_value=0.,
        max_value=1.,
        verbose=True,
        geospatial=True
):

    model = CatBoostRegressor(
        learning_rate=0.2,
        depth=5,
        loss_function='RMSE',
        verbose=0,
        num_trees=300

    )

    for target_image_number in restoration_order:
        target_image = img_volume[target_image_number]
        target_mask = mask_volume[target_image_number]
        if np.count_nonzero(target_mask) == 0:
            continue
        train_mask = dilation(target_mask, disk(1))
        train_mask = train_mask ^ target_mask
        training_images = np.concatenate([img_volume[:target_image_number], img_volume[target_image_number + 1:]],
                                         axis=0)
        threshold = (target_image.shape[0] - window_size, target_image.shape[1] - window_size)
        for i in tqdm(range(0, target_image.shape[0], window_size), disable=not verbose):
            for j in range(0, target_image.shape[1], window_size):
                if i > threshold[0]:
                    i = threshold[0]
                if j > threshold[1]:
                    j = threshold[1]
                for channel in range(img_volume.shape[-1]):
                    train_data, train_label, restore_data, \
                    target_mask_window, target_image_window = create_train_data_quick(
                        target_image[:, :, channel],
                        training_images[:, :, :, channel],
                        target_mask,
                        train_mask, i, j,
                        window_size, geospatial=geospatial
                    )
                    if type(train_data) == int:
                        continue
                    train_pool = Pool(train_data, train_label)
                    try:
                        model.fit(train_pool)
                        restore_pool = Pool(restore_data)
                        res = model.predict(restore_pool)
                    except CatBoostError:
                        res = np.full((restore_data.shape[0],), np.average(train_label[0]))
                    target_image_window[target_mask_window] = res
                    img_volume[target_image_number, i:i + window_size, j: j + window_size,
                    channel] = target_image_window
    img_volume = np.nan_to_num(img_volume, nan=0.0)
    img_volume = np.clip(img_volume, min_value, max_value)
    return img_volume


def _restore_images(
        img_volume,
        mask_volume,
        restoration_order,
        epsilon=0.8,
        max_length=150,
        masks_to_sum=1,
        average_segment_length=318,
        min_value=0.,
        max_value=1.,
        verbose=True,
        geospatial=True
):
    total_max_segments = int(img_volume.shape[1] // average_segment_length)
    model = CatBoostRegressor(
        learning_rate=0.2,
        depth=4,
        loss_function='RMSE',
        verbose=0,
        num_trees=210

    )
    for target_image_number in restoration_order:
        target_image = img_volume[target_image_number]
        target_mask = mask_volume[target_image_number]
        if np.count_nonzero(target_mask) == 0:
            continue
        train_mask = dilation(target_mask, disk(1))
        restore_indexes = np.where(train_mask)
        indexes = np.moveaxis(np.array(restore_indexes), 0, 1)
        labels = DBSCAN(eps=1.5).fit_predict(indexes)
        train_masks = []
        target_masks = []
        for i in set(labels):
            ind = np.where(labels == i)
            train_mask_2 = np.zeros(shape=train_mask.shape, dtype=bool)
            rst = tuple((restore_indexes[0][ind], restore_indexes[1][ind]))
            train_mask_2[rst] = True
            target_masks.append(train_mask_2 * target_mask)
            train_masks.append((train_mask_2 ^ target_masks[-1]) * ~target_mask)
        lengths = []
        max_area = target_masks[0].shape[-1]
        train_masks_aux = [np.zeros(train_masks[0].shape, dtype=bool)]
        target_masks_aux = [np.zeros(train_masks[0].shape, dtype=bool)]
        i = 0
        for train_mask, target_mask in zip(train_masks, target_masks):
            if i < masks_to_sum:
                train_masks_aux[-1] = train_masks_aux[-1] + train_mask
                target_masks_aux[-1] = target_masks_aux[-1] + target_mask
                i += 1
            else:
                i = 1
                train_masks_aux.append(train_mask)
                target_masks_aux.append(target_mask)
        train_masks = train_masks_aux
        target_masks = target_masks_aux
        for _ in range(len(target_masks)):
            auxiliary = np.where(target_masks[_])[1]
            lengths.append(auxiliary[-1] - auxiliary[0])

        training_images = np.concatenate([img_volume[:target_image_number], img_volume[target_image_number + 1:]],
                                         axis=0)
        for train_mask_current, target_mask_current, current_area in tqdm(zip(train_masks, target_masks, lengths),
                                                                          total=len(target_masks), disable=not verbose):
            horizontal_iteration = create_horizontal_iteration(
                target_image[:, :, 0],
                train_mask_current,
                epsilon=epsilon,
                n=1,
                total_max_segments=int(np.round(total_max_segments * current_area / max_area + 1)),
                max_covered_length=max_length
            )
            for m in range(len(horizontal_iteration) - 1):
                for channel in range(img_volume.shape[-1]):
                    j = horizontal_iteration[m]
                    window_size = horizontal_iteration[m + 1] - j
                    train_data, train_label, restore_data, \
                    target_mask_window, target_image_window = create_train_data(
                        target_image[:, :, channel],
                        training_images[:, :, :, channel],
                        target_mask_current,
                        train_mask_current,
                        j, window_size, geospatial=geospatial
                    )
                    if type(train_data) == int:
                        continue
                    train_pool = Pool(train_data, train_label)
                    try:
                        model.fit(train_pool)
                        restore_pool = Pool(restore_data)
                        res = model.predict(restore_pool)
                    except CatBoostError:
                        res = np.full((restore_data.shape[0],), np.average(train_label[0]))
                    target_image_window[target_mask_window] = res
                    img_volume[target_image_number, :, j: j + window_size, channel] = target_image_window
    img_volume = np.nan_to_num(img_volume, nan=0.0)
    img_volume = np.clip(img_volume, min_value, max_value)
    return img_volume


def quick_restore(
        img_volume,
        mask_volume,
        range_object=None,
        min_value=0.,
        max_value=1.,
        verbose=True
):
    try:
        assert img_volume.shape[0] == mask_volume.shape[0]
    except AssertionError:
        print('First dimension of img_volume and mask_volume must be equal!')

    try:
        assert len(mask_volume.shape) == 3
    except AssertionError:
        print('Mask must have 3 dimensions (num_images, height, width)!')

    try:
        assert check_intersection(img_volume)
    except AssertionError:
        print('Your images do not fully cover all missing pixels! Add more images to your batch.')

    if len(img_volume.shape) == 3:
        img_volume = np.expand_dims(img_volume, -1)
    for i in range(img_volume.shape[-1]):
        img_volume[:, :, :, i][mask_volume] = 0
    if range_object is None:
        range_object = create_order(img_volume)
    img_volume = _dummy_restore(img_volume, mask_volume)
    img_volume = _quick_restore(
        img_volume,
        mask_volume,
        range_object,
        min_value=min_value,
        max_value=max_value,
        verbose=verbose
    )
    return img_volume


def full_restore(
        img_volume,
        mask_volume,
        range_object=None,
        min_value=0.,
        max_value=1.,
        no_quick=False,
        verbose=True
):
    if range_object is None:
        range_object = create_order(img_volume)
    img_volume = quick_restore(
        img_volume,
        mask_volume,
        range_object=[] if no_quick else None,
        min_value=min_value,
        max_value=max_value,
        verbose=verbose
    )
    img_volume = _restore_images(
        img_volume,
        mask_volume,
        range_object,
        min_value=min_value,
        max_value=max_value,
        verbose=verbose
    )
    return img_volume


def main():
    parser = argparse.ArgumentParser(
        description='This script performs restoration of the Landsat 7 images.'
    )
    parser.add_argument('path_to_file', type=str, help='Path to the file you want to restore. Currently supported '
                                                       'formats are .npz and .tif. Note that your file should have'
                                                       'shape (num_images, height, width, num_channels).')
    parser.add_argument('path_to_mask', type=str, help='Path to the mask. Note that the mask must be of'
                                                       'shape (num_images, height, width). You can create proper '
                                                       'mask with "create_mask.py".')
    parser.add_argument('--algorithm', type=str, default='quick', choices=['quick', 'full'],
                        help='Type of the algorithm to use.')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--order', type=int, action='extend', nargs='+',
                        help='You can specify which images you want to restore and the order of restoration.'
                             "If you don't specify this parameter, all of your images will be restored."
                             "Note that if you use full version of the algorithm even if you specify only "
                             "part of the images to be restored, still all of your images will be restored"
                             "but the full version of the algorithm will be applied only to specified images."
                             "If you don't" ' want this behavior use option "--noquick"')
    parser.add_argument('--noquick', action='store_true')
    parser.add_argument('--min', type=float, default=-1.)
    parser.add_argument('--max', type=float, default=1.)
    args = parser.parse_args()
    print(args)
    img_volume, extension = parse_image(args.path_to_file)
    mask_volume, _ = parse_image(args.path_to_mask)
    if args.algorithm == 'quick':
        img_volume = quick_restore(
            img_volume,
            mask_volume,
            range_object=args.order,
            min_value=args.min,
            max_value=args.max,
            verbose=not args.silent
        )
    elif args.algorithm == 'full':
        img_volume = full_restore(
            img_volume,
            mask_volume,
            range_object=args.order,
            min_value=args.min,
            max_value=args.max,
            verbose=not args.silent,
            no_quick=args.noquick
        )
    save_image(img_volume, extension)
    print('Restoration has ended! Result is saved in file "res.{}"'.format(extension))


if __name__ == '__main__':
    main()

