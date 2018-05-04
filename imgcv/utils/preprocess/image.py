from __future__ import print_function

import os
import sys
import functools

import tensorflow as tf  # pylint: disable=g-bad-import-order

from imgcv.utils.preprocess import box

VERTICAL_TOP    = 0
VERTICAL_CENTER = 1
VERTICAL_BOTTOM = 2

HORIZONTAL_LEFT     = 0
HORIZONTAL_CENTER   = 1
HORIZONTAL_RIGHT    = 2

def decode_crop_and_flip(image_buffer, bbox, num_channels):
    """Crops the given image to a random part of the image, and randomly flips.

    We use the fused decode_and_crop op, which performs better than the two ops
    used separately in series, but note that this requires that the image be
    passed in as an un-decoded string Tensor.

    Args:
        image_buffer: scalar string Tensor representing the raw JPEG image buffer.
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged as
            [ymin, xmin, ymax, xmax].
        num_channels: Integer depth of the image buffer for decoding.

    Returns:
        3-D tensor with cropped image.

    """
    # A large fraction of image datasets contain a human-annotated bounding box
    # delineating the region of the image containing the object of interest.  We
    # choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image_buffer),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Reassemble the bounding box in the format the crop op requires.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

    # Use the fused decode and crop op here, which is faster than each in series.
    cropped = tf.image.decode_and_crop_jpeg(
    image_buffer, crop_window, channels=num_channels)

    # Flip to add a little more random distortion in.
    cropped = tf.image.random_flip_left_right(cropped)
    return cropped

def central_crop(image, crop_height, crop_width):
    """Performs central crops of the given image list.

    Args:
        image: a 3-D image tensor
        crop_height: the height of the image following the crop.
        crop_width: the width of the image following the crop.

    Returns:
        3-D tensor with cropped image.
    """
    return region_crop(image, crop_height, crop_width)

def top_left_crop(image, crop_height, crop_width):
    return region_crop(image, crop_height, crop_width, vertical=VERTICAL_TOP, horizontal=HORIZONTAL_LEFT)

def bottom_left_crop(image, crop_height, crop_width):
    return region_crop(image, crop_height, crop_width, vertical=VERTICAL_BOTTOM, horizontal=HORIZONTAL_LEFT)

def top_right_crop(image, crop_height, crop_width):
    return region_crop(image, crop_height, crop_width, vertical=VERTICAL_TOP, horizontal=HORIZONTAL_RIGHT)

def bottom_right_crop(image, crop_height, crop_width):
    return region_crop(image, crop_height, crop_width, vertical=VERTICAL_BOTTOM, horizontal=HORIZONTAL_RIGHT)

def region_crop(image, crop_height, crop_width, vertical=VERTICAL_CENTER, horizontal=HORIZONTAL_CENTER):
    """Performs central crops of the given image list.

    Args:
        image: a 3-D image tensor
        crop_height: the height of the image following the crop.
        crop_width: the width of the image following the crop.

    Returns:
        3-D tensor with cropped image.
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    if vertical == VERTICAL_CENTER:
        crop_top = amount_to_be_cropped_h // 2
    elif vertical == VERTICAL_TOP:
        crop_top = 0
    else:
        crop_top = amount_to_be_cropped_h

    amount_to_be_cropped_w = (width - crop_width)
    if horizontal == HORIZONTAL_CENTER:
        crop_left = amount_to_be_cropped_w // 2
    elif horizontal == HORIZONTAL_LEFT:
        crop_left = 0
    else:
        crop_left = amount_to_be_cropped_w

    return tf.slice(
        image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

def mean_image_subtraction(image, means, num_channels):
    """Subtracts the given means from each image channel.

    For example:
        means = [123.68, 116.779, 103.939]
        image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
        image: a tensor of size [height, width, C].
        means: a C-vector of values to subtract from each channel.
        num_channels: number of color channels in the image that will be distorted.

    Returns:
        the centered image.

    Raises:
        ValueError: If the rank of `image` is unknown, if `image` has a rank other
            than three or if the number of channels in `image` doesn't match the
            number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    # We have a 1-D tensor of means; convert to 3-D.
    means = tf.expand_dims(tf.expand_dims(means, 0), 0)

    return image - means

def smallest_size_at_least(height, width, resize_min):
    """Computes new shape with the smallest side equal to `smallest_side`.

    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.

    Args:
        height: an int32 scalar tensor indicating the current height.
        width: an int32 scalar tensor indicating the current width.
        resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
        new_height: an int32 scalar tensor indicating the new height.
        new_width: an int32 scalar tensor indicating the new width.
    """
    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return new_height, new_width

def aspect_preserving_resize(image, resize_min):
    """Resize images preserving the original aspect ratio.

    Args:
        image: A 3-D image `Tensor`.
        resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
        resized_image: A 3-D tensor containing the resized image.
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    new_height, new_width = smallest_size_at_least(height, width, resize_min)

    return resize_image(image, new_height, new_width)

def resize_image(image, height, width):
    """Simple wrapper around tf.resize_images.

    This is primarily to make sure we use the same `ResizeMethod` and other
    details each time.

    Args:
        image: A 3-D image `Tensor`.
        height: The target height for the resized image.
        width: The target width for the resized image.

    Returns:
        resized_image: A 3-D tensor containing the resized image. The first two
          dimensions have the shape [height, width].
    """
    return tf.image.resize_images(
        image, [height, width], method=tf.image.ResizeMethod.BILINEAR,
        align_corners=False)

def random_horizontal_flip(image,
                        boxes=None,
                        seed=None,
                        ):
    """Randomly flips the image and detections horizontally.

    The probability of flipping the image is 50%.

    Args:
        image: rank 3 float32 tensor with shape [height, width, channels].
        boxes: (optional) rank 2 float32 tensor with shape [N, 4]
                     containing the bounding boxes.
                     Boxes are in normalized form meaning their coordinates vary
                     between [0, 1].
                     Each row is in the form of [ymin, xmin, ymax, xmax].
        masks: (optional) rank 3 float32 tensor with shape
                     [num_instances, height, width] containing instance masks. The masks
                     are of the same height, width as the input `image`.
        keypoints: (optional) rank 3 float32 tensor with shape
                             [num_instances, num_keypoints, 2]. The keypoints are in y-x
                             normalized coordinates.
        keypoint_flip_permutation: rank 1 int32 tensor containing the keypoint flip
                                                             permutation.
        seed: random seed
        preprocess_vars_cache: PreprocessorCache object that records previously
                                                     performed augmentations. Updated in-place. If this
                                                     function is called multiple times with the same
                                                     non-null cache, it will perform deterministically.

    Returns:
        image: image which is the same shape as input image.

        If boxes, masks, keypoints, and keypoint_flip_permutation are not None,
        the function also returns the following tensors.

        boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
                     Boxes are in normalized form meaning their coordinates vary
                     between [0, 1].
        masks: rank 3 float32 tensor with shape [num_instances, height, width]
                     containing instance masks.
        keypoints: rank 3 float32 tensor with shape
                             [num_instances, num_keypoints, 2]

    Raises:
        ValueError: if keypoints are provided but keypoint_flip_permutation is not.
    """

    def _flip_image(image):
        # flip image
        image_flipped = tf.image.flip_left_right(image)
        return image_flipped

    with tf.name_scope('RandomHorizontalFlip', values=[image, boxes]):
        result = []
        # random variable defining whether to do flip or not
        generator_func = functools.partial(tf.random_uniform, [], seed=seed)
        do_a_flip_random = generator_func()
        do_a_flip_random = tf.greater(do_a_flip_random, 0.5)

        # flip image
        image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)
        result.append(image)

        # flip boxes
        if boxes is not None:
            boxes = tf.cond(do_a_flip_random, lambda: box.flip_boxes_left_right(boxes),
                                            lambda: boxes)
            result.append(boxes)

        return tuple(result)

def strict_random_crop_image(image, boxes, labels,
                            min_object_covered=1.0,
                            aspect_ratio_range=(0.75, 1.33),
                            area_range=(0.1, 1.0),
                            overlap_thresh=0.3):
    with tf.name_scope('RandomCropImage', values=[image, boxes]):
        image_shape = tf.shape(image)

        # boxes are [N, 4]. Lets first make them [N, 1, 4].
        boxes_expanded = tf.expand_dims(
                tf.clip_by_value(boxes, clip_value_min=0.0, clip_value_max=1.0), 1)

        generator_func = functools.partial(
                tf.image.sample_distorted_bounding_box,
                image_shape,
                bounding_boxes=boxes_expanded,
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=100,
                use_image_if_no_bounding_boxes=True)

        im_box_begin, im_box_size, im_box = generator_func()

        new_image = tf.slice(image, im_box_begin, im_box_size)
        new_image.set_shape([None, None, image.get_shape()[2]])

        # [1, 4]
        im_box_rank2 = tf.squeeze(im_box, squeeze_dims=[0])
        # [4]
        im_box_rank1 = tf.squeeze(im_box)

        # remove boxes that are outside cropped image
        boxes, inside_window_ids = box.prune_completely_outside_window(
                boxes, im_box_rank1)
        labels = tf.gather(labels, inside_window_ids )

        # remove boxes that are outside image
        overlapping_boxes, keep_ids = box.prune_non_overlapping_boxes(
                boxes, im_box_rank2, overlap_thresh)
        labels = tf.gather(labels, keep_ids)

        # change the coordinate of the remaining boxes
        new_labels = labels
        new_boxes = box.change_coordinate_frame(overlapping_boxes, im_box_rank1)
        new_boxes = tf.clip_by_value(
                new_boxes, clip_value_min=0.0, clip_value_max=1.0)

        result = [new_image, new_boxes, new_labels]

        return tuple(result)

