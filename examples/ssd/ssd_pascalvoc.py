from __future__ import print_function

import os
import sys
import collections
import functools

import tensorflow as tf  # pylint: disable=g-bad-import-order

from imgcv.dataset import DetectionDataSet
from imgcv.dataset import batcher
from imgcv import detection
from imgcv.models import ssd
from imgcv.models import vgg
from imgcv.utils import preprocess as pp
from imgcv.utils import ops

slim = tf.contrib.slim

SSD_VGG_SIZE = 300
NUM_CLASSES = 20
NUM_IMAGES = {
    'train': 5011,
    'test': 4952,
}

class PascalVocDataSet(DetectionDataSet):
    def __init__(self, flags):
        super(PascalVocDataSet, self).__init__(flags)

    def input_fn(self, mode, num_epochs=1):
        dataset = self.get_raw_dataset(mode, num_epochs)

        input_queue = self.batch_queue(dataset)
        batched_tensors = input_queue.dequeue()
        return batched_tensors, True

    def get_raw_dataset(self, mode, num_epochs=1):
        data_dir = os.path.expanduser(self.flags.data_dir)
        filenames = [
            os.path.join(data_dir, 'trainval_merge.record'),
        ]
        dataset = self.read_and_parse_dataset(filenames, num_epochs)
        return dataset

    def tf_summary_image(self, image, bboxes, name='image'):
        """Add image with bounding boxes to summary.
        """
        image = tf.to_float(tf.expand_dims(image, 0))
        bboxes = tf.expand_dims(bboxes, 0)
        image_with_box = tf.image.draw_bounding_boxes(image, bboxes)
        tf.summary.image(name, image_with_box)

    def batch_queue(self, dataset):
        tensor_dict = dataset.make_one_shot_iterator().get_next()

        self.tf_summary_image(tensor_dict['image'], tensor_dict['groundtruth_boxes'], 'raw_image')

        tensor_dict = self.preprocess(tensor_dict)

        self.tf_summary_image(tensor_dict['image'], tensor_dict['groundtruth_boxes'], 'preprocess_image')

        #tensor_dict['image'] = tf.to_float(tf.expand_dims(tensor_dict['image'], 0))

        input_queue = batcher.BatchQueue(
            tensor_dict,
            batch_size=self.flags.batch_size,
            batch_queue_capacity=self.flags.batch_queue_capacity,
            num_batch_queue_threads=self.flags.num_batch_queue_threads,
            prefetch_queue_capacity=self.flags.prefetch_queue_capacity)
        return input_queue

    def preprocess(self, tensor_dict):
        image = tensor_dict['image']
        boxes = tensor_dict['groundtruth_boxes']
        labels = tensor_dict['groundtruth_classes']

        image, boxes = pp.image.random_horizontal_flip(image, boxes)

        image, boxes, labels = self.ssd_random_crop(image, boxes, labels)

        image = tf.image.resize_images(
            image, tf.stack([SSD_VGG_SIZE, SSD_VGG_SIZE]),
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)

        tensor_dict['image'] = image
        tensor_dict['groundtruth_boxes'] = boxes
        tensor_dict['groundtruth_classes'] = labels
        return tensor_dict

    def random_crop_image(self, image, boxes, labels,
                      min_object_covered=1.0,
                      aspect_ratio_range=(0.75, 1.33),
                      area_range=(0.1, 1.0),
                      overlap_thresh=0.3,
                      random_coef=0.0,
                      seed=None):
        def strict_random_crop_image_fn():
            return pp.image.strict_random_crop_image(
                    image,
                    boxes,
                    labels,
                    min_object_covered=min_object_covered,
                    aspect_ratio_range=aspect_ratio_range,
                    area_range=area_range,
                    overlap_thresh=overlap_thresh)

        # avoids tf.cond to make faster RCNN training on borg. See b/140057645.
        if random_coef < sys.float_info.min:
            result = strict_random_crop_image_fn()
        else:
            generator_func = functools.partial(tf.random_uniform, [], seed=seed)
            do_a_crop_random = generator_func()
            do_a_crop_random = tf.greater(do_a_crop_random, random_coef)

            outputs = [image, boxes, labels]

            result = tf.cond(do_a_crop_random, strict_random_crop_image_fn,
                             lambda: tuple(outputs))
        return result


    def ssd_random_crop(self,
                        image,
                        boxes,
                        labels,
                        min_object_covered=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
                        aspect_ratio_range=((0.5, 2.0),) * 7,
                        area_range=((0.1, 1.0),) * 7,
                        overlap_thresh=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
                        random_coef=(0.15,) * 7,
                        seed=None):
        def random_crop_selector(selected_result, index):
            image, boxes, labels = selected_result
            return self.random_crop_image(
                    image=image,
                    boxes=boxes,
                    labels=labels,
                    min_object_covered=min_object_covered[index],
                    aspect_ratio_range=aspect_ratio_range[index],
                    area_range=area_range[index],
                    overlap_thresh=overlap_thresh[index],
                    random_coef=random_coef[index],
                    seed=seed)

        result = self.apply_with_random_selector_tuples(
                tuple(t for t in (image, boxes, labels) if t is not None),
                random_crop_selector,
                num_cases=len(min_object_covered))
        return result

    def input_fn_new(self, mode, num_epochs=1):
        num_images = NUM_IMAGES['train']

        reader = tf.TFRecordReader
        decoder = slim.tfexample_decoder.TFExampleDecoder(
            self.keys_to_features, self.items_to_handlers)

        data_dir = os.path.expanduser(self.flags.data_dir)
        filenames = [
            os.path.join(data_dir, 'trainval_merge.record'),
        ]
        dataset = slim.dataset.Dataset(
                data_sources=filenames,
                reader=reader,
                decoder=decoder,
                num_samples=num_images,
                items_to_descriptions=None,
                num_classes=NUM_CLASSES,
                labels_to_names=None)
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=self.flags.num_readers,
            common_queue_capacity=20 * self.flags.batch_size,
            common_queue_min=10 * self.flags.batch_size,
            shuffle=True)
        [image, boxes, classes] = provider.get(['image',
                                                'groundtruth_boxes',
                                                'groundtruth_classes'])

    def read_and_parse_dataset(self, filenames, num_epochs):
        keys_to_features = {
            'image/encoded':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/filename':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/key/sha256':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/source_id':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/height':
                tf.FixedLenFeature((), tf.int64, 1),
            'image/width':
                tf.FixedLenFeature((), tf.int64, 1),
            # Object boxes and classes.
            'image/object/bbox/xmin':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax':
                tf.VarLenFeature(tf.float32),
            'image/object/class/label':
                tf.VarLenFeature(tf.int64),
            'image/object/class/text':
                tf.VarLenFeature(tf.string),
            'image/object/area':
                tf.VarLenFeature(tf.float32),
            'image/object/is_crowd':
                tf.VarLenFeature(tf.int64),
            'image/object/difficult':
                tf.VarLenFeature(tf.int64),
            'image/object/group_of':
                tf.VarLenFeature(tf.int64),
            'image/object/weight':
                tf.VarLenFeature(tf.float32),
        }
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
            'groundtruth_boxes': slim.tfexample_decoder.BoundingBox(
                    ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
            'groundtruth_classes': slim.tfexample_decoder.Tensor('image/object/class/label'),
        }
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                        items_to_handlers)
        dataset = self.read_dataset(
            functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),
            functools.partial(self.parse_record, decoder),
            filenames, num_epochs)
        return dataset

    def get_raw_input(self, data_dir):
        data_dir = os.path.expanduser(data_dir)
        filenames = [
            os.path.join(data_dir, 'trainval_merge.record'),
        ]
        dataset = tf.data.TFRecordDataset(filenames,
                buffer_size=8 * 1000 * 1000)
        return dataset

    def parse_record(self, decoder, record):
        serialized_example = tf.reshape(record, shape=[])
        keys = decoder.list_items()
        tensors = decoder.decode(serialized_example, items=keys)
        tensor_dict = dict(zip(keys, tensors))
        tensor_dict['image'].set_shape([None, None, 3])

        return tensor_dict


class PascalVocModel(ssd.SSDVGGModel):

    def get_inputs(self, inputs):
        label_id_offset = 1
        def extract_images_and_targets(read_data):
            """Extract images and targets from the input dict."""
            image = read_data['image']
            location_gt = read_data['groundtruth_boxes']
            classes_gt = tf.cast(read_data['groundtruth_classes'], tf.int32)
            classes_gt -= label_id_offset
            classes_gt = ops.padded_one_hot_encoding(
                    indices=classes_gt, depth=self.num_classes, left_pad=0)
            return (image, location_gt, classes_gt)

        images, location_gt, classes_gt = zip(*map(extract_images_and_targets, inputs))

        '''
        preprocessed_images = []
        true_image_shapes = []
        for image in images:
            resized_image = self.preprocess(image)
            true_image_shape = tf.expand_dims(tf.stack(resized_image.shape.as_list()), 0)
            preprocessed_images.append(resized_image)
            true_image_shapes.append(true_image_shape)

        images = tf.concat(preprocessed_images, 0)
        true_image_shapes = tf.concat(true_image_shapes, 0)

        '''
        images = tf.stack(images)
        tf.logging.info(images)
        tf.logging.info(location_gt)
        tf.logging.info(classes_gt)
        return images, location_gt, classes_gt


class PascalVocEstimator(detection.Estimator):
    def __init__(self, flags, train_num_images, num_classes):
        super(PascalVocEstimator, self).__init__(flags, weight_decay=1e-4)

        self.train_num_images = train_num_images
        self.num_classes = num_classes
        self.batch_size = flags.batch_size

    def new_model(self, features, labels, mode, params):
        anchor = detection.SSDAnchorGenerator()
        box_coder = detection.SSDBoxCoder()
        model = PascalVocModel(
                    num_classes=self.num_classes,
                    anchor=anchor,
                    box_coder=box_coder)
        return model

    def optimizer_fn(self, learning_rate):
        optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=0.9)
        return optimizer

    def learning_rate_fn(self, global_step):
        batch_denom = 256
        boundary_epochs = [30, 60, 80, 90]
        decay_rates = [1, 0.1, 0.01, 0.001, 1e-4]

        initial_learning_rate = 0.1 * self.batch_size / batch_denom
        batches_per_epoch = self.train_num_images / self.batch_size

        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
        vals = [initial_learning_rate * decay for decay in decay_rates]

        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    def model_fn(self, features, labels, mode, params):
        return super(PascalVocEstimator, self).model_fn(features, labels, mode, params)


class PascalVocRunner(detection.Runner):
    def __init__(self, flags, estimator, dataset):
        super(PascalVocRunner, self).__init__(flags, estimator, dataset)

    def run(self):
        #self.debug_run()
        #return

        self.setup()
        def input_fn_train():
            return self.input_function(tf.estimator.ModeKeys.TRAIN, 1)

        self.classifier.train(input_fn=input_fn_train,
                         max_steps=1)

    def debug_run(self):
        mode = tf.estimator.ModeKeys.TRAIN
        batched_tensors, _ = self.dataset.input_fn(mode)

        writer = tf.summary.FileWriter('./debug')
        name = 'debug_image'
        for tensor_dict in batched_tensors:
            image = tensor_dict['image']
            image = tf.expand_dims(image, 0)
            boxes = tensor_dict['groundtruth_boxes']
            bboxes = tf.expand_dims(boxes, 0)
            image_with_box = tf.image.draw_bounding_boxes(image, bboxes)
            tf.summary.image(name, image_with_box)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge(summaries, name='summary_op')
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            #print(sess.run(data))
            summary = sess.run(summary_op)
            writer.add_summary(summary, 0)
            coord.request_stop()
            coord.join(threads)
        writer.close()


def main(argv):
    parser = detection.ArgParser()
    parser.set_defaults(data_dir='~/data/vision/pascal-voc/tfrecords/',
                        model_dir='./model_test',
                        train_epochs=10,
                        epochs_between_evals=10,
                        data_format='channels_last',

                        #train
                        batch_size=32,
                        batch_queue_capacity=150,
                        num_batch_queue_threads=8,
                        prefetch_queue_capacity=5,

                        #reader
                        filenames_shuffle_buffer_size=100,
                        shuffle_buffer_size=2048,
                        prefetch_size=512,
                        num_readers=32,
                        read_block_length=32,
                        )

    flags = parser.parse_args(args=argv[1:])
    tf.logging.info('flags: %s', flags)

    estimator = PascalVocEstimator(flags, NUM_IMAGES['train'], NUM_CLASSES)
    dataset = PascalVocDataSet(flags)

    runner = PascalVocRunner(flags, estimator, dataset)
    runner.run()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)
