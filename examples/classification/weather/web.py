#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.options
from tornado import gen
import logging
import json

import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

import resnet_app as app

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


class DataSetHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def get(self, path):
        logging.info('path: {}'.format(path))
        if path == 'transfer':
            yield self.transfer()
        elif path == 'data':
            yield self.data()
        else:
            yield self.index()

    @gen.coroutine
    def index(self):
        mode = str(self.get_argument('dataset', 'train', True))
        page = int(self.get_argument('page', 0, True))
        size = int(self.get_argument('size', 6, True))
        predict = int(self.get_argument('predict', 0, True)) == 1
        logging.info('predict: {}, page: {}, size: {}'.format(predict, page, size))
        data = yield self.application.proxy.on_dataset_index(mode, predict, page, size)
        logging.info('data: {}'.format(data))
        self.write({'data': data})

    @gen.coroutine
    def data(self):
        mode = str(self.get_argument('dataset', 'train', True))
        index = int(self.get_argument('index', 0, True))
        method = str(self.get_argument('method', 'random', True))
        logging.info('method: {}'.format(method))
        data = self.application.proxy.on_dataset_data(mode, index, method)
        self.write(data)


class ImageHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def get(self, path):
        self.set_header('Content-Type', '')
        with open(path , 'rb') as file:
            self.write(file.read())


class Application(tornado.web.Application):

    def __init__(self, proxy, static_path, debug=True):
        self.proxy = proxy

        settings = dict(
            cookie_secret = "__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
            xsrf_cookies = True,
            static_path = static_path,
            #template_path = this_file_path + "/../front/templates",
        )
        handlers = [
            (r"/",              MainHandler),
            (r"/dataset/(.*)",       DataSetHandler),
            (r'/img(/.*)',      ImageHandler),
        ]
        super(Application, self).__init__(handlers, debug=debug, **settings)


def start_server(proxy, static_path, port=8100):

    #tornado.options.parse_command_line()
    tornado.log.enable_pretty_logging()

    app = Application(proxy, static_path, False)
    server = tornado.httpserver.HTTPServer(app)
    server.bind(port)
    server.start(0)  # autodetect number of cores and fork a process for each

    tornado.ioloop.IOLoop.current().start()


class WebRunner(app.WeatherRunner):
    def get_mode(self, mode):
        if mode == 'train':
            mode = tf.estimator.ModeKeys.TRAIN
        elif mode == 'test':
            mode = tf.estimator.ModeKeys.EVAL
        else:
            mode = tf.estimator.ModeKeys.PREDICT
        return mode

    @gen.coroutine
    def on_dataset_index(self, mode, predict, page, size):
        mode = self.get_mode(mode)
        df = self.dataset.get_raw_input(mode)

        from_index = page * size
        to_index = (page + 1) * size
        df = df[from_index:to_index]
        if predict:
            def input_fn(mode, df=df, parser=self.dataset, num_parallel_calls=self.flags.num_parallel_calls):
                dataset = tf.data.Dataset.from_tensor_slices(dict(df))
                dataset = dataset.map(lambda value: parser.parse_record(mode, value),
                                    num_parallel_calls=num_parallel_calls)
                return dataset
            self.input_function = input_fn
            self.setup()
            output = self.predict()

        items = []
        i = 0
        for index, row in df.iterrows():
            cls, title = row['image'].rsplit('/', 2)[1:]
            item = {
                'id': index,
                'title': title,
                'image': '/img' + row['image'],
                'attr': {
                    'class': cls,
                    'label': row['label'],
                }
            }
            if predict:
                v = next(output)
                prob = np.mean(v['probabilities'], axis=0)
                pred = np.argmax(prob)
                probs = np.char.mod('%.4f', prob)
                #prob_str = ';'.join(np.char.mod('%.4f', prob))
                item['predict'] = {
                    'pred': pred,
                    'probs': probs.tolist(),
                }
            items.append(item)
            i += 1
        #return items
        raise gen.Return(items)


def main(argv):
    parser = app.WeatherArgParser()
    parser.set_defaults(data_dir='~/data/vision/weather/Image',
                        model_dir='./models/experiment',
                        train_epochs=10,
                        predict_yield_single=False,
                        pretrain_model_dir='~/data/models/resnet50',
                        pretrain_warm_vars='^((?!dense).)*$')

    flags = parser.parse_args(args=argv[1:])

    dataset = app.WeatherDataSet(flags)

    estimator = app.WeatherEstimator(flags,
            train_num_images=dataset.train_num_images,
            num_classes=dataset.num_classes)

    runner = WebRunner(flags, estimator, dataset)

    start_server(proxy=runner, static_path=flags.data_dir)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)
