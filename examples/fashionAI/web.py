#!/usr/bin/env python
# encoding: utf-8

import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.options
from tornado import gen
import logging
import json

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
    def transfer(self):
        mode = str(self.get_argument('dataset', 'train', True))
        index = int(self.get_argument('index', 0, True))
        data = self.application.proxy.on_dataset_transfer(mode, index)
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
