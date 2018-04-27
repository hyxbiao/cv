#!/usr/bin/env python
# encoding: utf-8

import tornado.ioloop
import tornado.web
import tornado.options
import logging
import json

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


class DataSetHandler(tornado.web.RequestHandler):
    def get(self, path):
        logging.info('path: {}'.format(path))
        if path == 'transfer':
            return self.transfer()
        elif path == 'data':
            return self.data()
        else:
            return self.index()

    def index(self):
        mode = str(self.get_argument('dataset', 'train', True))
        page = int(self.get_argument('page', 0, True))
        size = int(self.get_argument('size', 6, True))
        logging.info('page: {}, size: {}'.format(page, size))
        data = self.application.proxy.on_dataset_index(mode, page, size)
        logging.info('data: {}'.format(data))
        self.write({'data': data})

    def transfer(self):
        mode = str(self.get_argument('dataset', 'train', True))
        index = int(self.get_argument('index', 0, True))
        data = self.application.proxy.on_dataset_transfer(mode, index)
        self.write({'data': data})

    def data(self):
        mode = str(self.get_argument('dataset', 'train', True))
        index = int(self.get_argument('index', 0, True))
        method = str(self.get_argument('method', 'random', True))
        logging.info('method: {}'.format(method))
        data = self.application.proxy.on_dataset_data(mode, index, method)
        self.write(data)


class ImageHandler(tornado.web.RequestHandler):
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
    app = Application(proxy, static_path)
    app.listen(port)

    tornado.ioloop.IOLoop.current().start()
