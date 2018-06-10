#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import argparse
import re
import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.options
from tornado import gen
import logging
import json

import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__(add_help=True)
        self.add_argument(
            "--dir", default="~/",
            help="[default: %(default)s] The location of the static files.",
        )
        self.add_argument(
            "--filter", default=".*",
            help="[default: %(default)s] Regex filter vars",
        )
        self.add_argument(
            "--port", type=int, default=8100,
            help="[default: %(default)s] Port.",
        )


class MainHandler(tornado.web.RequestHandler):
    def get(self, path):
        router = {
            'root': self.root
        }
        if path in router:
            router[path]()
        else:
            self.index()

    def index(self):
        self.write("Hello, world")

    def root(self):
        path = str(self.get_argument('path', '', True))
        if path:
            path = os.path.expanduser(path)
            self.application.static_path = path
        self.write({'data': self.application.static_path})

class DirHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def get(self):
        path = str(self.get_argument('path', '/', True))
        #path = os.path.join(self.application.static_path, path)
        path = self.application.static_path + path
        logging.info('dir path: {}'.format(path))
        #data = sorted(os.listdir(path))
        data = sorted(next(os.walk(path))[1])
        logging.info('static_path: {}, sub dirs: {}'.format(self.application.static_path, data))
        self.write({'data': data})


class FileHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def get(self):
        root_dir = self.application.static_path
        path = str(self.get_argument('path', '/', True))
        page = int(self.get_argument('page', 0, True))
        size = int(self.get_argument('size', 16, True))
        path = self.application.static_path + path
        dirx, _, files = next(os.walk(path))
        files = sorted(files)
        rel_dir = dirx[len(root_dir) + 1:]
        start = page * size
        end = start + size
        data = []
        for f in files[start:end]:
            data.append(os.path.join('/static', rel_dir, f))
        logging.info('file list: {}'.format(data))
        self.write({'data': data})


class SearchHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def get(self):
        pattern_str = str(self.get_argument('pattern', '', True))
        if not pattern_str:
            raise gen.Return()
        root_dir = self.application.static_path
        pattern = re.compile(pattern_str)
        data = []
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if pattern.search(f):
                    rel_dir = root[len(root_dir) + 1:]
                    data.append(os.path.join('/static', rel_dir, f))
        logging.info('search file list: {}'.format(data))
        self.write({'data': data})


class MyStaticFileHandler(tornado.web.StaticFileHandler):
    @gen.coroutine
    def get(self, path, include_body=True):
        logging.info('static root path: {}, new: {}'.format(self.root, self.application.static_path))
        self.root = self.application.static_path
        super(MyStaticFileHandler, self).get(path, include_body)


class WebApplication(tornado.web.Application):

    def __init__(self, static_path, debug=True):
        self.static_path = static_path

        settings = dict(
            cookie_secret = "__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
            xsrf_cookies = True,
            #static_path = static_path,
            #template_path = this_file_path + "/../front/templates",
        )
        handlers = [
            (r"/admin/(.*)",    MainHandler),
            (r"/dir",           DirHandler),
            (r"/file",          FileHandler),
            (r"/search",        SearchHandler),
            (r"/static/(.*)",   MyStaticFileHandler, {'path': self.static_path}),
        ]
        super(WebApplication, self).__init__(handlers, debug=debug, **settings)


class Application(object):
    def __init__(self, flags):
        self.flags = flags
        self.dir = os.path.expanduser(self.flags.dir)

    def run(self):
        self.start_server()
        return
        files = glob.glob(self.dir, '170927_070545809_Camera_6*', recursive=True)
        #dirs = sorted(os.listdir(self.dir))
        for dirx in files:
            #logging.info('sub dir: {}'.format(dirx))
            print('sub dir: {}'.format(dirx))

    def start_server(self):

        #tornado.options.parse_command_line()
        tornado.log.enable_pretty_logging()

        logging.info('default directory: {}'.format(self.dir))
        app = WebApplication(self.dir, True)
        server = tornado.httpserver.HTTPServer(app)
        server.bind(self.flags.port)
        #server.start(0)  # autodetect number of cores and fork a process for each
        server.start()

        tornado.ioloop.IOLoop.current().start()


def main(argv=None):
    if argv is None:
        argv = sys.argv
    parser = ArgParser()
    flags = parser.parse_args(args=argv[1:])

    app = Application(flags)
    app.run()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)
