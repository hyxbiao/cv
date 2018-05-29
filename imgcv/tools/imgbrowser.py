#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
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
            "--dir",
            help="The location of the image files.",
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
    def get(self):
        self.write("Hello, world")


class DirHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def get(self):
        path = str(self.get_argument('path', '/', True))
        #path = os.path.join(self.application.static_path, path)
        path = self.application.static_path + path
        #data = sorted(os.listdir(path))
        data = sorted(next(os.walk(path))[1])
        logging.info('static_path: {}, sub dirs: {}'.format(self.application.static_path, data))
        self.write({'data': data})


class FileHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def get(self):
        path = str(self.get_argument('path', '/', True))
        page = int(self.get_argument('page', 0, True))
        size = int(self.get_argument('size', 9, True))
        path = self.application.static_path + path
        dirx, _, files = next(os.walk(path))
        files = sorted(files)
        rel_dir = dirx[len(self.application.static_path) + 1:]
        start = page * size
        end = start + size
        data = []
        for f in files[start:end]:
            data.append(os.path.join('/static', rel_dir, f))
        logging.info('file list: {}'.format(data))
        self.write({'data': data})


class WebApplication(tornado.web.Application):

    def __init__(self, static_path, debug=True):
        self.static_path = static_path

        settings = dict(
            cookie_secret = "__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
            xsrf_cookies = True,
            static_path = static_path,
            #template_path = this_file_path + "/../front/templates",
        )
        handlers = [
            (r"/",              MainHandler),
            (r"/dir",           DirHandler),
            (r"/file",          FileHandler),
        ]
        super(WebApplication, self).__init__(handlers, debug=debug, **settings)


class Application(object):
    def __init__(self, flags):
        self.flags = flags
        self.dir = os.path.expanduser(self.flags.dir)

    def run(self):
        self.start_server()
        return
        dirs = sorted(next(os.walk(self.dir))[1])
        #dirs = sorted(os.listdir(self.dir))
        for dirx in dirs:
            #logging.info('sub dir: {}'.format(dirx))
            print('sub dir: {}'.format(dirx))

    def start_server(self):

        #tornado.options.parse_command_line()
        tornado.log.enable_pretty_logging()

        static_path = self.dir
        app = WebApplication(static_path, False)
        server = tornado.httpserver.HTTPServer(app)
        server.bind(self.flags.port)
        server.start(0)  # autodetect number of cores and fork a process for each

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
