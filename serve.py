import tornado.ioloop
import tornado.web

import json

class MainHandler(tornado.web.RequestHandler):
    def initialize(self, ann):
        self.ann = ann

    def get(self):
        text = self.get_argument("text")
        self.write(json.dumps({
            'output': self.ann.predict(text),
        }, ensure_ascii=False))
        self.set_header('Content-Type', 'application/json')
        self.set_header('Access-Control-Allow-Origin', '*')


def serve(model):
    app = tornado.web.Application([
        (r"/", MainHandler, dict(ann=model)),
    ])
    
    app.listen(8881)
    tornado.ioloop.IOLoop.current().start()