#!/usr/bin/python3

import version_check
version_check.assert_min_version('3.5')

import http.server
import os
import urllib.parse
import io
import imageio
import scipy.ndimage
import skimage.transform
import numpy as np
from ImageSearchEngine import ImageSearchEngine
import MySQLdb
import json
import sys
import ImageVectorize
import ImageDataLoader

conn = MySQLdb.connect('127.0.0.1', 'root', 'DM44DoJ8alquuShI', 'Photos')

class ImageSearchRequestHandler(http.server.BaseHTTPRequestHandler):
	def do_GET(self):
		"""Serve a GET request."""
		responsetext = b"""<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN"><html>
			<title>Form test</title>
			<body>
			<form method="post" action="/imagesearch">
				<label for="img_id">Image ID:</label><input name="img_id" type="number"/><br />
				<label for="max_results">Max Results:</label><input name="max_results" type="number"/>
				<input type="submit" value="send"/></form>
			</body>
			</html>
		"""
		self.send_response(200)
		self.send_header("Content-type", "text/html")
		self.send_header("Content-Length", str(len(responsetext)))
		self.end_headers()
		f = self.wfile
		f.write(responsetext)
		f.flush()

	def do_POST(self):
		req_parts=urllib.parse.urlparse(self.path)
		if req_parts.path == '/imagesearch':
			self._do_imagesearch_POST()
		else:
			self._do_default_POST()

	def _do_default_POST(self):
		err_msg = "There's nothing here. Double-check that you are accessing the right endpoint"
		self.send_response(404)
		self.send_header("Content-type", "text/plain")
		self.send_header("Content-Length", str(len(err_msg)))
		self.end_headers()
		f = self.wfile
		f.write(err_msg.encode())
		f.flush()


	def _do_imagesearch_POST(self):
		content_length = int(self.headers['Content-length'])
		req_body = urllib.parse.parse_qs(self.rfile.read(content_length).decode())

		img_id = int(req_body['img_id'][0])

		max_results = 10
		if 'max_results' in req_body:
			max_results = int(req_body['max_results'][0])

		response = self.do_image_search(img_id, max_results=max_results)
		response_body = json.dumps(response['body']).encode()

		self.send_response(response['status'])
		self.send_header("Content-type", "text/json")
		self.send_header("Content-Length", str(len(response_body)))
		self.end_headers()
		f = self.wfile
		f.write(response_body)
		f.flush()

	def do_image_search(self, img_id, max_results=10):
		response = {'status': 200, 'body': {'images': list(), 'errstr': ''}}

		cur = conn.cursor()
		num_results = cur.execute("SELECT folder1, folder2, sys_file FROM Photo WHERE id = %s", [img_id])
		if num_results == 0:
			response['status'] = 404
			response['body']['errstr'] = "No image was found with id {}".format(img_id)
			return response

		folder1, folder2, sys_file = cur.fetchone()
		filename = os.path.join("/var/www/uploads", folder1, folder2, sys_file)
		if not os.path.isfile(filename):
			response['status'] = 500
			response['body']['errstr'] = "Image record was in database but data not in filesystem"
			return response

		image_results = []
		loader = self.server.search_engine.image_loader
		try:
			raw_image_data = scipy.ndimage.imread(filename)
			image_data = loader.reshape_img(loader.fix_img_size(raw_image_data))
			image_results = self.server.search_engine.lookup_img(image_data, k_max=max_results)
		except IOError:
			response['status'] = 500
			response['body']['errstr'] = "Failed to parse image data"
			return response
		except ValueError:
			response['status'] = 500
			response['body']['errstr'] = "Failed to process image data"
			return response

		for result in image_results:
			response['body']['images'].append({'img_id': int(result[1]), 'diff': float(result[0])})

		return response


class ImageSearchServer(http.server.HTTPServer):
	def __init__(self, config_options, search_engine, *args, **kwargs):
		self.config_options = config_options
		self.search_engine = search_engine
		super(ImageSearchServer, self).__init__(*args, **kwargs)


if __name__ == '__main__':
	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument('--port', dest='port', type=int, action='store', default=8000, help='Port to run the server on')
	parser.add_argument('--vectorizer-type', dest='vectorizer_type', type=str, action='store', default='autoencoder', help='Type of image vectorizer to use')
	parser.add_argument('--image-source-type', dest='image_source_type', type=str, action='store', default='mysql', help='Type of image source to use')
	cmdargs = parser.parse_args(sys.argv[1:])

	vectorizer = ImageVectorize.AutoencoderVectorizer() if cmdargs.vectorizer_type == 'autoencoder' else ImageVectorize.FlatVectorizer()
	image_loader = ImageDataLoader.MysqlImageDataLoader() if cmdargs.image_source_type == 'mysql' else ImageDataLoader.FilesystemImageDataLoader()

	server = ImageSearchServer(None, ImageSearchEngine(vectorizer=vectorizer, image_loader=image_loader, max_images=10), ('', cmdargs.port), ImageSearchRequestHandler)
	server.serve_forever()

	conn.close()
