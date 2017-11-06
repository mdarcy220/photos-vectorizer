The backend server lives in `ImageSearchServer.py` and can be run with:
```
python3 ImageSearchServer.py
```
If you go to the server in a browser (i.e., send a GET request), you will just
find a simple test page that lets you manually enter API parameters in a form
and see the response. The search API is simple: send a POST to `/imagesearch`
with the (urlencoded) body containing `img_id` and optionally `max_results`,
both of which should be integers. You will get a JSON response that looks
something like this:

```
{
    "images": [
        {
            "img_id": INTEGER,
            "diff": FLOATING_POINT,
        },
        ...
    ],
    "errstr": STRING
}
```

The `errstr` is used to give a human-readable error message if something goes
wrong (mainly for debugging if the status code returned is not 200). The `diff`
field indicates how different the image is from the image that was passed in
the request, and the `images` array is sorted by this field (so most similar
images are first). The `img_id` field will be an integer corresponding to an
image ID in the database. If `max_results` was passed in the request, the
server will not return more than that number of results in the `images` array.

More notes:
 
- The server runs on port 8000 by default. Pass the `--port
  <port_number>` option at startup to change it.
- The server will vectorize all the images it finds in Mysql when it
  starts up and search through those (but images added to Mysql after
  the server is already started will not be searched; that feature is
  forthcoming).
- The vectorizer used will be an autoencoder by default, which expects
  that a saved model is available (i.e., assumes you have already run
  `train_model.py`). Since this may make testing the server slow or
  troublesome if you don't have a good GPU for training/eval, you can
  pass `--vectorizer-type "flat"` to the server startup and it will use
  an alternative "dumb" vectorizer implementation that does not require
  a neural network.
- The images will be sourced from Mysql by default, which is probably
  what you want. If you want to directly load images from the
  filesystem instead, you can pass `--image-source-type "filesystem"`
  to the server at startup, but be warned that the `img_id` field will
  be hard to interpret (IDs will be assigned to files on the fly during
  data loading).

