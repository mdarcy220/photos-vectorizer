# Photos Vectorizer
Image Vectorizer for use with johnliggett/Photos for image retrieval.

## How to run it

### Dependencies

A complete list of dependencies is not available, but at the very least you will need CNTK 2 and Python 3 installed, along with the Python3 modules Numpy, Scipy, and Matplotlib.

### Running

You will need training data in the `largedata/train/` directory (you will need to create the `train` folder). Both .jpg and .png images will be detected. In previous versions, images were required to have a very specific size and number of color channels, but those requirements have since been relaxed. Any image with at least 3 color channels should work, but be aware that they will be rescaled to `128 x 128` resolution when being passed through the network. If the original image has a non-square aspect ratio, this may lead to distortion and lower quality results.

Once the training data is set up, just run:
```
python3 train_model.py
```

This will train the model and save it in `largedata/autoencoder_checkpoint` (this may take a while). Then you can test a sample query from the model with:
```
python3 ImageSearchEngine.py
```

There is now a server available as well. Documentation for that can be found in [`docs/server.md`](docs/server.md).


An NVIDIA GPU is highly recommended. Training on a CPU may take days or even weeks, whereas a GTX 1070/1080 will complete the (admittedly short for testing purposes) training loop in just a few minutes.
