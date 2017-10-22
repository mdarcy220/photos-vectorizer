# Photos Vectorizer
Image Vectorizer for use with johnliggett/Photos for image retrieval.

## How to run it

### Dependencies

A complete list of dependencies is not available, but at the very least you will need CNTK 2 and Python 3 installed, along with the Python3 modules Numpy, Scipy, and Matplotlib.

### Running

You will need training data in the `largedata/train/` directory (you will need to create the `train` folder). Both .jpg and .png images will be detected, but they must have a very specific shape:

- They must have three color channels (i.e., RGB but NOT RGBA)
- The resolution must be _exactly_ `250x250` pixels

These should guarantee that the shape of the image tensor in memory will be `(250, 250, 3)` when loaded by `scipy.ndimage.imread`. More sophisticated preprocessing will likely be added in the future to rlax these requirements.

Once the training data is set up, just run:
```
python3 train_model.py
```

This will train the model and save it in `largedata/autoencoder_checkpoint`. Then you can test a sample query from the model with:
```
python3 do_image_lookup.py
```

