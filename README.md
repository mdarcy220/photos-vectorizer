# Photos Vectorizer
Image Vectorizer for use with johnliggett/Photos for image retrieval.

## How to run it

### Dependencies

A complete list of dependencies is not available, but at the very least you will need CNTK 2 and Python 3 installed, along with the Python3 modules `numpy`, `scipy`, `matplotlib`, and `mysqlclient`.

### Reverse Image Search Setup

You will need training data in the `largedata/train/` directory (you will need to create the `train` folder). Both `.jpg` and `.png` images will be detected. Any image with at least 3 color channels should work, but be aware that they will be rescaled to `250 x 250` resolution when being passed through the network. If the original image has a non-square aspect ratio, this may lead to distortion and lower quality results.

Once the training data is set up, just run:
```
python3 train_model.py
```

This will train the model and save it in `largedata/autoencoder_checkpoint` (this may take a while). Then you can test a sample query from the model with:
```
python3 ImageSearchEngine.py
```

### Auto-tagger Setup

Run the `download_resnet.py` script to get the pre-trained CNTK resnet (or manually download any of the "ImageNet 1K" ResNets from [CNTK's repo](https://github.com/Microsoft/CNTK/blob/db2e817cdcdf35c14344b96b8c6a2cf3cbe5866b/PretrainedModels/Image.md#resnet) and save as `saved_model.dnn`)

Running the `auto_tagger.py` script will tag all untagged images in the database. The auto-tagger is also integrated with the REST server (see "Server Setup" below) to tag new images as well.


### Server Setup

Documentation for the server can be found in [`docs/server.md`](docs/server.md).


## Notes

An NVIDIA GPU is highly recommended. Training on a CPU may take days or even weeks, whereas a GTX 1070/1080 will complete the (admittedly short for testing purposes) training loop in just a few minutes.
