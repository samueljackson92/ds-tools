from PIL import Image
import numpy as np

def load_tiff(path):
    """Load a single tiff file from file

    Args:
        path: the file path of the .tiff file.
    Returns:
        a numpy array containing the tiff data.
    """
    img = Image.open(path)
    return np.array(img)

def normalize(img):
    """Min/Max normalization to rescale to unity

    Args:
        img: image to rescale to the 0-1 range
    Returns:
        a numpy array with rescaled intensity
    """
    return (img - img.min()) / (img.max() - img.min())
