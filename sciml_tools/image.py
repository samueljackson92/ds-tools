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


def crop_center(img,percent):
    """Central crop an image by percent

    Args:
        img: image to centrally crop by
    Returns:
        a numpy array centrally cropped numpy array
    """
    y,x = img.shape
    cropx, cropy = int(x*percent), int(y*percent)
    startx = (x - cropx)//2
    starty = (y - cropy)//2
    return img[starty:starty+cropy,startx:startx+cropx]
