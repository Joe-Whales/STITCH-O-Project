import os
import cv2
import numpy as np

class OpenCVReader:
    def __init__(self, image_dir, color_mode):
        self.image_dir = image_dir
        self.color_mode = color_mode
        assert color_mode in ["RGB", "BGR", "GRAY"], f"{color_mode} not supported"
        if color_mode != "BGR":
            self.cvt_color = getattr(cv2, f"COLOR_BGR2{color_mode}")
        else:
            self.cvt_color = None

    def __call__(self, filename, is_mask=False):
        filename = os.path.join(self.image_dir, filename)
        assert os.path.exists(filename), filename
        if is_mask:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            return img
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        if self.color_mode != "BGR":
            img = cv2.cvtColor(img, self.cvt_color)
        return img

class NPYReader:
    def __init__(self, image_dir, target_size=(224, 224)):
        self.image_dir = image_dir
        self.target_size = target_size

    def __call__(self, filename):
        filename = os.path.join(self.image_dir, filename)
        assert os.path.exists(filename), filename
        
        # Load the .npy file
        img = np.load(filename)
        
        # Ensure the image has at least 2 dimensions
        if img.ndim == 1:
            img = img.reshape(1, -1)
        
        # Resize the image to target size
        if img.shape[:2] != self.target_size:
            # If it's a 2D array (grayscale), add a channel dimension
            if img.ndim == 2:
                img = img[..., np.newaxis]
            
            # Resize each channel separately and stack them back together
            resized_channels = []
            for i in range(img.shape[2]):
                resized_channel = cv2.resize(img[..., i], self.target_size, interpolation=cv2.INTER_LINEAR)
                resized_channels.append(resized_channel)
            
            img = np.stack(resized_channels, axis=-1)
        
        return img

def build_image_reader(cfg_reader):
    if cfg_reader["type"] == "opencv":
        return OpenCVReader(**cfg_reader["kwargs"])
    elif cfg_reader["type"] == "npy":
        return NPYReader(**cfg_reader["kwargs"])
    else:
        raise TypeError("no supported image reader type: {}".format(cfg_reader["type"]))