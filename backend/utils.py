import numpy as np
from PIL import Image

def preprocess(file, target_size=(224, 224)):
    """
    Convert uploaded image into tensor for model prediction.
    Works with Keras 3.x + TensorFlow 2.20.
    """
    img = Image.open(file.stream).convert("RGB")
    img = img.resize(target_size)
    x = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(x, axis=0)
