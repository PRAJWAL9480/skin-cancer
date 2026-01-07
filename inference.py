# inference.py - lightweight inference; prefers final_stub if present
import os
from PIL import Image
import numpy as np

# prefer importing from final_stub if available (guarantees identical behavior)
try:
    import final_stub as final
    print('inference: using final_stub')
    preprocess_vgg_img = final.preprocess_vgg
    preprocess_dn_img = final.preprocess_dn
    is_skin_lesion_model = final.is_skin_lesion_model
    predict_single_image = final.predict_single_image
    vgg_model = getattr(final, 'vgg_model', None)
    densenet_model = getattr(final, 'densenet_model', None)
except Exception as e:
    print('inference: final_stub not available, falling back to minimal local stubs:', e)
    vgg_model = None
    densenet_model = None
    def preprocess_vgg_img(img):
        arr = np.asarray(img.resize((224,224))).astype('float32')
        return np.expand_dims(arr,0)
    def preprocess_dn_img(img):
        arr = np.asarray(img.resize((224,224))).astype('float32')
        return np.expand_dims(arr,0)
    def is_skin_lesion_model(path, detector=None, img_size=(224,224), threshold=0.5):
        return False, 0.0, 'no_detector'
    def predict_single_image(path, *args, **kwargs):
        return {'accepted': False, 'reject_reason': 'no_models_loaded'}
