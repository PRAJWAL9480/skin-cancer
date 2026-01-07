# final.py shim — re-export everything from final_stub so old imports work
try:
    from final_stub import *
    # also expose module-level objects under this module name
    from final_stub import vgg_model, densenet_model, skin_detector, CLASS_NAMES, DEFAULT_VGG_WEIGHT
except Exception as e:
    # if final_stub fails to load, raise an informative error so compare script shows it
    raise RuntimeError('final.py shim failed to import final_stub: ' + str(e))
