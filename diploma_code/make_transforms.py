import albumentations as A

import numpy as np

from diploma_code.patched_hwb import HandWrittenBlot
from ml_collections import ConfigDict


class AlbuHandWrittenBlot(A.DualTransform):
    def __init__(self, hw_blot, always_apply=False, p=0.5):
        super(AlbuHandWrittenBlot, self).__init__(always_apply, p)
        self.hw_blot = hw_blot
        
    def get_transform_init_args_names(self):
        return ("hw_blot", "always_apply", "p")

    def apply(self, image, **params):
        return self.hw_blot(image)


def make_blot_transform(blot_config: ConfigDict):
    if not blot_config.enabled:
        return None
    
    return AlbuHandWrittenBlot(HandWrittenBlot(**blot_config.params), p=blot_config.p)
     

def make_basic_albums(albums_config: ConfigDict):
    
    transforms_list = []
    
    for al_name, params_dict in albums_config.items():
        
        if not params_dict.enabled:
            continue
            
        transforms_list.append(eval(f"A.{al_name}")(
            **params_dict.params
        ))
        
    return transforms_list
            
    
def make_transforms(config: ConfigDict):
    
    transforms_list = []
    
    maybe_blot_transform = make_blot_transform(config.blot)
    if maybe_blot_transform is not None:
        transforms_list.append(maybe_blot_transform)

    transforms_list.extend(make_basic_albums(config.basic_albums))
    if not transforms_list:
        return None
    return A.Compose(transforms_list, p=1.0)
    