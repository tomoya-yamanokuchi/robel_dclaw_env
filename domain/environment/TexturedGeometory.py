from mimetypes import init
from cv2 import DFT_INVERSE
import numpy as np
from typing import List, Any

'''
・テクスチャが割り当てられているジオメトリのリストです
'''

class TexturedGeometory:
    def __call__(self) -> Any:
        return tuple([
            'floor',
            'FFbase_xh28',
            'FF10_metal_clamping',
            'FF10_metal_clamping_small',
            'FF10_xh28',
            'FFL11_metal_clamping_small',
            'FFL11_xh28',
            'FFL11_metal_clamping',
            'FFL12_metal_clamping',
            'FFL12_plastic_finger',
            'MFbase_xh28',
            'MF20_metal_clamping',
            'MF20_metal_clamping_small',
            'MF20_xh28',
            'MFL21_metal_clamping_small',
            'MFL21_xh28',
            'MFL21_metal_clamping',
            'MFL22_metal_clamping',
            'MFL22_plastic_finger',
            'THbase_xh28',
            'TH30_metal_clamping',
            'TH30_metal_clamping_small',
            'TH30_xh28',
            'THL31_metal_clamping_small',
            'THL31_xh28',
            'THL31_metal_clamping',
            'THL32_metal_clamping',
            'THL32_plastic_finger',
            'valve_3',
        ])


if __name__ == '__main__':

    print(TexturedGeometory()())