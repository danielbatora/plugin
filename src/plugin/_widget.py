"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.draw import polygon2mask
import napari
from imagetools.util import calculate_lag
import numpy as np 
import scipy
import os 

if TYPE_CHECKING:
    import napari





folder = "python_course/brain/folder_1"


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


@magic_factory(call_button ="Calculate")
def adjust_contrast(image: "napari.layers.Image"):
    if type(image) is napari.layers.image.image.Image:
        image.contrast_limits = [0, image.data.max()]

@magic_factory(call_button ="Calculate")
def register_images(reference: "napari.layers.Image", target_image:"napari.layers.Image", roi: "napari.layers.Shapes", viewer:"napari.viewer.Viewer"): 

    print(reference, target_image)


    if target_image is not None: 
        target_image.contrast_limits = [0, target_image.data.max()]

    if len(viewer.layers) >2 and roi.data:

        lags = []
        for i in range(len(roi.data)):
            crop = roi.data[i].astype(int)

            crop_x = np.unique(np.array([i[0] for i in crop]))
            crop_y = np.unique(np.array([i[1] for i in crop]))

            ref = reference.data[crop_x.min(): crop_x.max(), crop_y.min(): crop_y.max()]
            target = target_image.data[crop_x.min(): crop_x.max(), crop_y.min(): crop_y.max()]
            lag = calculate_lag(ref, target)
            lags.append(lag)
            
            #np.save(os.path.join(folder, "ref"), ref)
            #np.save(os.path.join(folder, "target"), target)
        print(lags)
        mean_lag = np.mean(np.array(lags), axis = 0).astype(int)

        corrected_target = scipy.ndimage.shift(target_image.data, -mean_lag)

        viewer.add_image(corrected_target, name="shifted_target")

    else: 
        print(type(reference), type(target_image), type(roi))
        print("Images or shapes are loaded incorrectly")