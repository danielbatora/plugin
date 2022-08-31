"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QWidget, QSlider
from skimage.draw import polygon2mask
import napari
from imagetools.util import calculate_lag, calculate_bounding_rectangle, calculate_props, calculate_rate, fit_dim, generate_polygon_mask, crop_with_polygon_mask, df_remove_suffixes, return_mask, return_positive_negative_cells
import numpy as np 
import scipy
import os 
import pandas as pd 
import cv2
from skimage.filters import threshold_otsu
from shapely.geometry import Polygon


if TYPE_CHECKING:
    import napari





folder = os.getcwd()



class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.slider = QSlider()
        self.slider.valueChanged.connect(self._on_click)
        if any([type(i)== napari.layers.image.image.Image for i in self.viewer.layers]):
            self.slider.setSliderPosition(threshold_otsu(self.viewer.layers[0].data))
            self.slider.setMinimum(self.viewer.layers[0].data.min())         
            self.slider.setMaximum(self.viewer.layers[0].data.max())
            if not any([i.name == "thresholded" for i in self.viewer.layers]):
                self.viewer.add_image(np.ones((10,10)), name = "thresholded")
                self.viewer.layers["thresholded"].data = self.viewer.layers[0].data > self.slider.value()


        else: 
            self.slider.setMinimum(0)         
            self.slider.setMaximum(10000)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.slider)

    def _on_click(self):
        if any([type(i)== napari.layers.image.image.Image for i in self.viewer.layers]):
            
            if not any([i.name == "thresholded" for i in self.viewer.layers]):
                self.viewer.add_image(np.ones((10,10)), name = "thresholded")
            self.viewer.layers["thresholded"].data = self.viewer.layers[0].data > self.slider.value()
        else: 
            print("No images")
            return None


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
        mean_lag = np.mean(np.array(lags), axis = 0).astype(int)

        corrected_target = scipy.ndimage.shift(target_image.data, -mean_lag, mode ="constant")

        viewer.add_image(corrected_target, name="shifted_target")

    else: 
        print(type(reference), type(target_image), type(roi))
        print("Images or shapes are loaded incorrectly")




@magic_factory(call_button ="Calculate")
def shapes2labels(shapes: "napari.layers.Shapes", viewer :"napari.viewer.Viewer"):

    types = np.array([type(x) for x in viewer.layers])
    img_ind = np.argwhere(types == napari.layers.image.image.Image)
    if img_ind.shape[0]:
        img_shape = viewer.layers[img_ind[0][0]].data.shape
        if shapes is not None and shapes.data: 
            labels = shapes.to_labels(img_shape)
            viewer.add_labels(labels, name=f"labelsfrom{shapes.name}")




@magic_factory(call_button ="Calculate")
def calculate_target_intensity(reference: "napari.layers.Labels", target:"napari.layers.Image", shapes: "napari.layers.Shapes", viewer :"napari.viewer.Viewer",save:str, property:str ="intensity_mean"):
    """
    Calculate the average pixel intensity (target) of a target protein on 
    all cells defined by the reference labels (reference)
    in a given region of interest (shapes)
    """



    savepath = os.path.join(folder, save + ".csv")
    ref_data = reference.data
    target_data = target.data
    output= {"shape": [],"reference": [], "target": [], property: [], f"{property}_std": [], "values":[], "total_cells" : []}


    if shapes is not None and shapes.data:
        counter = 0
        for polygon in shapes.data: 



            polygon_fit = fit_dim(polygon, ref_data.shape)
        #get closest rectange
            rectangle = calculate_bounding_rectangle(polygon_fit)
            crop_x = np.unique(np.array([i[0] for i in rectangle]))
            crop_y = np.unique(np.array([i[1] for i in rectangle]))
        #work with cropped image
            ref_crop = ref_data[crop_x.min(): crop_x.max(), crop_y.min(): crop_y.max()]
            target_crop = target_data[crop_x.min(): crop_x.max(), crop_y.min(): crop_y.max()]
        #remove additional labels that are not in the polygon
            poly_mask = generate_polygon_mask(ref_crop, rectangle, polygon_fit)
            ref_poly_only = crop_with_polygon_mask(ref_crop, poly_mask)
            values = calculate_props(ref_poly_only, target_crop, property)
            output["reference"].append(reference.name)
            output["target"].append(target.name)
            output[property].append(np.mean(np.array(values)))
            output[f"{property}_std"].append(np.std(np.array(values)))
            output["values"].append(values)
            output["total_cells"].append(len(values))

            output["shape"].append(f"{shapes.name}_{counter}")
            counter += 1
    
        df = pd.DataFrame(output)
        if os.path.isfile(savepath): 
            df_load = pd.read_csv(savepath, index_col = 0)
            df_save = pd.merge(df_load, df, on=["shape", "reference", "target"], how= "outer")
            df_save = df_remove_suffixes(df_save)
            df_save.to_csv(savepath)
        else: 
            df.to_csv(savepath)






@magic_factory(call_button ="Calculate")
def calculate_target_rate(reference: "napari.layers.Labels", target:"napari.layers.Image", shapes: "napari.layers.Shapes", viewer :"napari.viewer.Viewer",save:str,  coloc_thr = 0.2):

    savepath = os.path.join(folder, save + ".csv")

    ref_data = reference.data
    target_data = target.data

    if np.unique(target_data).shape[0] > 2: 
        #Binarize file if it is not a binary file 
        target_data = target.data > threshold_otsu(target_data)
    else: 
        target_data = target.data
    rates = {"shape":[],"reference": [], "target": [], "rate": [], "total_cells":[], "positive_cells":[], "total_area":[]}
    if shapes is not None and shapes.data: 
        counter = 0
        for polygon in shapes.data: 
            
        #Crop polygon coordinates not to exceed image dimensions
            polygon_fit = fit_dim(polygon, ref_data.shape)

        #get closest rectange
            rectangle = calculate_bounding_rectangle(polygon_fit)
            crop_x = np.unique(np.array([i[0] for i in rectangle]))
            crop_y = np.unique(np.array([i[1] for i in rectangle]))
        #work with cropped image
            ref_crop = ref_data[crop_x.min(): crop_x.max(), crop_y.min(): crop_y.max()]
            target_crop = target_data[crop_x.min(): crop_x.max(), crop_y.min(): crop_y.max()]

        #remove additional labels that are not in the polygon
            poly_mask = generate_polygon_mask(ref_crop, rectangle, polygon_fit)

            ref_poly_only = crop_with_polygon_mask(ref_crop, poly_mask)
            target_poly_only = crop_with_polygon_mask(target_crop, poly_mask)

        #calculate rate for each region 
            rate = calculate_rate(ref_poly_only, target_poly_only, coloc_thr)
            print(f"Target ratio for {counter} polygon is: {rate}")
            rates["shape"].append(f"{shapes.name}_{counter}")
            rates["rate"].append(rate)
            rates["total_cells"].append(np.delete(np.unique(ref_poly_only), 0).shape[0])
            rates["positive_cells"].append(int(np.delete(np.unique(ref_poly_only), 0).shape[0] * rate))
            rates["total_area"].append(Polygon(polygon_fit).area)
            rates["reference"].append(reference.name)
            rates["target"].append(target.name)
            counter += 1

        df = pd.DataFrame(rates)
        if os.path.isfile(savepath):
            df_load = pd.read_csv(savepath, index_col = 0)
            df_save = pd.merge(df_load, df, on=["shape", "reference", "target"], how="outer")
            df_save = df_remove_suffixes(df_save)
            df_save.to_csv(savepath)            
        else: 
            df.to_csv(savepath)


    else: 
        rate = calculate_rate(ref_data, target_data, coloc_thr)
        rates["rate"] = rate 
        rates["total_cells"].append(np.delete(np.unique(ref_data), 0).shape[0])
        rates["positive_cells"].append(int(np.delete(np.unique(ref_poly_only), 0).shape[0] * rate))
        rates["total_area"].append(ref_data.shape[0]* ref_data.shape[1])
        rates["reference"].append(reference.name)
        rates["target"].append(target.name)
        rates["shape"].append(shapes.name)
        df = pd.DataFrame(rates)
        if os.path.isfile(savepath):
            df_load = pd.read_csv(savepath, index_col = 0)
            df_save = pd.merge(df_load, df, on=["shape", "reference", "target"], how="outer")
            df_save = df_remove_suffixes(df_save)
            df_save.to_csv(savepath)    
        else: 
            df.to_csv(savepath)



@magic_factory(call_button ="Calculate")
def draw_contours(shapes:"napari.layers.Shapes", viewer :"napari.viewer.Viewer",data:str): 

    types = np.array([type(x) for x in viewer.layers])
    img_ind = np.argwhere(types == napari.layers.image.image.Image)
    if img_ind.shape[0]:
        img_shape = viewer.layers[img_ind[0][0]].data.shape
        empty_image = np.zeros(img_shape)
    else: 
        return
    if os.path.isfile(os.path.join(folder, data + ".csv")): 
        df = pd.read_csv(os.path.join(folder, data + ".csv"), index_col = 0)
    else: 
        return
    if shapes is not None and len(shapes.data) == df.loc[[shapes.name in i for i in df.index]].shape[0]: 
        labels = shapes.to_labels(img_shape)
        contours, hierarchy = cv2.findContours(labels,cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(empty_image, contours, -1, 1, 3)

        label_inds = np.delete(np.unique(labels), 0)
        counter = 0
        for i in label_inds:
            empty_image[labels == i ] = df.loc[[shapes.name in i for i in df.index]].rate.iloc[counter] 
            counter += 1

        viewer.add_image(empty_image, name="contours")

@magic_factory(call_button ="Calculate")
def stardist_segmentation(shapes:"napari.layers.Shapes", target:"napari.layers.Image", viewer:"napari.viewer.Viewer", method:str):
    
    if shapes is not None and target is not None: 
        for rectangle in shapes.data:
            crop =rectangle.astype(int)
            crop_x = np.unique(np.array([i[0] for i in crop]))
            crop_y = np.unique(np.array([i[1] for i in crop]))
            target_crop = target.data[crop_x.min(): crop_x.max(), crop_y.min(): crop_y.max()]
            mask = return_mask(target_crop, method = method)
            empty_target = np.zeros_like(target.data)
            empty_target[crop_x.min(): crop_x.max(), crop_y.min(): crop_y.max()] = mask




            viewer.add_image(empty_target, name = f"{method}_mask")


@magic_factory(call_button ="Calculate")
def calculate_antibody_intensity(reference: "napari.layers.Labels", target:"napari.layers.Image",antibody_image:"napari.layers.Image", shapes: "napari.layers.Shapes", viewer :"napari.viewer.Viewer",save:str,  coloc_thr = 0.2, property:str ="intensity_mean"): 

    savepath = os.path.join(folder, save + ".csv")



    ref_data = reference.data
    target_data = target.data
    antibody_data = antibody_image.data
    output= {"shape": [],"reference": [], "target": [], "intensity_positive_cells":[],"std_positive_cells":[], "intensity_negative_cells":[],"std_negative_cells":[], "num_positive_cells":[], "num_negative_cells":[]}

    if np.unique(target_data).shape[0] > 2: 
        #Binarize file if it is not a binary file 
        target_data = target.data > threshold_otsu(target_data)
    else: 
        target_data = target.data
    if shapes is not None and shapes.data: 
        counter = 0
        for polygon in shapes.data: 
            
        #Crop polygon coordinates not to exceed image dimensions
            polygon_fit = fit_dim(polygon, ref_data.shape)

        #get closest rectange
            rectangle = calculate_bounding_rectangle(polygon_fit)
            crop_x = np.unique(np.array([i[0] for i in rectangle]))
            crop_y = np.unique(np.array([i[1] for i in rectangle]))
        #work with cropped image
            ref_crop = ref_data[crop_x.min(): crop_x.max(), crop_y.min(): crop_y.max()]
            target_crop = target_data[crop_x.min(): crop_x.max(), crop_y.min(): crop_y.max()]
            antibody_crop = antibody_data[crop_x.min(): crop_x.max(), crop_y.min(): crop_y.max()]
        #remove additional labels that are not in the polygon
            poly_mask = generate_polygon_mask(ref_crop, rectangle, polygon_fit)
            ref_poly_only = crop_with_polygon_mask(ref_crop, poly_mask)
            target_poly_only = crop_with_polygon_mask(target_crop, poly_mask)
            antibody_poly_only = crop_with_polygon_mask(antibody_crop, poly_mask)

        #calculate rate for each region 
            positive_cells, negative_cells = return_positive_negative_cells(ref_poly_only, target_poly_only, coloc_thr)


            values_positive = calculate_props(positive_cells, antibody_poly_only, property)
            values_negative = calculate_props(negative_cells, antibody_poly_only, property)



            output["reference"].append(reference.name)
            output["target"].append(target.name)
            output["shape"].append(f"{shapes.name}_{counter}")
            output["intensity_positive_cells"].append(np.mean(values_positive))
            output["intensity_negative_cells"].append(np.mean(values_negative))
            output["num_positive_cells"].append(np.delete(np.unique(positive_cells),0 ).shape[0])
            output["num_negative_cells"].append(np.delete(np.unique(negative_cells),0 ).shape[0])
            output["std_positive_cells"].append(np.std(values_positive))
            output["std_negative_cells"].append(np.std(values_negative))

            counter += 1

            df = pd.DataFrame(output)
            if os.path.isfile(savepath): 
                df_load = pd.read_csv(savepath, index_col = 0)
                df_save = pd.merge(df_load, df, on=["shape", "reference", "target"], how= "outer")
                df_save = df_remove_suffixes(df_save)
                df_save.to_csv(savepath)
            else: 
                df.to_csv(savepath)


