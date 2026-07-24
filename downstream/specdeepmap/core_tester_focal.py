import csv
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from osgeo import gdal
from torchvision.transforms import v2

try:
    from enmapbox.apps.SpecDeepMap.core_deep_learning_trainer_mica_vit import MyModel as VitMyModel
except ImportError:
    from core_deep_learning_trainer_mica_vit import MyModel as VitMyModel

try:
    from enmapbox.apps.SpecDeepMap.core_deep_learning_trainer_mica_swin import MyModel as SwinMyModel
except ImportError:
    from core_deep_learning_trainer_mica_swin import MyModel as SwinMyModel

# import albumentations as A

### Data augmentation

transforms_v2 = v2.Compose([
    v2.RandomRotation(degrees=45),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
])


def _resolve_model_class(hyperpara: dict):
    architecture = str(hyperpara.get("architecture", ""))
    backbone = str(hyperpara.get("backbone", "")).lower()
    if architecture == "DPT" or "vit" in backbone:
        return VitMyModel
    return SwinMyModel


def load_model_and_tile_size(model_checkpoint, acc):
    # Load the model checkpoint
    if acc =='gpu':
        acc_d = 'cuda'
    else:
        acc_d='cpu'
    checkpoint = torch.load(model_checkpoint, map_location=torch.device(acc_d), weights_only=False)

    # Retrieve hyperparameters from the checkpoint
    hyperpara = checkpoint['hyper_parameters']
    model_cls = _resolve_model_class(hyperpara)

    # Extract tile size and other relevant hyperparameters
    tile_size_x = hyperpara['img_x']
    tile_size_y = hyperpara['img_y']
    num_classes = hyperpara['classes']

    in_channels = hyperpara["in_channels"]
    architecture_used = hyperpara['architecture']
    backbone_used = hyperpara['backbone']
    pre_process = hyperpara['preprocess']
    remove_c = hyperpara['remove_background_class']
    cls_values = hyperpara["class_values"]
    reverse_mapping = hyperpara["reverse_mapping"]
    scaler = hyperpara.get("scaler", None)




    print('reverse mapping',reverse_mapping)

    # Load the model with the extracted hyperparameters
    model = model_cls.load_from_checkpoint(
        model_checkpoint,
        hparams={
            "architecture": architecture_used,
            "backbone": backbone_used,
            "transform": None,
            "class_weights_balanced": False,
            "acc_type": acc,
            "batch_size": 1,
            "classes": num_classes,
            "in_channels": in_channels,
            "preprocess": pre_process,
            'remove_background_class': remove_c,
            "reverse_mapping": reverse_mapping,
            "scaler": scaler,

        }
    )

    # Set the model to evaluation mode
    model.eval()

    # Return the model and tile sizes
    return model, tile_size_x, tile_size_y, num_classes, remove_c, cls_values


def read_image_with_gdal(image_path):
    dataset = gdal.Open(image_path)

    # channel first
    data_array = dataset.ReadAsArray().astype(np.float32)

    image = dataset.ReadAsArray()  # Automatically reads all bands
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    no_data_value = dataset.GetRasterBand(1).GetNoDataValue()

    # Create a no-data mask for the first band
    no_data_mask = None

    if no_data_value is not None:
        # no_data_mask = image[0] == no_data_value

        first_band = dataset.GetRasterBand(1)
        nodata_first_band = first_band.GetNoDataValue()

        # Read data from modified raster
        nodata_first_band_arr = first_band.ReadAsArray()
        # data_arr = dataset.ReadAsArray(0)

        no_data_mask = (nodata_first_band_arr == nodata_first_band)

    dataset = None  # Close the GDAL dataset
    return image, geotransform, projection, no_data_value, no_data_mask


def save_prediction_as_geotiff(pred_array, geotransform, projection, output_path, no_data_value, no_data_mask):
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(output_path, pred_array.shape[1], pred_array.shape[0], 1)
    out_raster.SetGeoTransform(geotransform)
    out_raster.SetProjection(projection)
    out_band = out_raster.GetRasterBand(1)

    out_band.WriteArray(pred_array)

    if no_data_value != None:
        out_band.SetNoDataValue(no_data_value)
    out_raster.FlushCache()
    out_raster = None


def compute_iou_per_class(pred, gt, cls_values):
    """Compute IoU for each class given a list of class values."""
    ious = []  # Use a list to store IoU values


    gt = gt.astype(pred.dtype)

    for cls in cls_values:

        pred_cls = (pred == cls)
        gt_cls = (gt == cls)
        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union == 0:
            ious.append(np.nan)  # Avoid dividing by zero
        else:
            ious.append(intersection / union)

    return ious


def compute_metrics_per_class(pred, gt, cls_values):
    """
    Compute accuracy, recall, precision, and F1 score per class
    
    Args:
        pred: Prediction array (H, W)
        gt: Ground truth array (H, W)
        cls_values: List of class values to evaluate
    
    Returns:
        Dictionary with metrics per class: {'accuracy': [...], 'recall': [...], 'precision': [...], 'f1': [...]}
    """
    pred = pred.astype(np.int64)
    gt = gt.astype(np.int64)
    
    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    
    for cls in cls_values:
        # True Positives: predicted as cls and actually cls
        tp = np.logical_and(pred == cls, gt == cls).sum()
        # False Positives: predicted as cls but actually not cls
        fp = np.logical_and(pred == cls, gt != cls).sum()
        # False Negatives: actually cls but predicted as not cls
        fn = np.logical_and(pred != cls, gt == cls).sum()
        # True Negatives: predicted as not cls and actually not cls
        tn = np.logical_and(pred != cls, gt != cls).sum()
        
        # Accuracy per class (pixel-level accuracy for this class)
        total_pixels = tp + fp + fn + tn
        if total_pixels > 0:
            accuracy = (tp + tn) / total_pixels
        else:
            accuracy = np.nan
        
        # Recall (Sensitivity) = TP / (TP + FN)
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = np.nan
        
        # Precision = TP / (TP + FP)
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = np.nan
        
        # F1 Score = 2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 and not (np.isnan(precision) or np.isnan(recall)):
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = np.nan
        
        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
    
    return {
        'accuracy': accuracy_list,
        'recall': recall_list,
        'precision': precision_list,
        'f1': f1_list
    }


# iou_per_class = compute_iou_per_class(pred, gt, cls_dict)

def process_images_from_csv(csv_file, model_checkpoint, acc_device=None, export_folder=None, csv_output_path=None,
                            no_data_label_mask=False, feedback=None):
    acc_options = ['cpu', 'gpu']
    acc = acc_options[acc_device]


    model, _, _, num_classes, remove_c, cls_values = load_model_and_tile_size(model_checkpoint, acc)


    if acc =='gpu':
        acc_d = 'cuda'
    else:
        acc_d='cpu'
    model.to(acc_d)

    print(num_classes)

    csv_folder = Path(csv_file).parent
    df = pd.read_csv(csv_file)
    for col in ["image", "mask"]:
        df[col] = df[col].apply(lambda rel_path: str(csv_folder / Path(rel_path)))

    all_ious = []
    all_metrics = {
        'accuracy': [],
        'recall': [],
        'precision': [],
        'f1': []
    }

    len_counter = len(df)
    if export_folder:
        len_counter = len(df) * 2

    counter = 0

    print('remove', remove_c)

    # variables to check zero presence in mask, if yes first class in iou count is skipped and num_classes is extended by 1
    Zero_in_mask = False
    # b = 0

    for index, row in df.iterrows():

        if feedback and feedback.isCanceled():
            raise KeyboardInterrupt("Prediction process canceled by user.")

        image_path = row['image']
        mask_path = row['mask']  # Assuming a 'mask' column with ground truth paths

        # Read the image using GDAL, including the no-data mask
        image, geotransform, projection, no_data_value, no_data_mask = read_image_with_gdal(image_path)

        # Create an empty prediction array for the entire image
        full_prediction = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)

        # Assuming the image is in [channels, height, width] format and normalized
        image = np.expand_dims(image, axis=0)

        # Make prediction using the model
        image = image.astype(np.float32)

        image = image

        image = torch.tensor(image).to(acc_d)

        full_prediction = model.predict(image)
        full_prediction_iou = full_prediction
        #preds = preds  ## adjsut

        #pred_np = preds.cpu().numpy()

        # Store the predictions for this image
        #full_prediction[:, :] = pred_np  # Directly copy the entire predicted image

        # Load the ground truth mask
        mask, _, _, _, _ = read_image_with_gdal(mask_path)

        # Overwrite predictions where the mask is 0 if `no_data_label_mask` is True

        if remove_c == 'Yes':
            full_prediction_iou[mask == 0] = 0

        if no_data_label_mask == True:
            full_prediction[mask == 0] = 0

            #full_prediction_iou = full_prediction[mask == 0] = 0  # no_data_value

        # Calculate IoU per class
        #print('pred_dtype',full_prediction_iou)
        print('gt_dtype',mask.dtype)
        full_prediction = full_prediction.astype(np.int64)
        mask = mask.astype(np.int64)
        print('mask UNIQUE',np.unique(mask))
        print('mask shape', np.shape(mask))
        print('full_dtype', full_prediction.dtype)
        print('full_unique', np.unique(full_prediction))
        print('full_shape', np.shape(full_prediction))
        ious = compute_iou_per_class(full_prediction_iou, mask, cls_values)
        print(cls_values)
        print(ious)
        all_ious.append(ious)
        
        # Compute accuracy, recall, precision, F1 per class
        metrics = compute_metrics_per_class(full_prediction_iou, mask, cls_values)
        all_metrics['accuracy'].append(metrics['accuracy'])
        all_metrics['recall'].append(metrics['recall'])
        all_metrics['precision'].append(metrics['precision'])
        all_metrics['f1'].append(metrics['f1'])

        # count first loop
        counter += 1

        if feedback:
            progress = (counter / len_counter) * 100
            feedback.setProgress(progress)

        # Export the prediction as a georeferenced GeoTIFF if export_folder is provided
        if export_folder:

            if no_data_mask is not None:
                full_prediction[no_data_mask] = 0  ######no_data_value
            # Create the output file path, using the original image filename with '_prediction' added
            original_filename = os.path.basename(image_path)
            output_filename = os.path.splitext(original_filename)[0] + '_prediction.tif'
            output_path = os.path.join(export_folder, output_filename)

            save_prediction_as_geotiff(full_prediction, geotransform, projection, output_path, no_data_value,
                                       no_data_mask)
            print(f"Exported prediction to {output_path}")
            counter += 1
            if feedback:
                progress = (counter / len_counter) * 100
                feedback.setProgress(progress)

    # Calculate the mean IoU across all images for each class
    mean_iou_per_class = np.nanmean(all_ious, axis=0)
    mean_iou = np.nanmean(mean_iou_per_class)
    
    # Calculate mean metrics per class
    mean_accuracy_per_class = np.nanmean(all_metrics['accuracy'], axis=0)
    mean_recall_per_class = np.nanmean(all_metrics['recall'], axis=0)
    mean_precision_per_class = np.nanmean(all_metrics['precision'], axis=0)
    mean_f1_per_class = np.nanmean(all_metrics['f1'], axis=0)
    
    # Calculate overall means
    mean_accuracy = np.nanmean(mean_accuracy_per_class)
    mean_recall = np.nanmean(mean_recall_per_class)
    mean_precision = np.nanmean(mean_precision_per_class)
    mean_f1 = np.nanmean(mean_f1_per_class)

    print(f"Mean IoU per class: {mean_iou_per_class}")
    print(f"Mean IoU across all classes: {mean_iou}")
    print(f"\nMean Accuracy per class: {mean_accuracy_per_class}")
    print(f"Mean Accuracy: {mean_accuracy}")
    print(f"\nMean Recall per class: {mean_recall_per_class}")
    print(f"Mean Recall: {mean_recall}")
    print(f"\nMean Precision per class: {mean_precision_per_class}")
    print(f"Mean Precision: {mean_precision}")
    print(f"\nMean F1 Score per class: {mean_f1_per_class}")
    print(f"Mean F1 Score: {mean_f1}")

    # write csv with actual class values and all metrics
    if csv_output_path:
        with open(csv_output_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Class', 'IoU', 'Accuracy', 'Recall', 'Precision', 'F1_Score'])

            # Write metrics for each class using actual class values
            for cls, iou, acc, rec, prec, f1 in zip(cls_values, mean_iou_per_class, 
                                                     mean_accuracy_per_class, mean_recall_per_class,
                                                     mean_precision_per_class, mean_f1_per_class):
                writer.writerow([cls, iou, acc, rec, prec, f1])

            # Write the mean metrics in the last row
            writer.writerow(['Mean', mean_iou, mean_accuracy, mean_recall, mean_precision, mean_f1])

        print(f"Metrics per class and mean metrics written to {csv_output_path}")
