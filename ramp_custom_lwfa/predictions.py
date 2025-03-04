import numpy as np
from rampwf.prediction_types.detection import Predictions as DetectionPredictions


def make_custom_predictions(iou_threshold):
    """Create class CustomPredictions using iou_threshold when bagging."""

    class CustomPredictions(DetectionPredictions):
        @classmethod
        def combine(cls, predictions_list, index_list=None):
            """Combine multiple predictions into a single one."""
            if index_list is None:
                index_list = range(len(predictions_list))

            # Get predictions arrays
            y_pred_list = [predictions_list[i].y_pred for i in index_list]
            n_images = len(y_pred_list[0])
            combined_predictions = []

            # Combine predictions for each image
            for i in range(n_images):
                all_preds = []
                for y_pred in y_pred_list:
                    if i < len(y_pred) and y_pred[i] is not None:
                        if isinstance(y_pred[i], list):
                            all_preds.extend(y_pred[i])
                        else:
                            all_preds.extend(y_pred[i].tolist())
                combined_predictions.append(all_preds)

            # Apply NMS if we have any predictions
            if any(len(preds) > 0 for preds in combined_predictions):
                from .geometry import apply_NMS_to_predictions
                filtered_predictions = apply_NMS_to_predictions(
                    combined_predictions, iou_threshold=iou_threshold
                )
            else:
                filtered_predictions = combined_predictions

            return cls(y_pred=np.array(filtered_predictions, dtype=object))
        
        @property
        def valid_indexes(self):
            """Return valid indexes, handling empty predictions."""
            try:
                if len(self.y_pred.shape) > 1:
                    # If shape is (n,0), handle as empty predictions
                    if self.y_pred.shape[1] == 0:
                        return np.ones(self.y_pred.shape[0], dtype=bool)
                    # If shape is (n,1) or any other wrong shape, raise error
                    else:
                        raise ValueError(
                            f"Predictions have wrong shape {self.y_pred.shape}. "
                            "Expected 1D array of lists, got 2D array. "
                            "Each prediction should be a list of detections."
                        )
                else:
                    # Handle normal case
                    empty = np.empty(len(self.y_pred), dtype=object)
                    for i in range(len(empty)):
                        empty[i] = []
                    return ~np.array([np.array_equal(p, []) for p in self.y_pred])
            except Exception as e:
                print(f"Warning: Error in valid_indexes: {str(e)}. Using empty predictions.")
                return np.ones(len(self.y_pred), dtype=bool)

    return CustomPredictions
