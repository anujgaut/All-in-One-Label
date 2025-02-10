# annotator/export_tools.py
import os
import json
from tkinter import messagebox

def export_yolo_format(app):
    if not app.image_obj or not app.image_path:
        return

    orig_width, orig_height = app.image_obj.size
    image_dir = os.path.dirname(app.image_path)
    labels_dir = os.path.join(image_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(app.image_path))[0]

    has_polygon = any(ann.type == "polygon" for ann in app.annotations)
    if has_polygon:
        annotation_file = os.path.join(labels_dir, base_name + "_mask.json")
        annotations_out = []
        for ann in app.annotations:
            if ann.type == "polygon":
                xs = ann.points[0::2]
                ys = ann.points[1::2]
                bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                try:
                    category_id = app.labels.index(ann.label)
                except ValueError:
                    category_id = -1
                annotations_out.append({
                    "category_id": category_id,
                    "segmentation": [ann.points],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3]
                })
            else:
                x1, y1, x2, y2 = ann.points
                bbox = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]
                try:
                    category_id = app.labels.index(ann.label)
                except ValueError:
                    category_id = -1
                segmentation = [[x1, y1, x2, y1, x2, y2, x1, y2]]
                annotations_out.append({
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3]
                })
        data = {"annotations": annotations_out}
        with open(annotation_file, "w") as f:
            json.dump(data, f, indent=2)
        app.system_message_label.config(
            text=f"Polygon annotations saved in Mask R-CNN format to:\n{annotation_file}"
        )
    else:
        annotation_file = os.path.join(labels_dir, base_name + ".txt")
        with open(annotation_file, "w") as f:
            for ann in app.annotations:
                x1, y1, x2, y2 = ann.points
                box_width = abs(x2 - x1)
                box_height = abs(y2 - y1)
                x_center = min(x1, x2) + box_width / 2
                y_center = min(y1, y2) + box_height / 2
                x_center_norm = x_center / orig_width
                y_center_norm = y_center / orig_height
                width_norm = box_width / orig_width
                height_norm = box_height / orig_height
                try:
                    class_index = app.labels.index(ann.label)
                except ValueError:
                    class_index = -1
                f.write(f"{class_index} {x_center_norm:.6f} {y_center_norm:.6f} "
                        f"{width_norm:.6f} {height_norm:.6f}\n")
        app.system_message_label.config(
            text=f"Annotations saved in YOLO format to:\n{annotation_file}"
        )
        app.image_status[app.image_path] = True
        app.tree.item(app.current_image_index, tags=("completed",))
    app.system_message_label.after(3000, lambda: app.system_message_label.config(text=""))

def export_voc_format(app):
    messagebox.showinfo("Export", "Exporting in Pascal VOC format... (Not fully implemented)")

def export_coco_format(app):
    messagebox.showinfo("Export", "Exporting in COCO JSON format... (Not fully implemented)")

def export_csv_format(app):
    messagebox.showinfo("Export", "Exporting annotations as CSV... (Not fully implemented)")
