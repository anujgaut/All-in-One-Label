# annotator/models.py
import json

class Annotation:
    def __init__(self, ann_type, points, label, attributes=None, canvas_ids=None):
        """
        ann_type: 'bbox' or 'polygon'
        points: For bbox: [x1, y1, x2, y2]. For polygon: [x1, y1, x2, y2, ...]
        label: The object class label.
        attributes: Extra info (currently not used).
        canvas_ids: List of canvas item IDs (for redrawing).
        """
        self.type = ann_type
        self.points = points
        self.label = label
        self.attributes = attributes if attributes is not None else {}
        self.canvas_ids = canvas_ids if canvas_ids is not None else []

    def to_dict(self):
        return {
            "type": self.type,
            "points": [int(p) for p in self.points],
            "label": self.label,
            "attributes": self.attributes
        }

    @staticmethod
    def from_dict(d):
        return Annotation(d["type"], d["points"], d["label"], d.get("attributes", {}), [])
