
def update_bbox(bbox1: ((float, float, float), (float, float, float)),
                bbox2: ((float, float, float), (float, float, float))):
    (Xmin1, Ymin1, Zmin1), (Xmax1, Ymax1, Zmax1) = bbox1
    (Xmin2, Ymin2, Zmin2), (Xmax2, Ymax2, Zmax2) = bbox2
    return ((min(Xmin1, Xmin2), min(Ymin1, Ymin2), min(Zmin1, Zmin2)),
            (max(Xmax1, Xmax2), max(Ymax1, Ymax2), max(Zmax1, Zmax2)))