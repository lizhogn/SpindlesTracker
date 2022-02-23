from .tmp import cvat_dataset

def spindle_detect(img):
    bboxes = None
    mask = None
    return bboxes, mask


def data_generate():
    xml_file = "/home/zhognli/YOLOX/datasets/sample2/annotations.xml"
    img_dir  = "/home/zhognli/YOLOX/datasets/sample2/images"
    dataset = cvat_dataset.CVATVideoDataset(img_dir=img_dir, anno_path=xml_file)
    for i in range(len(dataset)):
        img, mask, bboxes, points = dataset[i]
        yield img, bboxes, mask
    