import numpy as np


def bboxes_filter(bboxes, pts):
    bboxes = np.array(bboxes)
    start = np.min(bboxes[:, 0]).astype(np.int32)
    end = np.max(bboxes[:, 0]).astype(np.int32)
    for i, t in enumerate(range(start, end+1)):
        if t not in bboxes[:, 0]:
            # bboxes = np.insert()
            value = [t, *bboxes[i-1, 1:]]
            bboxes = np.insert(bboxes, i, value, 0)
            pts.insert(i, [])
            
    # moving average
    pass
    return bboxes, pts


if __name__ == "__main__":
    bboxes = np.random.randint(0, 100, size=(10, 5))
    bboxes[:, 0] = [1, 2, 4, 5, 7, 8, 10, 11, 12, 13]
    print(bboxes_filter(bboxes))