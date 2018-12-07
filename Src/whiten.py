import cv2 as cv
from path import PATH_TRN_VAL
from core.dataset import OCR_DATASET
from core.utils.average_meter import AverageMeter

dataset = OCR_DATASET()
avg_mtr_mean = AverageMeter(NVars=3)
avg_mtr_var = AverageMeter(NVars=3)
for ith,(img, lbl) in enumerate(dataset):
    print(ith)
    img=img.numpy()
    avg_mtr_mean.update(val=[img[ch].mean()
                             for ch in range(3)],
                        n=img[0].size)
    avg_mtr_var.update(val=[img[ch].var()
                            for ch in range(3)],
                       n=img[0].size)

print(avg_mtr_mean.avg())
print(avg_mtr_var.avg())
