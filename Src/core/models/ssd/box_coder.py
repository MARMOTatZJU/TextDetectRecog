'''Encode object boxes and labels.'''
'''Adapted for TextBoxes++'''
import math
import torch
import itertools

if __name__=='__main__':
    import sys
    sys.path.append(r'../../../')
from core.utils import meshgrid
from core.utils.box import change_box_order, box_iou, box_nms, box_hrzt_rect

# Adapted from SSDBoxCoder for TextBoxes++
class TextBoxesppCoder:
    def __init__(self, ssd_model):
        self.steps = ssd_model.steps
        self.box_sizes = ssd_model.box_sizes
        self.aspect_ratios = ssd_model.aspect_ratios
        self.fm_sizes = ssd_model.fm_sizes
        self.default_boxes = self._get_default_boxes()
        self.default_boxes_xyxy = change_box_order(self.default_boxes, 'xywh2xyxy')
    def _get_default_boxes(self):
        boxes = []
        for i, fm_size in enumerate(self.fm_sizes):
            for h, w in itertools.product(range(fm_size), repeat=2):
                cx = (w + 0.5) * self.steps[i]
                cy = (h + 0.5) * self.steps[i]
                s = self.box_sizes[i]
                boxes.append((cx, cy, s, s))

                s = math.sqrt(self.box_sizes[i] * self.box_sizes[i+1])
                boxes.append((cx, cy, s, s))

                s = self.box_sizes[i]
                for ar in self.aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))
        return torch.Tensor(boxes)  # xywh

    def encode(self, boxes_irr, labels=None):
        '''Encode target bounding boxes and class labels.

        SSD coding rules:
          tx = (x - anchor_x) / (variance[0]*anchor_w)
          ty = (y - anchor_y) / (variance[0]*anchor_h)
          tw = log(w / anchor_w) / variance[1]
          th = log(h / anchor_h) / variance[1]

        Args:
          boxes_irr: (tensor) bounding boxes sized [#obj, 8].
                          (x1,y1,x2,y2,x3,y3,x4,y4)
          labels: (tensor) object class labels, sized [#obj,].

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,12].
                                (dx,dy,dw,dh,dx1,dy1,dx2,dy2,dx3,dy3,dx4,dy4)
          cls_targets: (tensor) encoded class labels, sized [#anchors,].

        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_coder.py
        '''
        def argmax(x):
            v, i = x.max(0)
            j = v.max(0)[1][0]
            return (i[j], j)
        boxes = change_box_order(box_hrzt_rect(boxes_irr), 'xywh2xyxy')
        if isinstance(labels, type(None)):
            labels = torch.LongTensor(boxes_irr.shape[0]).fill_(0)
        default_boxes = self.default_boxes_xyxy  # xywh
        default_boxes_xyxy = self.default_boxes_xyxy  # xywh

        ious = box_iou(default_boxes_xyxy, boxes)  # [#anchors, #obj]
        index = torch.LongTensor(len(default_boxes_xyxy)).fill_(-1) # anchor -> gt,
                                                                # [#anchors, #obj]
        masked_ious = ious.clone()
        # stage 1: first match
        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i,j] < 1e-6:
                break
            index[i] = j
            masked_ious[i,:] = 0 # clear anchor i
            masked_ious[:,j] = 0 # clear bbox j

        # stage 2L match the supplimentary
        mask = (index<0) & (ious.max(1)[0]>=0.5)
        if mask.any():
            index[mask] = ious[mask.nonzero().squeeze()].max(1)[1]

        boxes_irr = boxes_irr[index.clamp(min=0)] # [#anchors, ]
        boxes = boxes[index.clamp(min=0)] # [#anchors, ]
        boxes = change_box_order(boxes, 'xyxy2xywh')

        variances = (0.1, 0.2)
        # quadrilateral rectangle regression
        loc_xy_irr = torch.Tensor(len(default_boxes_xyxy), 8) # [#anchors, 8]
        loc_xy_irr[:,[0,2,4,6]] = \
            (boxes_irr[:,[0,2,4,6]]-default_boxes_xyxy[:,[0,2,2,0]])/default_boxes[:,[2]]
        loc_xy_irr[:,[1,3,5,7]] = \
            (boxes_irr[:,[1,3,5,7]]-default_boxes_xyxy[:,[1,1,3,3]])/default_boxes[:,[3]]
        # horizontal rectangle regression
        loc_xy = (boxes[:,:2]-default_boxes[:,:2]) / default_boxes[:,2:] / variances[0]
        loc_wh = torch.log(boxes[:,2:]/default_boxes[:,2:]) / variances[1]
        # concat regression outputs
        loc_targets = torch.cat([loc_xy,loc_wh,loc_xy_irr], 1)
        cls_targets = 1 + labels[index.clamp(min=0)] # objects
        cls_targets[index<0] = 0 #background
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        '''Decode predicted loc/cls back to real box locations and class labels.

        Args:
          loc_preds: (tensor) predicted loc, sized [8732,4].
          cls_preds: (tensor) predicted conf, sized [8732,21].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.

        Returns:
          boxes: (tensor) bbox locations, sized [#obj,4].
          labels: (tensor) class labels, sized [#obj,].
        '''
        variances = (0.1, 0.2)
        xy = loc_preds[:,:2] * variances[0] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
        wh = torch.exp(loc_preds[:,2:]*variances[1]) * self.default_boxes[:,2:]
        box_preds = torch.cat([xy-wh/2, xy+wh/2], 1)

        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.size(1)
        for i in range(num_classes-1):
            score = cls_preds[:,i+1]  # class i corresponds to (i+1) column
            mask = score > score_thresh
            if not mask.any():
                continue
            box = box_preds[mask.nonzero().squeeze()]
            score = score[mask]

            keep = box_nms(box, score, nms_thresh)
            boxes.append(box[keep])
            labels.append(torch.LongTensor(len(box[keep])).fill_(i))
            scores.append(score[keep])

        boxes = torch.cat(boxes, 0)
        labels = torch.cat(labels, 0)
        scores = torch.cat(scores, 0)
        return boxes, labels, scores

def test_TextBoxesppCoder():
    boxes_irr = torch.Tensor([[226.6,343.78,225.45,394.72,334.6,396.25,336.52,339.56,],
                           [40.86,298.14,38.95,391.34,68.31,389.42,83.63,295.59,]])
    from net import SSD300
    model = SSD300()
    boxcoder = TextBoxesppCoder(model)
    print('# default boxes: %d'%boxcoder.default_boxes.shape[0])
    print(boxcoder.default_boxes[10:20,...])

    loc_targets, cls_targets = boxcoder.encode(boxes_irr[:1, :])
    print(loc_targets.shape, cls_targets.shape)
    
    print(cls_targets.sum())

if __name__=='__main__':
    test_TextBoxesppCoder()
