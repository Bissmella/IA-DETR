
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache

import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.logger import create_small_table
import torch.nn.functional as F

from detectron2.evaluation.evaluator import DatasetEvaluator
from ..utils import box_ops



class BboxEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC AP.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that this is a rewrite of the official Matlab API.
    The results should be similar, but not identical to the one produced by
    the official API.
    """

    def __init__(self, dataset_name, output_dir, device):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        self.output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self.device = device
        # meta = MetadataCatalog.get(dataset_name)
        dirname = "/home/bibahaduri/dataset/PascalVOC"
        self._anno_file_template = os.path.join(
             dirname, "Annotations", "{}.xml"
         )
        split = "val"
        self._image_set_path = os.path.join(
             dirname, "ImageSets", "Main", split + ".txt"
         )
        self._class_names = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
                            ]
        self.classesNum = len(self._class_names)
        # # add this two terms for calculating the mAP of different subset
        # self._base_classes = meta.base_classes
        # self._novel_classes = meta.novel_classes
        # assert meta.year in [2007, 2012], meta.year
        # self._is_2007 = meta.year == 2007
        # self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self.iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95 zjq
        self.niou = self.iouv.numel()
        self.stats = []
        

    def reset(self):
        self.stats = []  # class name -> list of prediction strings

    def process(self, inputs, outputs, classname= None):

        #TODO outputs are not normalized format
        #change pred_boxes to good shape
        #create targets from the inputs
        #breakpoint()
        """
        boxes = outputs['pred_boxes']
        for i in range(boxes.shape[0]):
            boxes[i, :, 0] *= inputs[i]['width']  # x_min
            boxes[i, :, 1] *= inputs[i]['height']  # y_min
            boxes[i, :, 2] *= inputs[i]['width']  # x_max
            boxes[i, :, 3] *= inputs[i]['height']
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        probas = outputs['pred_logits']
        scores = F.softmax(probas, -1)
        #breakpoint()
        
        scores, labels = scores.max(-1)
        scores = scores.unsqueeze(-1)
        #TODO from following 2 only one should be selected
        
        classes = torch.zeros_like(probas[:, :, :1]).float()

        classes[labels.unsqueeze(-1)[:,:, 0] == 2] = self.classesNum
        classes[labels.unsqueeze(-1)[:,:, 0] == 0] = self.classesNum
        
        """
        
        boxes = torch.cat([out['boxes'].unsqueeze(0) for out in outputs], dim =0)
        scores = torch.cat([out['scores'].unsqueeze(0).unsqueeze(2) for out in outputs], dim=0)
        classes = torch.cat([out['labels'].unsqueeze(0).unsqueeze(2) for out in outputs], dim=0)
        
        out = torch.cat((boxes, scores, classes.float()), dim=-1 )
        targets = []

        for idx, batch_per_image in enumerate(inputs):  
            batch_ratios = torch.tensor([batch_per_image['orig_size'][1], batch_per_image['orig_size'][0], batch_per_image['orig_size'][1], batch_per_image['orig_size'][0]]).to(self.device)
            tar_boxes = box_ops.box_cxcywh_to_xyxy(batch_per_image['boxes']) * batch_ratios
            combined_targets = torch.cat((tar_boxes, batch_per_image['labels'].unsqueeze(1).float()), dim=1)
            
            targets.append(combined_targets)
        

        for si, pred in enumerate(out):
            
            if classname != None and targets[si].shape[0] > 0:
                labels = targets[si][targets[si][:, 4] == float(classname[si])]
            elif targets[si].shape[0] > 0:
                labels = targets[si]
            else:
                labels = []
            nl = len(labels)
            tcls = labels[:, 4].tolist() if nl else []  # target class
            ##path = Path(paths[si])
            ##seen += 1
            
            ##stats = []
            if len(pred) == 0:
                if nl:
                    self.stats.append((torch.zeros(0, self.niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            ##scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            """
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                    "class_id": int(cls),
                                    "box_caption": "%s %.3f" % (names[cls], conf),
                                    "scores": {"class_score": conf},
                                    "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                    'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                    'bbox': [round(x, 3) for x in b],
                                    'score': round(p[4], 5)})
            """
            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], self.niou, dtype=torch.bool, device=self.device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 4]

                # target boxes
                tbox = labels[:, 0:4] ##box_ops.box_cxcywh_to_xyxy(labels[:, 1:5])   # xywh2xyxy
                ## scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                ## if plots:
                ##    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                ##for cls in torch.unique(tcls_tensor):
                for clss in torch.unique(tcls_tensor):
                    ti = (clss == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices

                    ##pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices
                        
                    
                    # just getting the positive predictions
                    clss_pred = torch.tensor([1]).float()
                    pi = (pred[:, 5] == clss_pred[0] ).nonzero(as_tuple=False).view(-1)   #pred[:, 5] == cls   changed to one

                    pred[pi, 5] = float(clss)
                    # Search for detections
                    ## if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_ops.box_iou(predn[pi, :4], tbox[ti])[0].max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > self.iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > self.iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

            # Append statistics (correct, conf, pcls, tcls)
            self.stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))



    def evaluate(self):
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy

        p, r, mp, mr, map50, map = 0., 0., 0., 0., 0., 0.
        ap_class = []
        names = {k: v for k, v in enumerate(self._class_names)}
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=self.classesNum)  # number of targets per class
        else:
            nt = torch.zeros(1)


        pf = '%20s' + '%12i'  + '%12.4g' * 4  # print format
        #print(pf % ('all', nt.sum(), mp, mr, map50, map))
        nc =91
        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            try:
                maps[c] = ap[i]
            except:
                breakpoint()
        #breakpoint()
        stats = {'mp': mp, 'mr':mr, 'map50':map50, 'map': map}
        maps_d = {index: value for index, value in enumerate(maps)}
        return stats, maps_d



def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c           #** here it should be compared with 0
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    """
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')
    """

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')




def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

