import glob
import logging
import os

import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from sklearn import metrics
import gc
import matplotlib.pyplot as plt


def dump(save_dir, outputs):
    filenames = outputs["filename"]
    batch_size = len(filenames)
    preds = outputs["pred"].cpu().numpy()  # B x 1 x H x W
    masks = outputs["mask"].cpu().numpy()  # B x 1 x H x W
    heights = outputs["height"].cpu().numpy()
    widths = outputs["width"].cpu().numpy()
    clsnames = outputs["clsname"]
    for i in range(batch_size):
        file_dir, filename = os.path.split(filenames[i])
        _, subname = os.path.split(file_dir)
        filename = "{}_{}_{}".format(clsnames[i], subname, filename)
        filename, _ = os.path.splitext(filename)
        save_file = os.path.join(save_dir, filename + ".npz")
        np.savez(
            save_file,
            filename=filenames[i],
            pred=preds[i],
            mask=masks[i],
            height=heights[i],
            width=widths[i],
            clsname=clsnames[i],
        )


def merge_together(save_dir):
    npz_file_list = glob.glob(os.path.join(save_dir, "*.npz"))
    fileinfos = []
    preds = []
    masks = []
    gc.collect()
    for npz_file in npz_file_list:
        npz = np.load(npz_file)
        fileinfos.append(
            {
                "filename": str(npz["filename"]),
                "height": npz["height"],
                "width": npz["width"],
                "clsname": str(npz["clsname"]),
            }
        )
        preds.append(npz["pred"])
        masks.append(npz["mask"])
    preds = np.concatenate(np.asarray(preds), axis=0)  # N x H x W
    masks = np.concatenate(np.asarray(masks), axis=0)  # N x H x W
    return fileinfos, preds, masks


class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )


class EvalDataMeta:
    def __init__(self, preds, masks):
        self.preds = preds  # N x H x W
        self.masks = masks  # N x H x W


class EvalImage:
    def __init__(self, data_meta, **kwargs):
        self.preds = self.encode_pred(data_meta.preds, **kwargs)
        self.masks = self.encode_mask(data_meta.masks)
        self.preds_good = sorted(self.preds[self.masks == 0], reverse=True)
        self.preds_defe = sorted(self.preds[self.masks == 1], reverse=True)
        self.num_good = len(self.preds_good)
        self.num_defe = len(self.preds_defe)
        
        # Check if predictions are inverted for defective cases
        self.is_inverted = np.mean(self.preds_defe) < np.mean(self.preds_good)

        # If inverted, adjust predictions
        if self.is_inverted:
            self.preds = np.max(self.preds) - self.preds
            self.preds_good = self.preds[self.masks == 0]
            self.preds_defe = self.preds[self.masks == 1]

    @staticmethod
    def encode_pred(preds):
        raise NotImplementedError

    def encode_mask(self, masks):
        N, _, _ = masks.shape
        masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(np.int_)  # (N, )
        return masks

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc

    def get_classification_metrics(self, threshold=0.5):
        binary_preds = (self.preds >= threshold).astype(int)
        
        accuracy = np.mean(binary_preds == self.masks)
        precision = metrics.precision_score(self.masks, binary_preds)
        recall = metrics.recall_score(self.masks, binary_preds)
        f1_score = metrics.f1_score(self.masks, binary_preds)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def find_optimal_threshold(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        
        # Calculate Youden's J statistic for all thresholds
        youden_j = tpr - fpr
        
        # Find the threshold with the maximum Youden's J statistic
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold
    
    def plot_distribution(self):
        plt.hist(self.preds_good, bins=50, alpha=0.5, label='Good Predictions')
        plt.hist(self.preds_defe, bins=50, alpha=0.5, label='Defective Predictions')
        plt.xlabel('Prediction Scores')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Prediction Score Distribution')
        plt.show()

    def plot_roc_curve(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        plt.plot(fpr, tpr, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

    def print_summary_statistics(self):
        print(f"Summary Statistics for Predictions")
        print(f"Mean: {np.mean(self.preds)}")
        print(f"Median: {np.median(self.preds)}")
        print(f"Standard Deviation: {np.std(self.preds)}")
        print(f"Min: {np.min(self.preds)}")
        print(f"Max: {np.max(self.preds)}")

        print(f"Summary Statistics for Good Predictions")
        print(f"Mean: {np.mean(self.preds_good)}")
        print(f"Median: {np.median(self.preds_good)}")
        print(f"Standard Deviation: {np.std(self.preds_good)}")
        print(f"Min: {np.min(self.preds_good)}")
        print(f"Max: {np.max(self.preds_good)}")

        print(f"Summary Statistics for Defective Predictions")
        print(f"Mean: {np.mean(self.preds_defe)}")
        print(f"Median: {np.median(self.preds_defe)}")
        print(f"Standard Deviation: {np.std(self.preds_defe)}")
        print(f"Min: {np.min(self.preds_defe)}")
        print(f"Max: {np.max(self.preds_defe)}")

    def plot_confusion_matrix(self, threshold=0.5):
        binary_preds = (self.preds >= threshold).astype(int)
        cm = metrics.confusion_matrix(self.masks, binary_preds)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix at Threshold {threshold}')
        plt.show()


class EvalImageMean(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).mean(axis=1)  # (N, )


class EvalImageStd(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).std(axis=1)  # (N, )


class EvalImageMax(EvalImage):
    @staticmethod
    def encode_pred(preds, avgpool_size):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        preds = (
            F.avg_pool2d(preds, avgpool_size, stride=1).cpu().numpy()
        )  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )


class EvalPerPixelAUC:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc


eval_lookup_table = {
    "mean": EvalImageMean,
    "std": EvalImageStd,
    "max": EvalImageMax,
    "pixel": EvalPerPixelAUC,
}


def performances(fileinfos, preds, masks, config, verbose=False):
    ret_metrics = {}
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    for clsname in clsnames:
        preds_cls = []
        masks_cls = []
        for fileinfo, pred, mask in zip(fileinfos, preds, masks):
            if fileinfo["clsname"] == clsname:
                preds_cls.append(pred[None, ...])
                masks_cls.append(mask[None, ...])
        preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)  # N x H x W
        masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)  # N x H x W
        data_meta = EvalDataMeta(preds_cls, masks_cls)

        # auc
        if config.get("auc", None):
            for metric in config.auc:
                evalname = metric["name"]
                kwargs = metric.get("kwargs", {})
                eval_method = eval_lookup_table[evalname](data_meta, **kwargs)
                optimal_threshold = eval_method.find_optimal_threshold()
                metrics = eval_method.get_classification_metrics(optimal_threshold)
    
                if verbose:
                    eval_method.print_summary_statistics()
                    eval_method.plot_distribution()
                    eval_method.plot_roc_curve()
                    eval_method.plot_confusion_matrix(optimal_threshold)
                    
                auc = eval_method.eval_auc()
                ret_metrics["{}_{}_auc".format(clsname, evalname)] = auc
                ret_metrics["{}_{}_auc".format(clsname, "Accuracy")] = metrics["accuracy"]

    if config.get("auc", None):
        for metric in config.auc:
            evalname = metric["name"]
            evalvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for clsname in clsnames
            ]
            mean_auc = np.mean(np.array(evalvalues))
            ret_metrics["{}_{}_auc".format("mean", evalname)] = mean_auc

    return ret_metrics


def log_metrics(ret_metrics, config):
    logger = logging.getLogger("global_logger")
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean"])) + ["mean"]

    # auc
    if config.get("auc", None):
        auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
        evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))
        record = Report(["clsname"] + evalnames)

        for clsname in clsnames:
            clsvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for evalname in evalnames
            ]
            record.add_one_record([clsname] + clsvalues)

        logger.info(f"\n{record}")
