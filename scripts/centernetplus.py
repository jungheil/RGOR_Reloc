# -*- coding: utf-8 -*-

import cv2
import numpy as np
import onnxruntime as ort


class CenterNetPlus:
    def __init__(
        self,
        onnx_path,
        intra_threads=0,
        k=100,
        nms_thresh=1.0,
        mean=(0.406, 0.456, 0.485),
        std=(0.225, 0.224, 0.229),
    ):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = intra_threads

        self._ort_session = ort.InferenceSession(
            onnx_path,
            # providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            providers=["CUDAExecutionProvider"],
            sess_options=options,
        )

        self.img_size = (
            self._ort_session.get_inputs()[0].shape[2],
            self._ort_session.get_inputs()[0].shape[3],
        )
        self.input_name = self._ort_session.get_inputs()[0].name
        self.mean = mean
        self.std = std
        self.stride = 4
        self.gs = 1.0
        self.grid_cell = self._create_grid(self.img_size, self.stride)
        self.k = k
        self.nms_thresh = nms_thresh

    def preproc(self, img):
        img = cv2.resize(img, self.img_size).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        img = img[:, :, (2, 1, 0)].transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def _create_grid(self, input_size, stride):
        h, w = input_size
        # generate grid cells
        ws, hs = w // stride, h // stride
        grid_y, grid_x = np.meshgrid(np.arange(hs), np.arange(ws))
        grid_xy = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
        grid_xy = grid_xy.reshape(1, hs * ws, 2)

        return grid_xy

    def _decode_boxes(self, xywh):
        """
        input box :  [delta_x, delta_y, sqrt(w), sqrt(h)]
        output box : [xmin, ymin, xmax, ymax]
        """
        output = np.zeros_like(xywh)
        xywh[:, :, :2] = self.grid_cell + self.gs * xywh[:, :, :2] - (self.gs - 1.0) / 2
        xywh[:, :, 2:] = xywh[:, :, 2:]

        # [c_x, c_y, w, h] -> [xmin, ymin, xmax, ymax]
        output[:, :, 0] = xywh[:, :, 1] - xywh[:, :, 2] / 2
        output[:, :, 1] = xywh[:, :, 0] - xywh[:, :, 3] / 2
        output[:, :, 2] = xywh[:, :, 1] + xywh[:, :, 2] / 2
        output[:, :, 3] = xywh[:, :, 0] + xywh[:, :, 3] / 2

        output[:, :, (0, 2)] = np.clip(
            output[:, :, (0, 2)] * self.stride / self.img_size[1], 0.0, 1.0
        )
        output[:, :, (1, 3)] = np.clip(
            output[:, :, (1, 3)] * self.stride / self.img_size[0], 0.0, 1.0
        )

        return output

    def topk_np_(self, arr, k, dim=-1):
        idx = np.argpartition(-arr, kth=k, axis=dim)
        idx = idx.take(indices=range(k), axis=dim)
        val = np.take_along_axis(arr, indices=idx, axis=dim)
        sorted_idx = np.argsort(-val, axis=dim)
        idx = np.take_along_axis(idx, indices=sorted_idx, axis=dim)
        val = np.take_along_axis(val, indices=sorted_idx, axis=dim)
        return val, idx

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.shape[2]
        ind = ind.reshape(-1, 1).repeat(1, dim).reshape(ind.shape[0], ind.shape[1], dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _topk(self, cls):
        B, C, H, W = cls.shape

        topk_scores, topk_inds = self.topk_np_(cls.reshape(B, C, -1), self.k)
        topk_inds = topk_inds % (H * W)

        topk_score, topk_ind = self.topk_np_(topk_scores.reshape(B, -1), self.k)
        topk_clses = (topk_ind / self.k).astype(np.int32)
        topk_inds = topk_inds.reshape(B, -1)
        topk_inds = np.take_along_axis(topk_inds.reshape(B, -1), topk_ind, axis=1)

        return topk_score, topk_inds, topk_clses

    def nms(self, dets, scores, thresh):
        """ "Pure Python NMS baseline."""
        x1 = dets[:, 0]  # xmin
        y1 = dets[:, 1]  # ymin
        x2 = dets[:, 2]  # xmax
        y2 = dets[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def postproc(self, feat, cls, xy, wh):
        xywh = np.concatenate([xy, wh], axis=1).transpose(0, 2, 3, 1).reshape(1, -1, 4)
        bbox = self._decode_boxes(xywh)
        topk_scores, topk_inds, topk_clses = self._topk(cls)

        topk_scores = topk_scores[0]
        topk_cls_inds = topk_clses[0]
        topk_cls = cls.reshape(1, -1, cls.shape[1])[0, topk_inds[0], :]
        topk_bbox = bbox[0, topk_inds[0], :]
        topk_xywh = xywh[0, topk_inds[0], :]
        topk_xywh[:, [1, 2]] = topk_xywh[:, [1, 2]] * self.stride / self.img_size[1]
        topk_xywh[:, [0, 3]] = topk_xywh[:, [0, 3]] * self.stride / self.img_size[0]

        if self.nms_thresh < 1.0:
            keep = self.nms(topk_bbox, topk_scores, self.nms_thresh)
            topk_bbox = topk_bbox[keep]
            topk_xywh = topk_xywh[keep]
            topk_scores = topk_scores[keep]
            topk_cls_inds = topk_cls_inds[keep]
            topk_cls = topk_cls[keep]
            
        # keep = topk_scores > 0.4
        # topk_bbox = topk_bbox[keep]
        # topk_xywh = topk_xywh[keep]
        # topk_scores = topk_scores[keep]
        # topk_cls_inds = topk_cls_inds[keep]
        # topk_cls = topk_cls[keep]

        return (
            topk_bbox,
            topk_xywh,
            topk_scores,
            topk_cls_inds,
            topk_cls,
        )

    def predict(self, img):
        img_shape = img.shape
        img = self.preproc(img)
        ort_inputs = {self.input_name: img}
        ort_outs = self._ort_session.run(None, ort_inputs)
        feat, cls, xy, wh = ort_outs
        box, xywh, score, cls_ind, cls_vct = self.postproc(feat, cls, xy, wh)
        box[:, [0, 2]] *= img_shape[1]
        box[:, [1, 3]] *= img_shape[0]
        xywh[:, [1, 2]] *= img_shape[1]
        xywh[:, [0, 3]] *= img_shape[0]
        
        

        return box, xywh, score, cls_ind, cls_vct, feat

    def __call__(self, img):
        return self.predict(img)


if __name__ == "__main__":
    model = CenterNetPlus("centernet_plus.onnx", nms_thresh=0.1)
    img = cv2.imread("test.jpg")

    box, _, score, cls_ind, cls_vct, _ = model.predict(img)

    for i in range(len(box)):
        cv2.rectangle(
            img,
            (int(box[i][0]), int(box[i][1])),
            (int(box[i][2]), int(box[i][3])),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img,
            f"{cls_ind[i]}: {score[i]:.2f}",
            (int(box[i][0]), int(box[i][1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    cv2.imshow("result", img)
    cv2.waitKey(0)
