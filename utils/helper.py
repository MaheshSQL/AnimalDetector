import math
import onnxruntime
from PIL import Image
import numpy as np
import time
import cv2
import torch
import torchvision
from matplotlib import pyplot as plot
from PIL import Image, ImageFont, ImageDraw, ImageEnhance



def getImage(input_image_path):
    
    input_image = Image.open(input_image_path)
    input_image = input_image.convert('RGB')
    input_image = input_image.resize(resize)

    input_image_numpy = np.array(input_image,dtype=np.float32)
    input_image_numpy.shape
    
    input_image_numpy = np.swapaxes(input_image_numpy,0,2)
    input_image_numpy.shape
    
    input_image_numpy = np.expand_dims(input_image_numpy, axis=0)
    input_image_numpy.shape
    
    input_image_numpy /= 255 # 0 - 255 to 0.0 - 1.0
    
    return input_image_numpy   

def readImage(input_image_path, imgsz, stride):
    # Read image
    img0 = cv2.imread(input_image_path)  # BGR    
    print(img0.shape)
    
     # Padded resize
    img = letterbox(im=img0, new_shape=imgsz, stride=stride)[0]    
    print(img.shape)
    
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    
    print(img.shape)
    
    return img, img0


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
#         LOGGER.warning(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
        print(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
        
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU        
        
def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.3 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            #LOGGER.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            print(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output

def output_to_target(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls in o:            
#             targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
            targets.append([cls, box[0],box[1],box[2],box[3], conf])
    return np.array(targets)

def loadModel(model_path, providers=['CPUExecutionProvider','CUDAExecutionProvider']):
    ort_session = onnxruntime.InferenceSession(model_path,providers=providers)
    
    return ort_session

def getPrediction(ort_session, input_image_numpy):
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: input_image_numpy}

    start = time.time()

    ort_outs = ort_session.run([ort_session.get_outputs()[0].name], ort_inputs)[0]

    end = time.time()
    print(f'Inference took {((end - start)):.3f} seconds.') 
    
    return ort_outs

def showPredictionPIL(input_image_path,bbox_xyxy_list, conf_out_list, cls_list, names):
    
    PILImage = Image.open(input_image_path).convert('RGB')
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(PILImage)
    
    for i, bbox in enumerate(bbox_xyxy_list):
            
        draw.rectangle(((bbox[0],bbox[1]), (bbox[2],bbox[3])), outline='yellow', width=2)
        draw.text((bbox[0],bbox[1]-10), f'{names[cls_list[i]]} '+'{:.0f}'.format(conf_out_list[i]*100) + '%', font=font, fill="yellow")   

    fig = plot.figure(figsize=(10, 10))
    plot.imshow(PILImage)
    plot.show()

def processInput(model, input_cv2_numpy_image,img_size, stride, conf_thres, iou_thres, classes, names, max_det, device):
    
    #-----------------------------------------------------#

    imgsz = check_img_size((img_size,), s=stride)
    imgsz

    #-----------------------------------------------------#
    
#     img_out = readImage(input_path,img_size, stride)
    
    #-----------------------------------------------------#
    img0 = None
    
#     img0 = cv2.imread(input_path)  # BGR    
    img0 = input_cv2_numpy_image
        
#     print(img0.shape)
    
     # Padded resize
    img = letterbox(im=img0, new_shape=img_size, stride=stride)[0]    
#     print(img.shape)
    
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    
    #-----------------------------------------------------#   
#     img = img_out[0]
#     img0 = img_out[1]
    #-----------------------------------------------------#
    
    ort_session = model

    im = torch.from_numpy(img).to(device)

    #MK:Modified
    # im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im = im.float()  # uint8 to fp16/32

    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

#     print(im.shape)

    im = im.cpu().numpy()  # torch to numpy
    # y = ort_session.run([ort_session.get_outputs()[0].name], {ort_session.get_inputs()[0].name: im})[0]
    y = getPrediction(ort_session, im)

#     print(f'y[0].shape: {y[0].shape}')

    if isinstance(y, np.ndarray):
        y = torch.tensor(y, device=device)

#     print(f'y[0].shape: {y[0].shape}')
    #-----------------------------------------------------#
    # NMS
    pred = non_max_suppression(y, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes, max_det=max_det)

#     print(pred[0].shape)
#     pred[0]
    #-----------------------------------------------------#
    # Process predictions
    for i, det in enumerate(pred):  # per image

        #MK changed variable being passed per implementation above
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh    
        imc = img0
        s = ''
        
        bbox_xyxy_list = []
        conf_out_list = []
        cls_list = []

        if len(det):
            
#             print(f'img0.shape:{img0.shape}')
            
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

#             print(s)            

            # Write results
            for *xyxy, conf, cls in reversed(det):            
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = [cls.item(), *xywh, conf.item()]  # label format MK: Modified

                #print(xywh)
#                 print(f'line (cls, *xywh, conf):{line}')
#                 print(f'xyxy:{xyxy}')

    #             bbox_xywh = [item*img_size for item in xywh]
    #             bbox_xywh = [xywh[0]*img0.shape[1],xywh[1]*img0.shape[0],xywh[2]*img0.shape[1],xywh[3]*img0.shape[0]]
    #             print(f'bbox_xywh: {bbox_xywh}')

                bbox_xyxy = [xyxy[0].item(),xyxy[1].item(),xyxy[2].item(),xyxy[3].item()]
#                 print(f'bbox_xyxy: {bbox_xyxy}')

                conf_out = conf.item()
#                 print(f'conf_out:{conf_out}')

                bbox_xyxy_list.append(bbox_xyxy)
                conf_out_list.append(conf_out)
                cls_list.append(int(cls.item()))
                
#         #Outside the loop
#         if showImage:
#             showPredictionPIL(input_path,bbox_xyxy_list, conf_out_list, cls_list, names)
        
        return bbox_xyxy_list, conf_out_list, cls_list, names
    #-----------------------------------------------------#   
