import copy
import logging
import numpy as np
import onnxruntime as rt
from argus_camera import ArgusCamera
from defs import (
    VisionClasses,
    STREAM_RESOLUTION,
    CONVERTER_RESOLUTION,
    NETWORK_RESOLUTION,
    DIGITAL_GAIN_RANGE,
    GAIN_RANGE,
    AWB_MODE,
    WB_GAINS,
    DETECTION_THRESH,
    NON_MAX_THRESH,
    IMG_MEAN,
    IMG_STD,
    LINE_THICKNESS,
    CIRCLE_SIZE,
)
from PIL import Image, ImageColor, ImageFont, ImageDraw


class JetsonDetect:
    def __init__(self, onnx_model, cam_exposure_time):
        self.onnx_model = onnx_model
        self.cam_exposure_time = cam_exposure_time

        # Load the onnx model and allocate tensors.
        self.argus_camera = None
        self.onnx_interpreter = None
        self.input_details = None
        self.output_scores = None
        self.output_boxes = None

        self.log = logging.getLogger(__name__)

    def start(self):
        self.__argus_camera = ArgusCamera(
            stream_resolution=STREAM_RESOLUTION,
            video_converter_resolution=CONVERTER_RESOLUTION,
            isp_digital_gain_range=DIGITAL_GAIN_RANGE,
            gain_range=GAIN_RANGE,
            awb_mode=AWB_MODE,
            wb_gains=WB_GAINS,
            exposure_time_range=(self.cam_exposure_time, self.cam_exposure_time),
        )
        if self.__argus_camera is None:
            self.log.fatal("Failed to initalise camera")
            return False

        # Load the ONNX model and allocate tensors.
        self.onnx_interpreter = rt.InferenceSession(self.onnx_model)
        self.input_details = self.onnx_interpreter.get_inputs()[0].name
        self.output_scores = self.onnx_interpreter.get_outputs()[0].name
        self.output_boxes = self.onnx_interpreter.get_outputs()[1].name
        return True

    def non_max_suppression(self, boxes, overlap_thresh):

        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute area and sort boxes
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have overlap greater than the
            # provided overlap threshold
            idxs = np.delete(
                idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
            )
        return boxes[pick].astype("int")

    def pre_process_image(self, image, copy_for_drawing=True):
        # resize the image to the network resolution
        pil_image = Image.fromarray(np.uint8(image))
        if copy_for_drawing:
            draw_frame = copy.deepcopy(pil_image)
        else:
            draw_frame = None
        pil_image = pil_image.resize(NETWORK_RESOLUTION)

        # swap the red and blue channels
        r, g, b = pil_image.split()
        pil_image = Image.merge("RGB", (b, g, r))

        # convert to numpy array
        input_array = np.asarray(pil_image)
        pixels = input_array.astype("float32")

        # normalize the image
        pixels = pixels - IMG_MEAN
        pixels = pixels * IMG_STD
        input_data = pixels.transpose()
        input_data = np.expand_dims(input_data, axis=0)

        return input_data, draw_frame

    def draw_boxes(self, img, boxes):
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("./arial.ttf", size=30)
        for d in boxes:
            # top = d[0], bottom = d[2], left = d[1], right = d[3]
            ctr_x = (d[3] + d[1]) / 2
            ctr_y = (d[0] + d[2]) / 2
            vis_class = VisionClasses(d[4])
            label = vis_class.name

            text_origin = tuple(
                np.array([d[1] + LINE_THICKNESS, d[0] + LINE_THICKNESS])
            )
            color = ImageColor.getrgb("yellow")
            draw.ellipse(
                (
                    ctr_x - CIRCLE_SIZE,
                    ctr_y - CIRCLE_SIZE,
                    ctr_x + CIRCLE_SIZE,
                    ctr_y + CIRCLE_SIZE,
                ),
                fill=CIRCLE_FILL,
                outline=(0, 0, 0),
            )
            draw.rectangle(
                [
                    left + LINE_THICKNESS,
                    top + LINE_THICKNESS,
                    right - LINE_THICKNESS,
                    bottom - LINE_THICKNESS,
                ],
                outline=color,
            )
            draw.text(text_origin, label, fill=color, font=font)

    def process_image(self):
        input_frame = self.__argus_camera.read()

        input_data = self.pre_process_image(input_frame)

        results = self.onnx_interpreter.run(
            [self.output_scores, self.output_boxes],
            {self.input_details: input_data},
        )
        scores = results[0]
        boxes = results[1]

        box_list = np.empty((0, 5), float)
        for i, score in enumerate(scores[0]):
            idx = int(np.argmax(score))
            confidence = score[idx]
            if idx > 0 and confidence > DETECTION_THRESH:
                det_list = []
                det_list.extend(boxes[0][i])
                det_list.append(confidence)
                box = boxes[0][i]
                box = np.append(box, idx)
                box = np.multiply(
                    box,
                    np.array(
                        [
                            CONVERTER_RESOLUTION[1],
                            CONVERTER_RESOLUTION[0],
                            CONVERTER_RESOLUTION[1],
                            CONVERTER_RESOLUTION[0],
                            1.0,
                        ]
                    ),
                )
                box_list = np.vstack([box_list, box])

        boxes = self.non_max_suppression(box_list, NON_MAX_THRESH)

        self.draw_boxes(draw_frame, boxes)

        return draw_frame, boxes
