# 你需要安装 openvino 和 opencv-python
# pip install openvino opencv-python

import cv2
import numpy as np
from openvino.runtime import Core

# ----------- 配置参数（与 yml 保持一致） -----------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
scale = 1.0 / 255.0

def resize_img_type0(img, limit_type="max", limit_side_len=960):
    h, w = img.shape[:2]
    ratio = 1.0
    if limit_type == "min":
        min_wh = min(h, w)
        if min_wh < limit_side_len:
            ratio = limit_side_len / float(h) if h < w else limit_side_len / float(w)
    else:
        max_wh = max(h, w)
        if max_wh > limit_side_len:
            ratio = limit_side_len / float(h) if h > w else limit_side_len / float(w)
    resize_h = int(h * ratio)
    resize_w = int(w * ratio)
    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)
    resize_img = cv2.resize(img, (resize_w, resize_h))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return resize_img, ratio_h, ratio_w

# ----------- 前处理函数 -----------
def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"图片读取失败，请检查路径或文件是否存在: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, ratio_h, ratio_w = resize_img_type0(img, limit_type="max", limit_side_len=960)
    img = img.astype(np.float32) * scale
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # batch
    return img, ratio_h, ratio_w

# ----------- OpenVINO 推理并画外接矩形 -----------
def infer_and_save(ir_model_xml, img_path, out_path, draw_rect=True):
    core = Core()
    model = core.read_model(model=ir_model_xml)
    compiled_model = core.compile_model(model, "CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    input_data, ratio_h, ratio_w = preprocess(img_path)
    result = compiled_model([input_data])[output_layer]

    seg_map = result[0, 0]  # shape: (H, W)
    seg_map_img = (seg_map * 255).astype(np.uint8)
    cv2.imwrite(out_path, seg_map_img)
    print(f"分割图已保存到: {out_path}")

    # 画最小外接矩形并映射到原图
    if draw_rect:
        _, binary = cv2.threshold(seg_map_img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        src_img = cv2.imread(img_path)
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box[:, 0] = (box[:, 0] / ratio_w).astype(np.int32)
            box[:, 1] = (box[:, 1] / ratio_h).astype(np.int32)
            cv2.polylines(src_img, [box], True, color=(0, 255, 0), thickness=2)
        rect_out_path = out_path.replace('.png', '_rect.png')
        cv2.imwrite(rect_out_path, src_img)
        print(f"外接矩形已画在原图并保存到: {rect_out_path}")

# ----------- 使用示例 -----------
if __name__ == "__main__":
    ir_model_xml = "/root/pyProjcet/PaddleOCR3.1/PaddleOCR/myTest/project1/test/train/IR/inference.xml"
    img_path = "/root/pyProjcet/PaddleOCR3.1/PaddleOCR/myTest/project1/dianrong1.bmp"
    out_path = "/root/pyProjcet/PaddleOCR3.1/PaddleOCR/seg_map_VINO.png"
    infer_and_save(ir_model_xml, img_path, out_path)