from yoloface import yolov5
from model import FullGenerator
from skimage import transform as trans
from torch.nn import functional as F
import torch
import cv2 
import numpy as np
import os

//commiting into the git hub

def crop_with_ldmk(img, ldmk):
    std_ldmk = np.array([[193, 240], [319, 240],
                         [257, 314], [201, 371],
                         [313, 371]], dtype=np.float32) / 2
    tform = trans.SimilarityTransform()
    tform.estimate(ldmk, std_ldmk)
    M = tform.params[0:2, :]
    cropped = cv2.warpAffine(img, M, (256, 256), borderValue=0.0)
    return cropped

class Inference():
    def __init__(self, model_path, yolov5_path, device="cpu"):
        self.device = device
        self.G = FullGenerator(256, 512, 8, channel_multiplier=1, narrow=0.5, device=device).to(device)
        self.G.eval()
        self.G.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.yolonet = yolov5(yolov5_path, confThreshold=0.3, nmsThreshold=0.5, objThreshold=0.3)

    def inference(self, img_rgb):
        dets = self.yolonet.detect(img_rgb)
        dets = self.yolonet.postprocess(img_rgb, dets)
        [confidence, bbox, landmark] = dets[0]
        landmark = landmark.reshape([5, 2])
        aligned_img = crop_with_ldmk(img_rgb, landmark)
        with torch.no_grad():
            img_tensor = torch.tensor(aligned_img.copy(), dtype=torch.float32).to(self.device).permute(2, 0, 1)[None] / 127.5 - 1.0
            fake_img = self.G(img_tensor)
            res = (fake_img.clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy()[0] + 1.) * 127.5
        return res

# Define paths and device directly in the code
img_path = "D:\\Ahmed\\projects\\CartoonBANK-main\\ahmed.jpg"
save_path = "D:\\Ahmed\\projects\\CartoonBANK-main\\result.png"
cartoon_model = "D:\\Ahmed\\projects\\CartoonBANK-main\\saved_models\\style5.pth"
yoloface_model = "D:\\Ahmed\\projects\\CartoonBANK-main\\saved_models\\yolov5s-face.onnx"
device = "cpu"

infer = Inference(model_path=cartoon_model, yolov5_path=yoloface_model, device=device)
img_bgr = cv2.imread(img_path)[..., :3]
img_rgb = img_bgr[..., ::-1]
res = infer.inference(img_rgb.copy())

# Ensure the directory for the save path exists
save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir) and save_dir != "":
    os.makedirs(save_dir)


cv2.imwrite(save_path, res[..., ::-1])











# from yoloface import yolov5
# from model import FullGenerator
# from skimage import transform as trans
# from torch.nn import functional as F

# import torch
# import cv2 
# import numpy as np
# import os

# def crop_with_ldmk(img, ldmk):
#     std_ldmk = np.array([[193, 240], [319, 240],
#                          [257, 314], [201, 371],
#                          [313, 371]], dtype=np.float32) / 2
#     tform = trans.SimilarityTransform()
#     tform.estimate(ldmk, std_ldmk)
#     M = tform.params[0:2, :]
#     cropped = cv2.warpAffine(img, M, (256, 256), borderValue=0.0)
#     return cropped

# class Inference():
#     def __init__(self, model_path, yolov5_path, device="cuda"):
#         self.device = device
#         self.G = FullGenerator(256, 512, 8, channel_multiplier=1, narrow=0.5, device=device).to(device)
#         self.G.eval()
#         self.G.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
#         self.yolonet = yolov5(yolov5_path, confThreshold=0.3, nmsThreshold=0.5, objThreshold=0.3)

#     def inference(self, img_rgb):
#         dets = self.yolonet.detect(img_rgb)
#         dets = self.yolonet.postprocess(img_rgb, dets)
#         if len(dets) == 0:
#             return img_rgb  # If no detection, return the original image
#         [confidence, bbox, landmark] = dets[0]
#         landmark = landmark.reshape([5, 2])
#         aligned_img = crop_with_ldmk(img_rgb, landmark)
#         with torch.no_grad():
#             img_tensor = torch.tensor(aligned_img.copy(), dtype=torch.float32).to(self.device).permute(2, 0, 1)[None] / 127.5 - 1.0
#             fake_img = self.G(img_tensor)
#             res = (fake_img.clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy()[0] + 1.) * 127.5
#         return res 

# # Define paths and device directly in the code
# input_path = "C:\\Users\\wELCOME\\Downloads\\1232.mp4"  # Change to video file path if needed
# save_path = "result.mp4"  # Output video file
# cartoon_model = "D:\\Ahmed\\projects\\CartoonBANK-main\\saved_models\\style4.pth"
# yoloface_model = "D:\\Ahmed\\projects\\CartoonBANK-main\\saved_models\\yolov5s-face.onnx"
# device = "cpu"

# infer = Inference(model_path=cartoon_model, yolov5_path=yoloface_model, device=device)

# # Check if input is a video
# if input_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
#     cap = cv2.VideoCapture(input_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_count += 1
#         print(f"Processing frame {frame_count}...")
#         img_rgb = frame[..., ::-1]  # Convert BGR to RGB
#         res = infer.inference(img_rgb.copy())
#         res_bgr = res[..., ::-1]  # Convert RGB back to BGR
#         out.write(res_bgr)

#     cap.release()
#     out.release()
#     print("Video processing completed!")
# else:
#     img_bgr = cv2.imread(input_path)[..., :3]
#     img_rgb = img_bgr[..., ::-1]
#     res = infer.inference(img_rgb.copy())
#     # Ensure the directory for the save path exists
#     save_dir = os.path.dirname(save_path)
#     if not os.path.exists(save_dir) and save_dir != "":
#         os.makedirs(save_dir)
#     cv2.imwrite(save_path, res[..., ::-1])
