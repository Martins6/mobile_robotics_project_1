from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import os
from numpngw import write_apng


def create_image_depth(fileimage, model, processor):
    image=Image.open(fileimage)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    return depth


if __name__ == "__main__":
  IMAGE_FOLDER_PATH="./data/raw_images"
  OUTPUT_IMAGEPATH="./data/turtle_run_depth.png"

  processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
  model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

  depth_list=[]
  for filename in os.listdir(IMAGE_FOLDER_PATH):
    if filename not in [".config"]:
      filename=os.path.join(IMAGE_FOLDER_PATH, filename)
      depth=create_image_depth(filename)
      depth_list.append(depth)
  depth_list=[np.array(i) for i in depth_list]

  write_apng(OUTPUT_IMAGEPATH, depth_list, delay=500)