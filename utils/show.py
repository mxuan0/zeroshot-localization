from skimage.measure import label, regionprops
#from skimage.measure import label as sklabel, regionprops
import numpy as np
from skimage.transform import rescale, resize
from skimage.draw import rectangle_perimeter
from copy import deepcopy
import pdb

# def show_cam_on_image(img: np.ndarray,
#                       mask: np.ndarray,
#                       use_rgb: bool = False,
#                       colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
#     if use_rgb:
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     heatmap = np.float32(heatmap) / 255

#     if np.max(img) > 1:
#         raise Exception(
#             "The input image should np.float32 in the range [0, 1]")

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     cam = heatmap + img
#     cam = cam / (np.max(cam) +1e-7) 
#     return np.uint8(255 * cam)

# def connected_component(image, image_size=50176/1000, connectivity=1):
#     image = image.astype('uint8')
#     nb_components, output, stats, centroids = label(image, connectivity=connectivity, )
#     sizes = stats[:, -1]

#     label_size = [(label, sizes[label]) for label in range(1, nb_components)]
#     if not label_size:
#       return None
#     label_size = sorted(label_size, key=lambda t: t[1], reverse=True)

#     components = []
#     for label, size in label_size:
#       img2 = np.zeros(output.shape)
  
#       if size > image_size:
#         img2[output == label] = 255
#         components.append(img2)
    
#     if not components:
#       return None
#     return components

def scale_cam_image(img, target_size=None):
    img = img - np.min(img)
    img = img / (1e-7 + np.max(img))
    if target_size is not None:
      img = resize(img, target_size)

    return img

def generate_bbox(image, threshold=0.65, connectivity=1, comp_size=50176/1000):
    bar = np.max(image)*threshold
    filtered = (image > bar).astype(int)
    pdb.set_trace()
    labeled = label(filtered, connectivity=connectivity)
    props = regionprops(labeled)
    if props is None:
      print('Only detected background.')
      return None
    
    return [p for p in props if p.area > comp_size]

def show_bbox_on_img(img, props):
  pdb.set_trace()
  bbox_img = deepcopy(img)
  for bbox in props:  
    row, col = rectangle_perimeter(start=(bbox.bbox[0]+1, bbox.bbox[1]+1), end=(bbox.bbox[2]-2, bbox.bbox[3]-2))
    bbox_img[row, col, :] = [255, 0, 0]
  return bbox_img

  