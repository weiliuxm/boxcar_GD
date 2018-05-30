import os,pickle
import cv2
import numpy as  np
import math



#%%
# change this to your location
#BOXCARS_DATASET_ROOT = "/mnt/matylda1/isochor/Datasets/BoxCars116k/" 
BOXCARS_DATASET_ROOT = "/media/weiliu/data/datasets/BoxCars116k"

#%%
BOXCARS_IMAGES_ROOT = os.path.join(BOXCARS_DATASET_ROOT, "images")
BOXCARS_DATASET = os.path.join(BOXCARS_DATASET_ROOT, "dataset.pkl")
BOXCARS_ATLAS = os.path.join(BOXCARS_DATASET_ROOT, "atlas.pkl")
BOXCARS_CLASSIFICATION_SPLITS = os.path.join(BOXCARS_DATASET_ROOT, "classification_splits.pkl")

def load_cache(path, encoding="latin-1", fix_imports=True):
    """
    encoding latin-1 is default for Python2 compatibility
    """
    with open(path, "rb") as f:
        return pickle.load(f, encoding=encoding, fix_imports=True)
    
    
    
    
## get the  insection of two lines    
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1, a2, b1, b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float))*db + b1






def getFocal(vp1, vp2, pp):
    return math.sqrt(- np.dot(vp1[0:2]-pp[0:2], vp2[0:2]-pp[0:2]))

def getViewpoint(p, vp1, vp2, pp):
    try:
        focal = getFocal(vp1, vp2, pp)
    except ValueError:
        return None    
    vp1W = np.concatenate((vp1[0:2]-pp[0:2], [focal]))
    vp2W = np.concatenate((vp2[0:2]-pp[0:2], [focal]))
    if vp1[0] < vp2[0]:
        vp2W = -vp2W
    vp3W = np.cross(vp1W, vp2W)
    vp1W, vp2W, vp3W = tuple(map(lambda u: u/np.linalg.norm(u), [vp1W, vp2W, vp3W]))
    pW = np.concatenate((p[0:2]-pp[0:2], [focal]))
    pW = pW/np.linalg.norm(pW)
    viewPoint = -np.dot(np.array([vp1W, vp2W, vp3W]), pW)
    return viewPoint


dataset = load_cache(BOXCARS_DATASET)
atlas = load_cache(BOXCARS_ATLAS)

camera_name = dataset['samples'][0]["camera"]
vp1 = dataset['cameras'][camera_name]['vp1']
vp2 = dataset['cameras'][camera_name]['vp2']
pp = dataset['cameras'][camera_name]['pp']


bb3d = dataset['samples'][0]["instances"][0]['3DBB']
bb3d_offset =  dataset['samples'][0]["instances"][0]['3DBB_offset']
bb3d_cropped = bb3d - bb3d_offset
center_bb3d = seg_intersect(bb3d_cropped[0], bb3d_cropped[6], bb3d_cropped[3],bb3d_cropped[5])
view_point = getViewpoint(center_bb3d, vp1, vp2, pp)
print(view_point)
print(np.arccos(view_point))

#center_x 
print(dataset['samples'][0]["instances"][0].keys())
#bb3d_cropped = bb_3d - 3DBB_offset
#print(bb2d)
#print(bb3d)
#print(bb3d_offset)
print(bb3d - bb3d_offset)

bb2d = dataset['samples'][0]["instances"][0]['2DBB']
image = cv2.imdecode(atlas[0][0], 1)
pt1 = (int(bb2d[0]), int(bb2d[1]))
pt2 = (int(bb2d[0]) + int(bb2d[2]), int(bb2d[1]) + int(bb2d[3]))
cv2.rectangle(image, pt1, pt2, (0,0,255))
cv2.imshow('original',image)
image = cv2.imdecode(atlas[0][0], 1)
image_cropped = image[int(bb2d[1]):int(bb2d[1]) + int(bb2d[3]),
                      int(bb2d[0]):int(bb2d[0]) + int(bb2d[2])]
cv2.imshow('cropped',image_cropped)
image_resized = cv2.resize(image_cropped,(224,224))
cv2.imshow('resized',image_resized)
cv2.waitKey(0)
#split = load_cache(BOXCARS_CLASSIFICATION_SPLITS)["hard"]
print(dataset.keys())
#print (dataset['samples'][0]["instances"])#dataset['samples'] is a list
#print (dataset['cameras']['031'])
#print(split.keys())
#print(split['types_mapping'])
#print(len(split['types_mapping']))
#print(split)
