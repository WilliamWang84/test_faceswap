import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import shutil
from PIL import Image
from deepface import DeepFace
#from glasses_detector import GlassesClassifier
from pathlib import Path
from tqdm import tqdm
#from yoloface import face_analysis
#face=face_analysis()


class GrayImageReader:
    """
    A generic class to read an image and ensure it is grayscale.

    Accepts three types of input:
    1. A string representing the file path to an image.
    2. A 3-dimensional NumPy array representing a color image (in BGR format).
    3. A 2-dimensional NumPy array representing a grayscale image.

    The processed image is always available as a 2D grayscale NumPy array.
    """
    def __init__(self, image_input):
        """
        Initializes the reader and processes the input image.

        Args:
            image_input (str | np.ndarray): The input image, which can be a
                filepath, a color cv_image, or a grayscale cv_image.

        Raises:
            TypeError: If the input is not a string or a NumPy array.
            FileNotFoundError: If the input is a string but the file cannot be found.
            ValueError: If the input NumPy array is not 2D or 3D.
        """
        self.gray_image = None
        self._process_input(image_input)

    def _process_input(self, image_input):
        """Internal method to handle the different input types."""
        if isinstance(image_input, str):
            # Input is a filepath
            self.gray_image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if self.gray_image is None:
                raise FileNotFoundError(f"Could not read image from path: {image_input}")
        
        elif isinstance(image_input, np.ndarray):
            # Input is a NumPy array (cv_image)
            if image_input.ndim == 2:
                # Already a grayscale image
                self.gray_image = image_input
            elif image_input.ndim == 3:
                # Color image, convert to grayscale
                self.gray_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError(f"Input NumPy array must be 2D or 3D, but got {image_input.ndim} dimensions.")
        
        else:
            raise TypeError("Input must be a filepath (str) or a cv_image (np.ndarray).")

    def get_image(self) -> np.ndarray:
        """
        Returns the processed grayscale image.

        Returns:
            np.ndarray: A 2D NumPy array representing the grayscale image.
        """
        return self.gray_image

class VJDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(28, 28), paint_color=(255, 0, 0), paint_thickness=2, default_cascade='haarcascade_frontalface_default.xml'):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        self.paint_color = paint_color
        self.paint_thickness = paint_thickness
        self.fc = cv2.CascadeClassifier(cv2.data.haarcascades + default_cascade)

    def detect(self, img):
        ls_faces = []
        reader_gray = GrayImageReader(img)
        img_gray = reader_gray.get_image()
        
        if img_gray.ndim == 2:
            faces = self.fc.detectMultiScale(img_gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors, minSize=self.min_size)
            for (x, y, w, h) in faces:
                ls_faces.append([x, y, w, h])
        else:
            raise Exception("Invalid input, must be a valid filepath or a cv image (2D or 3D numpy array)")

        return ls_faces


    def display(self, cv_img, bboxes):
        dis_img = cv_img.copy()
        for (x, y, w, h) in bboxes:
            cv2.rectangle(dis_img, (x, y), (x+w, y+h), self.paint_color, self.paint_thickness)
        cv2.imshow("Face Detection", dis_img)
        cv2.waitKey(0) # Wait indefinitely for a key press
        cv2.destroyAllWindows()

    def crop(self, cv_img, bboxes, save_path='./', save_prefix='test_', save_suffix='.jpg', target_size=None):        
        for ind, (x, y, w, h) in enumerate(bboxes):
            img = cv_img.copy()
            if w >= target_size[0] and h >= target_size[1]:
                img_crop = img[y:y+h, x:x+w]
                if target_size:
                    img_crop = self.resize(img_crop, target_size)
                crop_seq = f"{(ind+1):03d}"
                cv2.imwrite(save_path+save_prefix+crop_seq+save_suffix, img_crop)

    def resize(self, cv_img, target_size=(256, 256)):
        return cv2.resize(cv_img, target_size)

class RecursiveImgParser:
    def __init__(self, root_dir=None, img_formats=['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']):
        self.root_dir = root_dir
        self.img_formats = img_formats
        self.img_paths = []
    
    def parse(self, target_dir=None, sorted=0):
        parse_dir = "./"
        if target_dir:
            parse_dir = target_dir
        elif self.root_dir:
            parse_dir = self.root_dir

        for ext in self.img_formats:
            # Construct the pattern for recursive search
            pattern = os.path.join(parse_dir, '**', ext)
            # Use glob.glob with recursive=True
            self.img_paths.extend(glob.glob(pattern, recursive=True))
        
        if sorted == -1:
            # Sort in descending / reverse order
            return self.img_paths.sort(reverse=True)
        elif sorted == 1:
            # Sort in ascending order
            return self.img_paths.sort()
        else:
            return self.img_paths

    def clear(self):
        self.img_paths.clear()

    def ls_print(self):
        if self.img_paths:
            for img_path in self.img_paths:
                print(img_path)
    
    def name_parser(self, input_path):
        pathparts = Path(input_path)
        return pathparts.parts[-2]

    def get_unique_count(self, ls_imgs):
        count_set = set()
        for img_path in ls_imgs:
            facenamewithid = os.path.basename(img_path)
            #print(facenamewithid)
            face_names = facenamewithid.split('_')
            #print(face_names)
            fullname = ''.join(face_names[:-2])
            #print(fullname)
            count_set.add(fullname)
        return len(count_set)

class GlassesAnalyser:
    def __init__(self, size='medium'):
        self.size = size
        self.kinds = ['sunglasses', 'eyeglasses', 'shadows']
        self.result_dict = {'present':True, 'absent':False}
        self.classifiers = {kd: GlassesClassifier(size=self.size, kind=kd) for kd in self.kinds}
    
    def analyze(self, img_path=None):
        res = {}
        for kd in self.kinds:
            res[kd] = self.result_dict[self.classifiers[kd].process_file(img_path)]
        return res

class DeepfaceAnalyser:
    def __init__(self, actions=['age', 'gender', 'emotion', 'race']):
        self.actions = actions
    
    def analyze(self, img_path=None):
        res = {}
        analysis = DeepFace.analyze(
            img_path=img_path,
            actions=self.actions
        )
        if analysis and isinstance(analysis, list):
            for i, face_info in enumerate(analysis):
                for act in self.actions:
                    res[act] = face_info[act] if act == 'age' else face_info['dominant_'+act]
        return res

vjd = VJDetector()
def detect_face_in_cvframe(cvframe=None, output_format='PIL'):
    ls_dets = vjd.detect(cvframe)
    #print(ls_dets)
    if len(ls_dets) >= 1:
        print("Multiple face handing not yet implemented, using the first face detected")
        x, y, w, h = ls_dets[0]
        if output_format == 'PIL':
            # Convert cv to pil
            img = cv2.cvtColor(cvframe[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            pil_res = Image.fromarray(img)        
            return pil_res, ls_dets[0]
        elif output_format == 'CV':
            return cvframe[y:y+h, x:x+w], ls_dets[0]
        else:
            raise Exception("Output image format must be PIL or CV (default PIL)")

    return None, None

def detect_face_in_pilimage(pilimage=None, output_format='PIL'):
    #print(pilimage)
    # Convert pil to cv
    np_img = np.array(pilimage)
    cvimg = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    faceimg, roi = detect_face_in_cvframe(cvframe=cvimg, output_format=output_format)
    return faceimg, roi


class DatasetUpdater:
    def __init__(self, root="./"):
        self.root = root
        self.rip = RecursiveImgParser(self.root)

    def naming_formatter(self, unformatted_name):
        # Convert unformatted name -> str to formatted
        # 1. split by space or comma
        lsnames = re.split(r'[ ,]', unformatted_name)
        
        # 2. capitalize first char
        ls_caps = []
        for nm in lsnames:
            ls_caps.append(nm.capitalize())

        # 3. _ join splitted capitalized
        return "_".join(ls_caps)

    def get_imgfile_ext(self, imgpath):
        img_name = os.path.basename(imgpath)
        lsimgnms = img_name.split('.')

        return lsimgnms[-1]

    def create(self, src_id=None, src_dir=None):
        if src_id:
            # Create a folder named src_id
            if isinstance(src_id, str):
                src_id = self.naming_formatter(src_id)
                new_dir = os.path.join(self.root, src_id)
                os.makedirs(new_dir, exist_ok=True)

            # Glob src_dir, find all .*p*g, return a list
            ls_files = self.rip.parse(src_dir)
            cnt_file = 0
            for f in ls_files:
                #formatted_seq = '{number:0{width}d}'.format(width=3, number=4)
                ext = self.get_imgfile_ext(f)

                formatted_seq = str(cnt_file).zfill(len(str(len(ls_files)))+1)
                dst_file = src_id + '_' + formatted_seq + '.' + ext
                newpath = os.path.join(new_dir, dst_file)
                shutil.copyfile(f, newpath)
                cnt_file += 1

            print(str(cnt_file)+" images added for "+ src_id)
            return cnt_file

        # On fail return -1    
        return -1

    def retrieve(self, dst_id=None):
        if dst_id:
            # Get file info seq: resolution, type(should be all jpg), color/gray
            ls_info = []
            cnt_file = 0
            glob_dir = os.path.join(self.root, dst_id)
            ls_files = self.rip.parse(glob_dir)
            for f in ls_files:
                dict_f = {}
                img_name = os.path.basename(f)
                dict_f['name'] = img_name                
                ext = self.get_imgfile_ext(f)
                dict_f['type'] = ext
                imf = cv2.imread(f)
                [H, W, D] = imf.shape
                dict_f['res'] = (W, H)
                dict_f['color'] = 'gray' if D == 1 else 'color'
                ls_info.append(dict_f)
                cnt_file += 1

            for info in ls_info:
                print(info)
            
            print(str(cnt_file)+" images info retrieved for "+ dst_id)
            return cnt_file    

        # On fail return -1    
        return -1 

    def update(self, src_id=None, dst_id=None):
        if src_id and dst_id:
            cnt_file = 0
            formatted_dst = self.naming_formatter(dst_id)
            # Check if src_id exists in self.root
            towalk_dir = os.path.join(self.root, src_id)
            if os.path.isdir(towalk_dir):
                # Replace everything src_id with dst_id
                os.makedirs(os.path.join(self.root, formatted_dst), exist_ok=True)
                ls_files = self.rip.parse(towalk_dir)
                for f in ls_files:
                    newfile = f.replace(src_id, formatted_dst)
                    shutil.move(f, newfile)
                    cnt_file += 1
                os.rmdir(towalk_dir)
                
                print(str(cnt_file)+" images updated for "+ dst_id)
                return cnt_file

        # On fail return -1    
        return -1
    
    def extend(self, dst_id=None, src_dir=None, zfill_digits=4):
        # Extend data in dst_id folder to include new images from src_dir
        if dst_id and src_dir:
            formatted_dst = self.naming_formatter(dst_id)
            towalk_dir1 = os.path.join(self.root, formatted_dst)
            #print(towalk_dir1)
            if os.path.isdir(towalk_dir1):
                # get dir # of images
                ls_existing_files = self.rip.parse(towalk_dir1)
                seq = len(ls_existing_files)
                self.rip.clear()
                ls_new_files = self.rip.parse(src_dir)
                cnt_file = 0
                for f in ls_new_files:
                    formatted_seq = str(seq).zfill(zfill_digits)
                    ext = self.get_imgfile_ext(f)
                    newpath = os.path.join(towalk_dir1, '_'.join([formatted_dst, formatted_seq])+'.'+ext)
                    shutil.copy(f, newpath)
                    cnt_file += 1
                    seq += 1

        print(str(cnt_file)+" images extended for "+ dst_id)
        return cnt_file
    
    def delete(self, dst_id=None):
        if dst_id:
            formatted_dst = self.naming_formatter(dst_id)
            towalk_dir1 = os.path.join(self.root, formatted_dst)
            ls_todelete = self.rip.parse(towalk_dir1)
            
            if os.path.isdir(towalk_dir1):
                cnt_file = len(ls_todelete)
                # remove entire folder and all nesting files
                shutil.rmtree(towalk_dir1)

                print(str(cnt_file)+" images deleted for "+ dst_id)
                return cnt_file

        # On fail return -1    
        return -1


if __name__ == '__main__':

    print("Running helper functions...")
    
    ############################################################### 
    # 1. Face extraction using VJ
    ###############################################################
    #rip1 = RecursiveImgParser(root_dir="C:\\Users\\wangs\\Downloads\\lfw_funneled\\")
    #ls_imgpaths = rip1.parse()
    #print(len(ls_imgpaths))
    #vjd1 = VJDetector()

    # # min_h, min_w = 4096, 4096
    # # for imgpath in ls_imgpaths:
    # #     face_name = rip1.name_parser(imgpath)

    # #     cv_img = cv2.imread(imgpath)
    # #     ls_res = vjd1.detect(cv_img)
    # #     #vjd1.display(cv_img, ls_res)
    # #     #vjd1.crop(cv_img, ls_res, save_path="./cropped\\", save_prefix=face_name, save_suffix=".png")
    # #     for (x, y, w, h) in ls_res:
    # #         min_w = w if w < min_w else min_w
    # #         min_h = h if h < min_h else min_h

    # min_h, min_w = 64, 64
    # for imgind, imgpath in tqdm(enumerate(ls_imgpaths), total=len(ls_imgpaths), desc="Processing items"):
    #     face_name = rip1.name_parser(imgpath)
    #     face_name += '_'
    #     face_name += str(imgind)
    #     face_name += '_'

    #     ##################################################################
    #     # VJ detector 
    #     ##################################################################
    #     cv_img = cv2.imread(imgpath)
    #     ls_res = vjd1.detect(cv_img)
    #     #vjd1.display(cv_img, ls_res)

    #     vjd1.crop(cv_img, ls_res, save_path="./cropped64\\", save_prefix=face_name, save_suffix=".jpg", target_size=(min_w, min_h))
    #     #print(f"Index: {imgind}, Item: {imgpath}")


    ###########################################################
    # 2. Attributes analysis using Glasses Detector and Deepface
    ###########################################################
    # ga1 = GlassesAnalyser()
    # da1 = DeepfaceAnalyser()
    # img_path = "C:\\Users\\wangs\\Downloads\\MyFaceswap\\training\\cropped128\\Melissa_Manchester_8843_001.png"
    # dict_ga1 = ga1.analyze(img_path)
    # dict_da1 = da1.analyze(img_path)
    # dict_attr = dict_ga1 | dict_da1
    # print(dict_attr)

    #########################################################
    # 3. Number of Unique Identities
    #########################################################
    # rip1 = RecursiveImgParser(root_dir="C:\\Users\\wangs\\Downloads\\Octoswap\\training\\cropped64\\")
    # ls_imgpaths = rip1.parse()
    # print(ls_imgpaths)
    # n_count = rip1.get_unique_count(ls_imgpaths)
    # print(n_count)

    ########################################################
    # 4. Dataset CRUD ops
    ########################################################
    #du1 = DatasetUpdater(root="C:\\Users\\wangs\\Downloads\\lfw_funned_test\\")
    #du1.create(src_id="zzztest", src_dir="C:\\Users\\wangs\\Downloads\\Octoswap\\test")
    #du1.retrieve(dst_id="Zzztest")
    #du1.update(src_id="Zzztest", dst_id="test outputs")
    #du1.extend(dst_id="test outputs", src_dir="C:\\Users\\wangs\\Downloads\\lfw_funned_test\\Zhang_Ziyi")
    #du1.delete(dst_id="test outputs")