""" Test for the Detectron2 library """


import cv2
import tempfile
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from dataclasses import dataclass
from computervision.fileutils import FileOP

__author__ = "Core for Computational Biomedicine at Harvard Medical School"
__copyright__ = "Core for Computational Biomedicine"
__license__ = "CC-BY-4.0"

@dataclass
class Detectron2Config:
    """
    Dataclass to hold and manage configuration settings for a Detectron2 model.
    Attributes:
        config_file: Path to the configuration file for the model.
        weights_file: Path to the weights file for the model.
    """
    config_file: str = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    weights_file: str = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    def get_cfg(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.weights_file)
        cfg.MODEL.DEVICE = 'cpu'
        return cfg

@dataclass
class Detectron2Predictor:
    """
    A class to create a Detectron2 predictor for making predictions on images.
    Attributes:
        cfg: An instance of Detectron2Config containing the configuration for the predictor.
    Methods:
        __post_init__():
            Initializes the predictor using the provided configuration.
        predict(image):
            Makes a prediction on the given image using the initialized predictor.
    """
    cfg: Detectron2Config
    def __post_init__(self):
        self.predictor = DefaultPredictor(self.cfg.get_cfg())
    def predict(self, image):
        return self.predictor(image)

def test_detectron2():
    """
    Test function for the Detectron2 model predictions.
    This test downloads an image, sets up the configuration for Detectron2,
    performs prediction on the image, and asserts that the number of detected
    instances matches the expected value. It prints the path of the downloaded
    image and the prediction results.
    Raises:
        AssertionError: If the number of detected instances does not equal 15.
    """
    image_link = 'http://images.cocodataset.org/val2017/000000439715.jpg'
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = FileOP().download_from_url(url=image_link,
                                                download_dir=temp_dir,
                                                ext_list=['.jpg'])
        print(image_path)
        cfg = Detectron2Config()
        predictor = Detectron2Predictor(cfg)
        im = cv2.imread(image_path)
        outputs = predictor.predict(im)
        results = outputs.get('instances').pred_classes.cpu().numpy()
        print(results)
        assert len(results) == 15