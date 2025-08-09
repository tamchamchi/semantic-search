from .semantic_extractor import load_semantic_extractor
# import cv2
from PIL import Image
import requests

semantic_extractor = load_semantic_extractor("beit3")
texts = ["The dog is walking", "The car is red"]
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
# image_features = semantic_extractor.extract_image_features(image)
text_features = semantic_extractor.extract_text_features(texts)

# print(image_features.shape)
print(text_features.shape)
print(text_features[0])
print(text_features[1])


# img = cv2.imread("/home/anhndt/ACMM/module/semantic/data/img_test/1234.jpg")
# semantic_extractor_beit3 = load_semantic_extractor("beit3")

# features = semantic_extractor.extract_text_features(texts)
# features_beit3 = semantic_extractor_beit3.extract_text_features(texts)
# Shape 640
# print(features.shape)
# print(features_beit3.shape)

# features_img = semantic_extractor_beit3.extract_image_features(img)
# print(features_img)
