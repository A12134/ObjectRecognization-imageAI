from imageai.Detection import ObjectDetection
import os
import time

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
start = time.time()
detections, objects_path = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path, "image.jpg"),
    output_image_path=os.path.join(execution_path, "imageNew.jpg"),
    minimum_percentage_probability=50,
    extract_detected_objects=True
)
print(time.time()- start)

for eachObject, eachObjectPath in zip(detections, objects_path):
    print(eachObject["name"], " : ", eachObject["percentage_probability"]," : ",eachObject["box_points"])