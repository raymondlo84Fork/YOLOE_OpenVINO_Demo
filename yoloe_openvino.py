#install the latest release of Ultralytics
#pip install git+https://github.com/ultralytics/ultralytics.git

from ultralytics import YOLOE
from ultralytics import YOLO
import cv2

model_name="yoloe-11l-seg.pt"
ov_model_name="yoloe-11l-seg_openvino_model"

# Initialize a YOLOE model
model = YOLOE(model_name) 

# Set text prompt
names = ["person", "ping pong paddle", "ball on a table", "blue cup", "table"]
model.set_classes(names, model.get_text_pe(names))

model.export(format="openvino", dynamic=False, half=True)

model_ov = YOLO(ov_model_name)
video_cap = cv2.VideoCapture(1)
#video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
	ret, frame = video_cap.read()

	#can choose between intel:cpu, intel:gpu, or intel:npu
	results = model_ov.predict(frame,conf=0.1, device="intel:gpu")

	# Show results
	frame_out=results[0].plot()
	if not ret:
		break
	cv2.imshow("OpenVINO x YOLO-E Real-Time Seeing Anything", frame_out)
	if cv2.waitKey(1) == ord("q"):
		break

video_cap.release()
cv2.destroyAllWindows()
