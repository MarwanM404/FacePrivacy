# Importing General libraries
from typing import List, Any

import ultralytics
import supervision as sv
from ultralytics import YOLO
import cv2
import time


def library_check():
    print("supervision.__version__:", sv.__version__)
    ultralytics.checks()


def filter_detection_by_tracker_id(detections, tracker_id_list):
    for id in tracker_id_list:
        detections = detections[detections.tracker_id != id]
    return detections


def face_detection(frame):
    model = YOLO('Yolov8small_trained_ver1.pt')
    unblurred_id_list = []

    # Setting up annotators & tracker
    byte_tracker = sv.ByteTrack()
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=1)
    label_annotator = sv.LabelAnnotator(text_padding=1)
    blur_filter_annotator = sv.BlurAnnotator(kernel_size=30)
    pixelate_annotator = sv.PixelateAnnotator(pixel_size=15)

    result = model(frame)[0]

    # Extracting detections and updating with tracker id
    detections_track = sv.Detections.from_ultralytics(result)
    detections_track = byte_tracker.update_with_detections(detections_track)

    # filtering using confidence
    # detections_track = detections_track[detections_track.confidence > 0.5]

    # setting up Labels format
    labels = [
        f"#{tracker_id} {model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id
        in detections_track
    ]

    # Drawing annotations on the image/frame
    annotated_image_track = label_annotator.annotate(
        scene=frame.copy(),
        detections=detections_track,
        labels=labels
    )

    annotated_image_track = bounding_box_annotator.annotate(
        scene=annotated_image_track,
        detections=detections_track
    )

    if unblurred_id_list:
        blurred_detections_track = filter_detection_by_tracker_id(detections_track,
                                                                  unblurred_id_list)
    else:
        blurred_detections_track = detections_track

    annotated_image_track = pixelate_annotator.annotate(
        scene=annotated_image_track,
        detections=blurred_detections_track
    )

    # annotated_image_track = blur_filter_annotator.annotate(
    #     scene = annotated_image_track,
    #     detections = blurred_detections_track
    # )
    return annotated_image_track


def check_id(file_path):
    """
        :type: file_path: str
        :rtype: list[int]
    """
    file = open(file_path, "r")
    content_list = file.readlines()
    ids_list = []
    for line in content_list:
        # Split the line by whitespace
        numbers = line.split()

        # Iterate through the numbers and convert them to integers
        for num in numbers:
            try:
                num = int(num)
                ids_list.append(num)
            except ValueError:
                pass
    file.close()
    return ids_list


def mouse_pos(event, x, y, flags, param):
    global mouseXY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseXY = [x, y]


def box_selector_with_id(mouseXY, detections):
    bounding_boxs_with_ids = dict(zip(detections.tracker_id.tolist(),
                                      detections.xyxy.tolist()))
    if mouseXY:
        event_x, event_y = mouseXY
        for id, bounding_box in bounding_boxs_with_ids.items():
            x1, y1, x2, y2 = bounding_box  # Unpack the bounding box coordinates
            if x1 <= event_x <= x2 and y1 <= event_y <= y2:
                mouseXY.clear()
                return id




if __name__ == '__main__':

    model = YOLO('Yolov8n_ver2.pt')
    video_path = "test_data/M G ROAD CROWD WALKING _ STOCK FOOTAGES.mp4"
    ################################################################################

    ids_to_unblurr = set()
    mouseXY = []
    frame_counter = 0
    subsampling_rate = 1
    num_frames = 5
    frame_buffer = []

    # Setting up annotators & tracker
    byte_tracker = sv.ByteTrack()
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=1)
    label_annotator = sv.LabelAnnotator(text_padding=1)
    # blur_filter_annotator = sv.BlurAnnotator(kernel_size=30)
    pixelate_annotator = sv.PixelateAnnotator(pixel_size=15)

    cap = cv2.VideoCapture(video_path)  # (0) for webcam

    # FPS counter
    # start_time = time.time()

    while True:
        ret, frame = cap.read()
        # FPS counter calculation
        # elapsed_time = time.time() - start_time
        # fps = 1 / elapsed_time
        #########################

        frame = cv2.resize(frame, (1280, 720))  # (1280, 720)(800, 600)(1920, 1080)

        # if frame_counter % subsampling_rate == 0:
        result = model(frame)[0]
        # results = model.predict(frame, stream=True, show=True, device=0)
        detections = sv.Detections.from_ultralytics(result)
        detections = byte_tracker.update_with_detections(detections)
        labels = [
            f"# {tracker_id} {model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id, _
            in detections
        ]

        id = box_selector_with_id(mouseXY, detections)
        if id in ids_to_unblurr:
            ids_to_unblurr.discard(id)
        else:
            ids_to_unblurr.add(id)

        # Drawing annotations on the image/frame
        annotated_frame = frame.copy()

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )

        if ids_to_unblurr:
            detections = filter_detection_by_tracker_id(detections,
                                                        ids_to_unblurr)
        annotated_frame = pixelate_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )

        # Draw FPS counter
        # cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # # Reset time for next frame
        # start_time = time.time()
        # frame_counter += 1
        cv2.imshow("Yolov8", annotated_frame)
        cv2.setMouseCallback("Yolov8", mouse_pos)

        # Esc to close the window
        if cv2.waitKey(30) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
