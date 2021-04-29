# import math
from pathlib import Path
import numpy as np
import math
from visualizer import initialize_OpenGL, get_vector_direction, get_vector_intersection, start_OpenGL
import cv2
import depthai as dai

p = dai.Pipeline()

# TODO: Work out how to get this from device in gen2
left_camera_position = (0.107, -0.038, 0.008)
right_camera_position = (0.109, 0.039, 0.008)
cameras = (left_camera_position, right_camera_position)

rotation_matrix = np.array([[0.999956,   -0.008674,   -0.003590],
    [0.008698,    0.999939,    0.006867],
    [0.003530,   -0.006898,    0.999970]])
translation_matrix = np.array([   -7.523639,
   -0.034132,
    0.048544,])
intrinsic_left = np.array([[847.716797,    0.000000,  628.660217],
    [0.000000,  848.163635,  402.847534],
    [0.000000,    0.000000,    1.000000]])
intrinsic_right = np.array([[851.803955,    0.000000,  634.575623],
    [0.000000,  852.288208,  411.274628],
    [0.000000,    0.000000,    1.000000,]])
distortion_left = np.array([-3.567049,   11.420240,    0.000314,   -0.000814,   -8.152797,   -3.640617,   11.683157,
   -8.406696,    0.000000,    0.000000,    0.000000,    0.000000,    0.000000,    0.000000,])
distortion_right = np.array([-5.211617,   18.090021,   -0.000381,    0.000292,  -18.919079,   -5.265417,   18.290663,
  -19.104439,    0.000000,    0.000000,    0.000000,    0.000000,    0.000000,    0.000000,])

R1, R2, projection_left, projection_right, Q, roi1, roi2 = cv2.stereoRectify(intrinsic_left, distortion_left,
                                                      intrinsic_right, distortion_right,
                                                      (1280,720),
                                                      rotation_matrix, translation_matrix)     



print('stereoRectifyResults')
print('R1: {}'.format(R1))
print('R2: {}'.format(R2))
print('projection_left: {}'.format(projection_left))
print('projection_right: {}'.format(projection_right))
print('Q: {}'.format(Q))
print('roi1: {}'.format(roi1))
print('roi2: {}'.format(roi2))
#exit(3)

def populatePipeline(p, name):
    cam = p.create(dai.node.MonoCamera)
    socket = dai.CameraBoardSocket.LEFT if name == "left" else dai.CameraBoardSocket.RIGHT
    cam.setBoardSocket(socket)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    # ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
    face_manip = p.create(dai.node.ImageManip)
    face_manip.initialConfig.setResize(300, 300)
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    face_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(face_manip.inputImage)

    # NN that detects faces in the image
    face_nn = p.create(dai.node.NeuralNetwork)
    face_nn.setBlobPath(str(Path("models/face-detection-retail-0004_2021.3_6shaves.blob").resolve().absolute()))
    face_manip.out.link(face_nn.input)

    # Send mono frames to the host via XLink
    cam_xout = p.create(dai.node.XLinkOut)
    cam_xout.setStreamName("mono_" + name)
    face_nn.passthrough.link(cam_xout.input)

    # Script node will take the output from the NN as an input, get the first bounding box
    # and if the confidence is greater than 0.2, script will send ImageManipConfig to the manip_crop
    image_manip_script = p.create(dai.node.Script)
    image_manip_script.inputs['nn_in'].setBlocking(False)
    image_manip_script.inputs['nn_in'].setQueueSize(1)
    face_nn.out.link(image_manip_script.inputs['nn_in'])
    image_manip_script.setScriptData("""
while True:
    nn_in = node.io['nn_in'].get()
    nn_data = nn_in.getFirstLayerFp16()

    conf=nn_data[2]
    if 0.2<conf:
        x_min=nn_data[3]
        y_min=nn_data[4]
        x_max=nn_data[5]
        y_max=nn_data[6]
        cfg = ImageManipConfig()
        cfg.setCropRect(x_min, y_min, x_max, y_max)
        cfg.setResize(48, 48)
        cfg.setKeepAspectRatio(False)
        node.io['to_manip'].send(cfg)
        #node.warn(f"1 from nn_in: {x_min}, {y_min}, {x_max}, {y_max}")
""")

    # This ImageManip will crop the mono frame based on the NN detections. Resulting image will be the cropped
    # face that was detected by the face-detection NN.
    manip_crop = p.create(dai.node.ImageManip)
    face_nn.passthrough.link(manip_crop.inputImage)
    image_manip_script.outputs['to_manip'].link(manip_crop.inputConfig)
    manip_crop.initialConfig.setResize(48, 48)
    manip_crop.setWaitForConfigInput(False)

    # Send ImageManipConfig to host so it can visualize the landmarks
    config_xout = p.create(dai.node.XLinkOut)
    config_xout.setStreamName("config_" + name)
    image_manip_script.outputs['to_manip'].link(config_xout.input)

    crop_xout = p.createXLinkOut()
    crop_xout.setStreamName("crop_" + name)
    manip_crop.out.link(crop_xout.input)

    # Second NN that detcts landmarks from the cropped 48x48 face
    landmarks_nn = p.createNeuralNetwork()
    landmarks_nn.setBlobPath(str(Path("models/landmarks-regression-retail-0009_2021.3_6shaves.blob").resolve().absolute()))
    manip_crop.out.link(landmarks_nn.input)

    landmarks_nn_xout = p.createXLinkOut()
    landmarks_nn_xout.setStreamName("landmarks_" + name)
    landmarks_nn.out.link(landmarks_nn_xout.input)


populatePipeline(p, "right")
populatePipeline(p, "left")

def get_landmark_cv2tri(landmark_left, landmark_right):
    # print('landmark L: {}'.format(landmark_left))
    # print('landmark R: {}'.format(landmark_right))

    points = []

    for n in range(len(landmark_left)):
        point_3d = cv2.triangulatePoints(projection_left, projection_right, landmark_left[n-1], landmark_right[n-1])
        points.append([point_3d[0][0], point_3d[1][0], point_3d[2][0]])
        # print('point_3d: {}'.format(point_3d))

    return points

def get_landmark_3d(landmark):
    focal_length = 842
    landmark_norm = 0.5 - np.array(landmark)

    # image size
    landmark_image_coord = landmark_norm * 640

    landmark_spherical_coord = [math.atan2(landmark_image_coord[0], focal_length),
                                -math.atan2(landmark_image_coord[1], focal_length) + math.pi / 2]

    landmarks_3D = [
        math.sin(landmark_spherical_coord[1]) * math.cos(landmark_spherical_coord[0]),
        math.sin(landmark_spherical_coord[1]) * math.sin(landmark_spherical_coord[0]),
        math.cos(landmark_spherical_coord[1])
    ]

    return landmarks_3D

initialize_OpenGL()

# Pipeline is defined, now we can connect to the device
with dai.Device(p) as device:
    # Set device log level - to see logs from the Script node
    device.setLogLevel(dai.LogLevel.WARN)
    device.setLogOutputLevel(dai.LogLevel.WARN)

    # Start pipeline
    device.startPipeline()
    queues = []
    for name in ["left", "right"]:
        queues.append(device.getOutputQueue(name="mono_"+name, maxSize=4, blocking=False))
        queues.append(device.getOutputQueue(name="crop_"+name, maxSize=4, blocking=False))
        queues.append(device.getOutputQueue(name="landmarks_"+name, maxSize=4, blocking=False))
        queues.append(device.getOutputQueue(name="config_"+name, maxSize=4, blocking=False))
    while True:
        lr_landmarks = []
        landmarks_map = {}
        for i in range(2):
            name = "left" if i == 1 else "right"
            # 300x300 Mono image frame
            inMono = queues[i*4].get()
            frame = inMono.getCvFrame()

            # Cropped+streched (48x48) mono image frame
            inCrop = queues[i*4 + 1].get()
            cropped_frame = inCrop.getCvFrame()

            inConfig = queues[i*4 + 3].tryGet()
            if inConfig is not None:
                xmin = int(300 * inConfig.getCropXMin())
                ymin = int(300 * inConfig.getCropYMin())
                xmax = int(300 * inConfig.getCropXMax())
                ymax = int(300 * inConfig.getCropYMax())
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                width = inConfig.getCropXMax()-inConfig.getCropXMin()
                height = inConfig.getCropYMax()-inConfig.getCropYMin()

                # Facial landmarks from the second NN
                inLandmarks = queues[i*4 + 2].get()
                landmarks_layer = inLandmarks.getFirstLayerFp16()
                landmarks = np.array(landmarks_layer).reshape(5, 2)

                lr_landmarks.append(list(map(get_landmark_3d, landmarks)))
                landmarks_map[name] = []
                for n in range(len(landmarks)):
                    landmarks_map[name].append(landmarks[n-1] * 1000)

                for landmark in landmarks:
                    cv2.circle(cropped_frame, (int(48*landmark[0]), int(48*landmark[1])), 3, (0, 255, 0))
                    w = landmark[0] * width + inConfig.getCropXMin()
                    h = landmark[1] * height + inConfig.getCropYMin()
                    cv2.circle(frame, (int(w * 300), int(h * 300)), 3, (0,255,0))

            # Display both mono/cropped frames
            cv2.imshow("mono_"+name, frame)
            cv2.imshow("crop_"+name, cropped_frame)

        cv2_tri_points = []

        if "left" in landmarks_map and len(landmarks_map["left"]) > 0 and "right" in landmarks_map and len(landmarks_map["right"]) > 0:
            cv2_tri_points = get_landmark_cv2tri(landmarks_map["left"], landmarks_map["right"])

        # 3D visualization
        if len(lr_landmarks) == 2 and len(lr_landmarks[0]) > 0 and len(lr_landmarks[1]) > 0:
                mid_intersects = []
                for i in range(5):
                    left_vector = get_vector_direction(left_camera_position, lr_landmarks[0][i])
                    right_vector = get_vector_direction(right_camera_position, lr_landmarks[1][i])
                    intersection_landmark = get_vector_intersection(left_vector, left_camera_position, right_vector,
                                                                    right_camera_position)
                    mid_intersects.append(intersection_landmark)

                start_OpenGL(mid_intersects, cameras, lr_landmarks[0], lr_landmarks[1], cv2_tri_points)

        if cv2.waitKey(1) == ord('q'):
            break
