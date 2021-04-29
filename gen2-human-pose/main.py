import argparse
import threading
import time
from pathlib import Path
from pose import getKeypoints, getValidPairs, getPersonwiseKeypoints
import cv2
import depthai as dai
import numpy as np
from imutils.video import FPS

# usage for 3d visualisation: python main.py -cam -vis

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)")
parser.add_argument('-vis', '--visualizer', action="store_true", help="Use 3d vizualizer")

def setup_camera_params():
    # TODO: Work out how to get this from device in gen2

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
                                                        (456, 256),
                                                        rotation_matrix, translation_matrix)     

    return projection_left, projection_right

args = parser.parse_args()

if not args.camera and not args.video:
    raise RuntimeError("No source selected. Please use either \"-cam\" to use RGB camera as a source or \"-vid <path>\" to run on video")

debug = not args.no_debug

if args.visualizer:
    from visualizer import initialize_OpenGL, get_vector_direction, get_vector_intersection, start_OpenGL
    initialize_OpenGL()

projection_left, projection_right = setup_camera_params()

def cos_dist(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def to_tensor_result(packet):
    return {
        tensor.name: np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
        for tensor in packet.getRaw().tensors
    }

def frame_norm(frame, bbox):
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2,0,1).flatten()

def populatePipeline(p, name):
    cam = p.create(dai.node.MonoCamera)
    socket = dai.CameraBoardSocket.LEFT if name == "left" else dai.CameraBoardSocket.RIGHT
    cam.setBoardSocket(socket)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    cam.setFps(20.0)

    # ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
    pose_manip = p.create(dai.node.ImageManip)
    pose_manip.initialConfig.setResize(456, 256)
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    pose_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(pose_manip.inputImage)

    # NN that detects faces in the image
    pose_nn = p.create(dai.node.NeuralNetwork)
    pose_nn.setBlobPath(str(Path("models/human-pose-estimation-0001_openvino_2021.2_6shave.blob").resolve().absolute()))
    pose_nn.setNumInferenceThreads(1)
    pose_manip.out.link(pose_nn.input)

    # Send mono frames to the host via XLink
    cam_xout = p.create(dai.node.XLinkOut)
    cam_xout.setStreamName("mono_" + name)
    #cam.out.link(cam_xout.input)
    #pose_nn.out.link(cam_xout.input)
    
    
    # Specify that network takes latest arriving frame in non-blocking manner
    pose_nn.input.setQueueSize(1)
    pose_nn.input.setBlocking(False)
    pose_nn_xout = p.createXLinkOut()
    pose_nn_xout.setStreamName("pose_nn_{}".format(name))
    pose_nn.out.link(pose_nn_xout.input)
    pose_nn.passthrough.link(cam_xout.input)


def create_pipeline(use_rgb):
    print("Creating pipeline...")
    pipeline = dai.Pipeline()

    if args.camera:
        if use_rgb:
            # ColorCamera
            print("Creating Color Camera...")
            cam = pipeline.createColorCamera()
            cam.setPreviewSize(456, 256)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setInterleaved(False)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam_xout = pipeline.createXLinkOut()
            cam_xout.setStreamName("cam_out")
            cam.preview.link(cam_xout.input)
            controlIn = pipeline.createXLinkIn()
            controlIn.setStreamName('control')
            controlIn.out.link(cam.inputControl)

    if use_rgb:
        # NeuralNetwork
        print("Creating Human Pose Estimation Neural Network...")
        pose_nn = pipeline.createNeuralNetwork()
        if args.camera:
            pose_nn.setBlobPath(str(Path("models/human-pose-estimation-0001_openvino_2021.2_6shave.blob").resolve().absolute()))
        else:
            pose_nn.setBlobPath(str(Path("models/human-pose-estimation-0001_openvino_2021.2_8shave.blob").resolve().absolute()))
        # Increase threads for detection
        pose_nn.setNumInferenceThreads(2)
        # Specify that network takes latest arriving frame in non-blocking manner
        pose_nn.input.setQueueSize(1)
        pose_nn.input.setBlocking(False)
        pose_nn_xout = pipeline.createXLinkOut()
        pose_nn_xout.setStreamName("pose_nn")
        pose_nn.out.link(pose_nn_xout.input)

        if args.camera:
            cam.preview.link(pose_nn.input)
        else:
            pose_in = pipeline.createXLinkIn()
            pose_in.setStreamName("pose_in")
            pose_in.out.link(pose_nn.input)

    if args.camera and not use_rgb:
        populatePipeline(pipeline, "right")
        populatePipeline(pipeline, "left")

    print("Pipeline created.")
    return pipeline


colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

running = True
pose = None
keypoints_list = {}
detected_keypoints = {}
personwiseKeypoints = {}


class FPSHandler:
    def __init__(self, cap=None):
        self.timestamp = time.time()
        self.start = time.time()
        self.framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None

        self.frame_cnt = 0
        self.ticks = {}
        self.ticks_cnt = {}

    def next_iter(self):
        if not args.camera:
            frame_delay = 1.0 / self.framerate
            delay = (self.timestamp + frame_delay) - time.time()
            if delay > 0:
                time.sleep(delay)
        self.timestamp = time.time()
        self.frame_cnt += 1

    def tick(self, name):
        if name in self.ticks:
            self.ticks_cnt[name] += 1
        else:
            self.ticks[name] = time.time()
            self.ticks_cnt[name] = 0

    def tick_fps(self, name):
        if name in self.ticks:
            return self.ticks_cnt[name] / (time.time() - self.ticks[name])
        else:
            return 0

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)


if args.camera:
    fps = FPSHandler()
else:
    cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
    fps = FPSHandler(cap)


def pose_thread(in_queue, cam, once=False):
    global keypoints_list, detected_keypoints, personwiseKeypoints

    while running:
        try:
            raw_in = in_queue.get()
        except RuntimeError as e:
            print('raw in error: {}'.format(e))
            return
        fps.tick('nn')
        heatmaps = np.array(raw_in.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
        pafs = np.array(raw_in.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
        heatmaps = heatmaps.astype('float32')
        pafs = pafs.astype('float32')
        outputs = np.concatenate((heatmaps, pafs), axis=1)

        new_keypoints = []
        new_keypoints_list = np.zeros((0, 3))
        keypoint_id = 0

        #print('pose thread w,h: {},{}'.format(w,h))

        for row in range(18):
            probMap = outputs[0, row, :, :]
            probMap = cv2.resize(probMap, (456, 256))  # (456, 256)
            keypoints = getKeypoints(probMap, 0.3)
            new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
            keypoints_with_id = []

            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoint_id += 1

            new_keypoints.append(keypoints_with_id)

        valid_pairs, invalid_pairs = getValidPairs(outputs, w, h, new_keypoints)
        newPersonwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, new_keypoints_list)

        detected_keypoints[cam], keypoints_list[cam], personwiseKeypoints[cam] = (new_keypoints, new_keypoints_list, newPersonwiseKeypoints)
        #time.sleep(1)

        if once:
            break

use_rgb = False 
with dai.Device(create_pipeline(use_rgb)) as device:
    print("Starting pipeline...")
    device.startPipeline()
    cam_out = []
    if args.camera:
        if use_rgb:
            cam_out.append(getOutputQueue("cam_out", 1, False))
            controlQueue = device.getInputQueue('control')
        else:
            cam_out.append(device.getOutputQueue("mono_left", 1, False))
            cam_out.append(device.getOutputQueue("mono_right", 1, False))
    else:
        pose_in = device.getInputQueue("pose_in")

    cams = []
    if use_rgb:
        cams = [""]
    else: 
        cams = ["left", "right"]

    pose_queues = []

    for cam in cams:
        if len(cam) > 0:
            cam_suffixed = "_" + cam
        else:
            cam_suffixed = cam

        pose_queues.append(device.getOutputQueue("pose_nn_{}".format(cam), 1, False))    
        
        #pose_nn = device.getOutputQueue("pose_nn{}".format(cam_suffixed), 1, False)
        t = threading.Thread(target=pose_thread, args=(pose_queues[-1], cam, False))
        t.start()

    def triangulate_keypoints():
        global keypoints_list, detected_keypoints, personwiseKeypoints
        geom_points = []
        geom_lines = []

        for i in range(18):
            for j in range(len(detected_keypoints[cam][i])):
                cv2.circle(debug_frame, detected_keypoints[cam][i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)

                try:
                    point_3d = cv2.triangulatePoints(projection_left, projection_right, detected_keypoints['left'][i][j][0:2], detected_keypoints['right'][i][j][0:2])
                    geom_points.append([point_3d[0][0], point_3d[1][0], point_3d[2][0], colors[i][0], colors[i][1], colors[i][2]])
                    # print(detected_keypoints['left'][i][j][0:2])
                    # print(geom_points[-1])
                except Exception as e:
                    pass
                    #print('triangulate error: {}'.format(e))

            for i in range(17):
                for n in range(len(personwiseKeypoints['left'])):
                    points_L = None
                    points_R = None
                    try:
                        index = personwiseKeypoints['left'][n][np.array(POSE_PAIRS[i])]
                        index2 = personwiseKeypoints['right'][n][np.array(POSE_PAIRS[i])]
                        if -1 in index or -1 in index2:
                            continue

                        B = np.int32(keypoints_list['left'][index.astype(int), 0])
                        A = np.int32(keypoints_list['left'][index.astype(int), 1])
                        points_L_1 = np.float32([B[0], A[0]])
                        points_L_2 = np.float32([B[1], A[1]])
                        
                        B = np.int32(keypoints_list['right'][index2.astype(int), 0])
                        A = np.int32(keypoints_list['right'][index2.astype(int), 1])
                        points_R_1 = np.float32([B[0], A[0]])
                        points_R_2 = np.float32([B[1], A[1]])
                        got_points = True
                    except Exception as e:
                        got_points = False
                        #pass # Ignore concurrency issues issues

                    try:
                        if got_points:
                            point_a = cv2.triangulatePoints(projection_left, projection_right, points_L_1, points_R_1)
                            point_b = cv2.triangulatePoints(projection_left, projection_right, points_L_2, points_R_2)

                            #cv2.line(debug_frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
                            geom_lines.append([point_a[0][0], point_a[1][0], point_a[2][0], point_b[0][0], point_b[1][0], point_b[2][0], colors[i][0], colors[i][1], colors[i][2]])
                            #print((B_L[0], B_L[1]),  A_L)
                            #print(geom_lines[-1])
                        #exit(0)
                    except Exception as e:
                        print('triangulate lines error: {}'.format(e))
                        #pass # Need to do some locking or add additional checks


        if args.visualizer:
            start_OpenGL(geom_points, geom_lines)

    def should_run():
        return cap.isOpened() if args.video else True


    def get_frame():
        try:
            if args.video:
                return cap.read()
            else:
                frames = []
                for cam in cam_out:
                    image = cam.get()
                    data, w, h = image.getData(), image.getWidth(), image.getHeight()
                    if w == 640: # This only happens if we manually disable pose_nn in code
                        frames.append(np.array(data).reshape((1, 400, 640)).transpose(1, 2, 0).astype(np.uint8))
                    else:
                        frames.append(np.array(data).reshape((3, 256, 456)).transpose(1, 2, 0).astype(np.uint8))

                return len(frames) > 0, frames

                # if use_rgb:
                #     frames.append(np.array(cam_out[0].get().getData()).reshape((3, 256, 456)).transpose(1, 2, 0).astype(np.uint8))
                # else:
                #     image = cam_out.get()
                #     data, w, h = image.getData(), image.getWidth(), image.getHeight()
                #     frame = np.array(data).reshape((3, 256, 456)).transpose(1, 2, 0).astype(np.uint8)
                #     return True, frame
        except Exception as e:
            print("error: ", e)
            exit(1)


    try:
        while should_run():
            read_correctly, frames = get_frame()

            if not read_correctly:
                break

            frame = frames[0]

            fps.next_iter()
            #h, w = frame.shape[:2]  # 256, 456
            h, w = 256, 456

            for x in range(len(cams)):
                frame = frames[x]
                debug_frame = frame.copy()

                if not args.camera:
                    nn_data = dai.NNData()
                    nn_data.setLayer("input", to_planar(frame, (456, 256)))
                    pose_in.send(nn_data)

                pose_nn = pose_queues[x]
                #pose_thread(pose_nn, cams[x], once=True)

                if 'left' in keypoints_list and 'right' in keypoints_list:
                    triangulate_keypoints()

                if debug:
                    cam = cams[x]
                    if cam in keypoints_list and keypoints_list[cam] is not None and detected_keypoints[cam] is not None and personwiseKeypoints[cam] is not None:
                        for i in range(18):
                            for j in range(len(detected_keypoints[cam][i])):
                                cv2.circle(debug_frame, detected_keypoints[cam][i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
                        for i in range(17):
                            for n in range(len(personwiseKeypoints[cam])):
                                try:
                                    index = personwiseKeypoints[cam][n][np.array(POSE_PAIRS[i])]
                                    if -1 in index:
                                        continue
                                    B = np.int32(keypoints_list[cam][index.astype(int), 0])
                                    A = np.int32(keypoints_list[cam][index.astype(int), 1])
                                    cv2.line(debug_frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
                                except:
                                    pass # Need to do some locking or add additional checks
                    cv2.putText(debug_frame, f"RGB FPS: {round(fps.fps(), 1)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    cv2.putText(debug_frame, f"NN FPS:  {round(fps.tick_fps('nn'), 1)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))


                cv2.imshow(cams[x], debug_frame)

                # for x in range(len(frames)-1):
                #     cv2.imshow("frame2", frames[x+1])

            key = cv2.waitKey(10)
            if key == ord('q'):
                break
                
            elif key == ord('t'):
                print("Autofocus trigger (and disable continuous)")
                ctrl = dai.CameraControl()
                ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
                ctrl.setAutoFocusTrigger()
                controlQueue.send(ctrl)            

    except KeyboardInterrupt:
        pass

    running = False

t.join()
print("FPS: {:.2f}".format(fps.fps()))
if not args.camera:
    cap.release()
