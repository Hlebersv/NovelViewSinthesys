from NovelFaceViewSinthesys.src.pose_utils import camera_matrix, ref_3d_model, ref2d_image_points
import numpy as np
import cv2


# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


def extract_angles(img, predictor, detector):
    img_h, img_w, img_c = img.shape

    faces = detector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
    shape = predictor(img, faces[0])

    face_3d_model = ref_3d_model()
    ref_img_pts = ref2d_image_points()

    focal_length = 1 * img_w
    cam_matrix = camera_matrix(focal_length, (img_h / 2, img_w / 2))
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        face_3d_model, ref_img_pts, cam_matrix, dist_matrix)

    if not success:
        raise "Problem with photo"

    rmat, jac = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    return angles



