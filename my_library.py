
#importing packages
from scipy.spatial import distance as dist
import numpy as np
import cv2


#calculating head pose angle
def get_head_pose(coords):
    #3D face points
	point_3D_17 = np.float32([6.825897, 6.760612, 4.402142])
	point_3D_21 = np.float32([1.330353, 7.122144, 6.903745])
	point_3D_22 = np.float32([-1.330353, 7.122144, 6.903745])
	point_3D_26 = np.float32([-6.825897, 6.760612, 4.402142])
	point_3D_36 = np.float32([5.311432, 5.485328, 3.987654]) 
	point_3D_39 = np.float32([1.789930, 5.393625, 4.413414]) 
	point_3D_42 = np.float32([-1.789930, 5.393625, 4.413414])
	point_3D_45 = np.float32([-5.311432, 5.485328, 3.987654])
	point_3D_31 = np.float32([2.005628, 1.409845, 6.165652]) 
	point_3D_35 = np.float32([-2.005628, 1.409845, 6.165652])
	point_3D_48 = np.float32([2.774015, -2.080775, 5.048531])
	point_3D_54 = np.float32([-2.774015, -2.080775, 5.048531])
	point_3D_57 = np.float32([0.000000, -3.116408, 6.097667]) 
	point_3D_8 = np.float32([0.000000, -7.415691, 4.070434]) 

	object_points = np.float32([point_3D_17,
								point_3D_21,
                               	point_3D_22,
                               	point_3D_26,
                               	point_3D_36,
                               	point_3D_39,
                               	point_3D_42,
                               	point_3D_45,
                               	point_3D_31,
                               	point_3D_35,
                               	point_3D_48,
                               	point_3D_54,
                               	point_3D_57,
                               	point_3D_8])

    #3D projection points source
	reprojection_source = np.float32([[10.0, 10.0, 10.0],
                       				  [10.0, 10.0, -10.0],
                       				  [10.0, -10.0, -10.0],
                       				  [10.0, -10.0, 10.0],
                       				  [-10.0, 10.0, 10.0],
                       				  [-10.0, 10.0, -10.0],
                       				  [-10.0, -10.0, -10.0],
                       				  [-10.0, -10.0, 10.0]])

    #2D face points
	image_points = np.float32([coords[17],
							   coords[21], 
							   coords[22], 
							   coords[26], 
							   coords[36],
							   coords[39], 
							   coords[42], 
							   coords[45], 
							   coords[31], 
							   coords[35],
							   coords[48], 
							   coords[54], 
							   coords[57], 
							   coords[8]])

	K =  [742.11486816, 0.0,            331.6961939,
		  0.0,          726.3464355,    212.95805438,
		  0.0,          0.0,            1.0         ]

    #camera specific camera matrix 
	camera_matrix = np.array(K).reshape(3, 3).astype(np.float32)

    #camera specific distortion coefficients
	distortion_coeffs = np.zeros((5,1), np.float32)

    #solving PnP problem
	_, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coeffs)

    #calculating projection points on 2D image from 3D points source
	reprojected_points, _ = cv2.projectPoints(reprojection_source, rotation_vector, translation_vector, camera_matrix, distortion_coeffs)
	reprojected_points = reprojected_points.reshape(8, 2)

	#calculating eular angle
	rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
	pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
	_, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_matrix)
	return reprojected_points, euler_angle


#adjust size of the frame maintaining its height width ratio
def resize(img, height=None, width=None):
	h, w, _ = img.shape
	if height is None and width is None:
		return img
	if height is None:
		t = w / float(width)
		dim = (width, int(h / t))
	else:
		t = h / float(height)
		dim = (int(w / t), height)
	new_img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
	return new_img


#calculating EAR
def eye_aspect_ratio(eye):
	a = dist.euclidean(eye[1], eye[5])
	b = dist.euclidean(eye[2], eye[4])
	c = dist.euclidean(eye[0], eye[3])
	ear = (a + b) / (2.0 * c)
	return ear


#calculating MAR
def mouth_aspect_ratio(mouth):
	a = dist.euclidean(mouth[1], mouth[7])
	b = dist.euclidean(mouth[2], mouth[6])
	c = dist.euclidean(mouth[3], mouth[5])
	d = dist.euclidean(mouth[0], mouth[4])
	mar = (a + b + c) / (3.0 * d)
	return mar

	
#drawing face mask
def mask(coords, frame, x, y):
    cv2.line(frame, (coords[0,0] + x, coords[0,1] + y), (coords[17,0] + x, coords[17,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[17,0] + x, coords[17,1] + y), (coords[18,0] + x, coords[18,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[18,0] + x, coords[18,1] + y), (coords[19,0] + x, coords[19,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[19,0] + x, coords[19,1] + y), (coords[20,0] + x, coords[20,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[20,0] + x, coords[20,1] + y), (coords[21,0] + x, coords[21,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[21,0] + x, coords[21,1] + y), (coords[27,0] + x, coords[27,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[27,0] + x, coords[27,1] + y), (coords[22,0] + x, coords[22,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[22,0] + x, coords[22,1] + y), (coords[23,0] + x, coords[23,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[23,0] + x, coords[23,1] + y), (coords[24,0] + x, coords[24,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[24,0] + x, coords[24,1] + y), (coords[25,0] + x, coords[25,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[25,0] + x, coords[25,1] + y), (coords[26,0] + x, coords[26,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[0,0] + x, coords[0,1] + y), (coords[1,0] + x, coords[1,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[1,0] + x, coords[1,1] + y), (coords[2,0] + x, coords[2,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[2,0] + x, coords[2,1] + y), (coords[3,0] + x, coords[3,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[3,0] + x, coords[3,1] + y), (coords[4,0] + x, coords[4,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[4,0] + x, coords[4,1] + y), (coords[5,0] + x, coords[5,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[5,0] + x, coords[5,1] + y), (coords[6,0] + x, coords[6,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[6,0] + x, coords[6,1] + y), (coords[7,0] + x, coords[7,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[7,0] + x, coords[7,1] + y), (coords[8,0] + x, coords[8,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[8,0] + x, coords[8,1] + y), (coords[9,0] + x, coords[9,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[9,0] + x, coords[9,1] + y), (coords[10,0] + x, coords[10,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[10,0] + x, coords[10,1] + y), (coords[11,0] + x, coords[11,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[11,0] + x, coords[11,1] + y), (coords[12,0] + x, coords[12,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[12,0] + x, coords[12,1] + y), (coords[13,0] + x, coords[13,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[13,0] + x, coords[13,1] + y), (coords[14,0] + x, coords[14,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[14,0] + x, coords[14,1] + y), (coords[15,0] + x, coords[15,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[15,0] + x, coords[15,1] + y), (coords[16,0] + x, coords[16,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[16,0] + x, coords[16,1] + y), (coords[26,0] + x, coords[26,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[19,0] + x, coords[19,1] + y), (coords[24,0] + x, coords[24,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[24,0] + x, coords[24,1] + y), (coords[20,0] + x, coords[20,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[20,0] + x, coords[20,1] + y), (coords[23,0] + x, coords[23,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[23,0] + x, coords[23,1] + y), (coords[21,0] + x, coords[21,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[21,0] + x, coords[21,1] + y), (coords[22,0] + x, coords[22,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[21,0] + x, coords[21,1] + y), (coords[39,0] + x, coords[39,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[39,0] + x, coords[39,1] + y), (coords[20,0] + x, coords[20,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[20,0] + x, coords[20,1] + y), (coords[38,0] + x, coords[38,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[38,0] + x, coords[38,1] + y), (coords[19,0] + x, coords[19,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[19,0] + x, coords[19,1] + y), (coords[37,0] + x, coords[37,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[37,0] + x, coords[37,1] + y), (coords[18,0] + x, coords[18,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[18,0] + x, coords[18,1] + y), (coords[36,0] + x, coords[36,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[36,0] + x, coords[36,1] + y), (coords[17,0] + x, coords[17,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[0,0] + x, coords[0,1] + y), (coords[36,0] + x, coords[36,1] + y), (0,255,255), 2, 8)	
    cv2.line(frame, (coords[39,0] + x, coords[39,1] + y), (coords[27,0] + x, coords[27,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[27,0] + x, coords[27,1] + y), (coords[42,0] + x, coords[42,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[42,0] + x, coords[42,1] + y), (coords[22,0] + x, coords[22,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[42,0] + x, coords[42,1] + y), (coords[23,0] + x, coords[23,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[23,0] + x, coords[23,1] + y), (coords[43,0] + x, coords[43,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[43,0] + x, coords[43,1] + y), (coords[24,0] + x, coords[24,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[24,0] + x, coords[24,1] + y), (coords[44,0] + x, coords[44,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[44,0] + x, coords[44,1] + y), (coords[25,0] + x, coords[25,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[25,0] + x, coords[25,1] + y), (coords[45,0] + x, coords[45,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[45,0] + x, coords[45,1] + y), (coords[26,0] + x, coords[26,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[45,0] + x, coords[45,1] + y), (coords[16,0] + x, coords[16,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[27,0] + x, coords[27,1] + y), (coords[28,0] + x, coords[28,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[28,0] + x, coords[28,1] + y), (coords[29,0] + x, coords[29,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[29,0] + x, coords[29,1] + y), (coords[30,0] + x, coords[30,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[30,0] + x, coords[30,1] + y), (coords[33,0] + x, coords[33,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[31,0] + x, coords[31,1] + y), (coords[32,0] + x, coords[32,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[32,0] + x, coords[32,1] + y), (coords[33,0] + x, coords[33,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[33,0] + x, coords[33,1] + y), (coords[34,0] + x, coords[34,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[34,0] + x, coords[34,1] + y), (coords[35,0] + x, coords[35,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[31,0] + x, coords[31,1] + y), (coords[30,0] + x, coords[30,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[31,0] + x, coords[31,1] + y), (coords[29,0] + x, coords[29,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[32,0] + x, coords[32,1] + y), (coords[30,0] + x, coords[30,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[34,0] + x, coords[34,1] + y), (coords[30,0] + x, coords[30,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[35,0] + x, coords[35,1] + y), (coords[30,0] + x, coords[30,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[35,0] + x, coords[35,1] + y), (coords[29,0] + x, coords[29,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[48,0] + x, coords[48,1] + y), (coords[49,0] + x, coords[49,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[49,0] + x, coords[49,1] + y), (coords[50,0] + x, coords[50,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[50,0] + x, coords[50,1] + y), (coords[51,0] + x, coords[51,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[51,0] + x, coords[51,1] + y), (coords[52,0] + x, coords[52,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[52,0] + x, coords[52,1] + y), (coords[53,0] + x, coords[53,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[53,0] + x, coords[53,1] + y), (coords[54,0] + x, coords[54,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[54,0] + x, coords[54,1] + y), (coords[55,0] + x, coords[55,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[55,0] + x, coords[55,1] + y), (coords[56,0] + x, coords[56,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[56,0] + x, coords[56,1] + y), (coords[57,0] + x, coords[57,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[57,0] + x, coords[57,1] + y), (coords[58,0] + x, coords[58,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[58,0] + x, coords[58,1] + y), (coords[59,0] + x, coords[59,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[59,0] + x, coords[59,1] + y), (coords[48,0] + x, coords[48,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[4,0] + x, coords[4,1] + y), (coords[48,0] + x, coords[48,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[48,0] + x, coords[48,1] + y), (coords[5,0] + x, coords[5,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[5,0] + x, coords[5,1] + y), (coords[59,0] + x, coords[59,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[59,0] + x, coords[59,1] + y), (coords[6,0] + x, coords[6,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[6,0] + x, coords[6,1] + y), (coords[58,0] + x, coords[58,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[58,0] + x, coords[58,1] + y), (coords[7,0] + x, coords[7,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[7,0] + x, coords[7,1] + y), (coords[57,0] + x, coords[57,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[57,0] + x, coords[57,1] + y), (coords[8,0] + x, coords[8,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[57,0] + x, coords[57,1] + y), (coords[9,0] + x, coords[9,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[9,0] + x, coords[9,1] + y), (coords[56,0] + x, coords[56,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[56,0] + x, coords[56,1] + y), (coords[10,0] + x, coords[10,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[10,0] + x, coords[10,1] + y), (coords[55,0] + x, coords[55,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[55,0] + x, coords[55,1] + y), (coords[11,0] + x, coords[11,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[11,0] + x, coords[11,1] + y), (coords[54,0] + x, coords[54,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[54,0] + x, coords[54,1] + y), (coords[12,0] + x, coords[12,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[48,0] + x, coords[48,1] + y), (coords[31,0] + x, coords[31,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[31,0] + x, coords[31,1] + y), (coords[49,0] + x, coords[49,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[49,0] + x, coords[49,1] + y), (coords[32,0] + x, coords[32,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[32,0] + x, coords[32,1] + y), (coords[50,0] + x, coords[50,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[50,0] + x, coords[50,1] + y), (coords[33,0] + x, coords[33,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[33,0] + x, coords[33,1] + y), (coords[51,0] + x, coords[51,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[33,0] + x, coords[33,1] + y), (coords[52,0] + x, coords[52,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[52,0] + x, coords[52,1] + y), (coords[34,0] + x, coords[34,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[34,0] + x, coords[34,1] + y), (coords[53,0] + x, coords[53,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[53,0] + x, coords[53,1] + y), (coords[35,0] + x, coords[35,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[35,0] + x, coords[35,1] + y), (coords[54,0] + x, coords[54,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[48,0] + x, coords[48,1] + y), (coords[3,0] + x, coords[3,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[3,0] + x, coords[3,1] + y), (coords[31,0] + x, coords[31,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[31,0] + x, coords[31,1] + y), (coords[2,0] + x, coords[2,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[2,0] + x, coords[2,1] + y), (coords[41,0] + x, coords[41,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[41,0] + x, coords[41,1] + y), (coords[1,0] + x, coords[1,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[1,0] + x, coords[1,1] + y), (coords[36,0] + x, coords[36,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[41,0] + x, coords[41,1] + y), (coords[31,0] + x, coords[31,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[31,0] + x, coords[31,1] + y), (coords[40,0] + x, coords[40,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[40,0] + x, coords[40,1] + y), (coords[29,0] + x, coords[29,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[40,0] + x, coords[40,1] + y), (coords[28,0] + x, coords[28,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[28,0] + x, coords[28,1] + y), (coords[39,0] + x, coords[39,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[54,0] + x, coords[54,1] + y), (coords[13,0] + x, coords[13,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[13,0] + x, coords[13,1] + y), (coords[35,0] + x, coords[35,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[35,0] + x, coords[35,1] + y), (coords[14,0] + x, coords[14,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[14,0] + x, coords[14,1] + y), (coords[46,0] + x, coords[46,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[46,0] + x, coords[46,1] + y), (coords[15,0] + x, coords[15,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[15,0] + x, coords[15,1] + y), (coords[45,0] + x, coords[45,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[46,0] + x, coords[46,1] + y), (coords[35,0] + x, coords[35,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[35,0] + x, coords[35,1] + y), (coords[47,0] + x, coords[47,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[47,0] + x, coords[47,1] + y), (coords[29,0] + x, coords[29,1] + y), (0,255,255), 2, 8)
    cv2.line(frame, (coords[42,0] + x, coords[42,1] + y), (coords[28,0] + x, coords[28,1] + y), (0,255,255), 2, 8)
