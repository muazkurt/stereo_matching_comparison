import numpy as np
import cv2
import matplotlib.pyplot as plt 



min_disp = -1

def calc_MSE(res, gt):
	err = 0.0
	for i in range (res.shape[0]):
		for j in range (16 + min_disp, res.shape[1]):
			err += (res[i][j]- gt[i][j]) ** 2
	return err / (res.shape[0] * res.shape[1])

def disparity_orb(left, right):
	orb_detector = cv2.ORB_create()
	BF_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


	kp1, des1 = orb_detector.detectAndCompute(left,None)
	kp2, des2 = orb_detector.detectAndCompute(right,None)
	
	
	matches = BF_matcher.match(des1, des2)
	matches = sorted(matches, key=lambda x: x.distance)
	kp1 = cv2.KeyPoint_convert(kp1)
	kp2 = cv2.KeyPoint_convert(kp2)


	matching_points_left	= []
	matching_points_right	= []
	for match in matches:
		matching_points_left.append((kp2[match.trainIdx][0], kp2[match.trainIdx][1]))
		matching_points_right.append((kp1[match.queryIdx][0], kp1[match.queryIdx][1]))
	output_image_l = np.zeros(left.shape, np.uint8)
	output_image_r = np.zeros(left.shape, np.uint8)
	for temp in range(len(matching_points_right)):
		output_image_l[int(matching_points_left[temp][1])][int(matching_points_left[temp][0])] = np.abs(matching_points_right[temp][0] - matching_points_left[temp][0]) 
		output_image_r[int(matching_points_right[temp][1])][int(matching_points_right[temp][0])] = np.abs(matching_points_right[temp][0] - matching_points_left[temp][0]) 
	output_image_l = cv2.normalize(output_image_l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	output_image_r = cv2.normalize(output_image_r, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	
	return output_image_l + output_image_r


def block_based(left,right):
	stereo = cv2.StereoSGBM_create(	minDisparity=min_disp, 
									numDisparities=16, 
									blockSize = 25)
	disparity = stereo.compute(left, right).astype(np.float32)

	return disparity


def main():
	directories = ["barn1/", "barn2/", "bull/", "poster/", "sawtooth/", "venus/"]
	for dir_name in directories:
		gt_norm	= cv2.imread(dir_name + 'disp2.pgm', 0)
		gt		= gt_norm.astype(np.float32)
		print(dir_name)
		cv2.imshow("Ground Trurth", gt_norm)
		for i in range(0, 7):
			left = cv2.imread(dir_name + 'im' + str(i) + '.ppm', 0)
			right = cv2.imread(dir_name + 'im' + str(i + 1) + '.ppm', 0)
			oput = disparity_orb(left, right)
			cv2.imshow("orb " + str(i) + " and " + str(i + 1), oput)  
			ret = block_based(left, right)
			err = calc_MSE(ret, gt)
			print("Mean square error for image pair " + str(i) + " and " + str(i + 1), " = ", err)

			ret = cv2.normalize(ret, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
			cv2.imshow("disparity of " + str(i) + " and " + str(i + 1), ret)  
			
			error = np.abs(ret - gt)
			error = cv2.normalize(error, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
			cv2.imshow("error image " + str(i) + " and " + str(i + 1), error)  

		cv2.waitKey()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
	cv2.destroyAllWindows()
