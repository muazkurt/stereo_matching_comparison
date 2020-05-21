import numpy as np
import cv2
import matplotlib.pyplot as plt 
SWS = 5
PFS = 5
PFC = 21
MDS = -25
NOD = 16
TTH = 50
UR = 10
SR = 15
SPWS = 100


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
	for match in matches[:20]:
		matching_points_left.append((kp2[match.trainIdx][0], kp2[match.trainIdx][1]))
		matching_points_right.append((kp1[match.queryIdx][0], kp1[match.queryIdx][1]))
	'''
		matching_points_left = np.array(matching_points_left)
		matching_points_right = np.array(matching_points_right)
		h, status = cv2.findHomography(matching_points_left, matching_points_right)  
		if(status[0][0] == 1 and 
			status[1][0] == 1 and 
			status[2][0] == 1 and 
			status[3][0] == 1
			):
			oput = cv2.warpPerspective(right, h, (left.shape[1], left.shape[0]))
			cv2.imshow("output", oput)
		cv2.imshow("L", left)
		cv2.imshow("r", right)
	'''
	output_image_l = np.zeros(left.shape, np.uint8)
	output_image_r = np.zeros(left.shape, np.uint8)
	for temp in range(len(matching_points_right)):
		output_image_l[int(matching_points_left[temp][1])][int(matching_points_left[temp][0])] = np.abs(matching_points_right[temp][0] - matching_points_left[temp][0]) 
		output_image_r[int(matching_points_right[temp][1])][int(matching_points_right[temp][0])] = np.abs(matching_points_right[temp][0] - matching_points_left[temp][0]) 
	output_image_l = cv2.normalize(output_image_l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	output_image_r = cv2.normalize(output_image_r, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	
	return output_image_l, output_image_r


def block_based(left,right):
	min_disp = -5
	win_size = 3
	max_disp = 63
	num_disp = max_disp - min_disp
	stereo = cv2.StereoSGBM_create(	minDisparity=min_disp, 
									numDisparities=16, 
									blockSize = 11,
									P1=8 * 3 * win_size ** 2, 
									P2=32 * 3 * win_size ** 2, 
									disp12MaxDiff=1, 
									uniquenessRatio=5, 
									speckleWindowSize=5,
									speckleRange=5
									
									)
	disparity = stereo.compute(left, right).astype(np.float32)
	'''
	disparity *= 8
	gap = 7
	
	disparity = np.zeros(left.shape, np.uint8)
	for y in range(left.shape[0] - gap, gap, -1):
		for x in range(gap, left.shape[1] - gap):
			patch = left[y - gap : y + gap, x - gap: x + gap]
			result = cv2.matchTemplate(right[y - gap: y + gap, ], patch, cv2.TM_CCOEFF_NORMED)
			(_, b, _, position) = cv2.minMaxLoc(result)
			if(b > 0.3):
				disparity[y][x] += (x - position[0]) * 8
		cv2.imshow('Matches', disparity)
		cv2.waitKey(1)
	'''
	'''

		print(i)
		x, y = gt.shape
		error = np.zeros((x,y), np.uint8)
		for i in range(len(error)):
			for j in range(len(error[i])):
				#print(error[i][j], ' = ', gt[i][j], ' - ', disparity_visual[i][j])
				error[i][j] = np.abs(gt[i][j] - output_image[i][j]) % 255


		cv2.imshow("error", error)  
		cv2.imshow("gt", gt)
	
	local_max = disparity.max()
	local_min = disparity.min()
	#disparity_visual = (disparity - local_min) * (1.0 / (local_max - local_min))

	'''

	return disparity


def main():
	directories = ["barn1/", "barn2/", "bull/", "poster/", "sawtooth/", "venus/"]
	for dir_name in directories:
		gt		= cv2.imread(dir_name + 'disp2.pgm', 0).astype(np.float32)
		gt2		= cv2.imread(dir_name + 'disp6.pgm', 0)
		cv2.imshow("gt1", gt)
		cv2.imshow("gt2", gt2)
		print(dir_name)
		print(gt.min(), gt.max())
		ret = np.zeros(gt.shape)
		for i in range(0, 1):
			left = cv2.imread(dir_name + 'im' + str(i) + '.ppm', 0)
			right = cv2.imread(dir_name + 'im' + str(i + 1) + '.ppm', 0)
			oput = disparity_orb(left, right)
			ret = block_based(left, right)
			error = ret - gt
			err = 0
			for i in range (ret.shape[0]):
				for j in range (ret.shape[1]):
					err += (ret[i][j]- gt[i][j]) ** 2
					error[i][j] = ret[i][j]- gt[i][j]
			print(err / (ret.shape[0] * ret.shape[1]))
			'''
			error = np.zeros(gt.shape, np.uint8)
			'''

			error = cv2.normalize(error, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
			cv2.imshow("error1", error)  
			
			print(ret.min(), ret.max())
			ret = cv2.normalize(ret, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
			print(ret.min(), ret.max())
			cv2.imshow("avg_disp", ret)  
			
			error = np.abs(ret - gt)
			cv2.imshow("error2", error)  
			'''
			for i in range (ret.shape[0]):
				for j in range (ret.shape[1]):
					error[i][j] = np.abs(ret[i][j] - gt[i][j])
			'''
			error = cv2.normalize(error, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
			cv2.imshow("error3", error)  
			

			cv2.waitKey()
			continue


if __name__ == "__main__":
	main()
	cv2.destroyAllWindows()
