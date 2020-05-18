import numpy as np
import cv2
import matplotlib.pyplot as plt 

for ij in range(8)	:
	for ia in range(3, 20):
		stereo = cv2.StereoBM_create(numDisparities=16 * ij, blockSize=2 * ia + 1)
		print(2 * ia + 1, ij* 16)
		for a in range(1):
			for i in range(0, 1):
				left = cv2.imread('sawtooth/im' + str(i) + '.ppm', 0)
				right = cv2.imread('sawtooth/im' + str(i + 1) + '.ppm', 0)
				'''		
		for j in range(1, 6):
			for i in range(1, 5):
				left	= cv2.imread('ohta/scene1.row' + str(j) + '.col' + str(i) + '.ppm', 0)
				right	= cv2.imread('ohta/scene1.row' + str(j) + '.col' + str(i + 1) + '.ppm', 0)
				'''
				gt		= cv2.imread('sawtooth/disp2.pgm', 0)

				# Initiate orb detector
				orb = cv2.ORB_create()
				# find the keypoints and descriptors with orb
				kp1, des1 = orb.detectAndCompute(left,None)
				kp2, des2 = orb.detectAndCompute(right,None)

				bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
				matches = bf.match(des1, des2)
				matches = sorted(matches, key=lambda x: x.distance)
				kp1 = cv2.KeyPoint_convert(kp1)
				kp2 = cv2.KeyPoint_convert(kp2)


				arr_temp = []
				base_arr = []
				for match in matches:
					arr_temp.append((kp2[match.trainIdx][0], kp2[match.trainIdx][1]))
					base_arr.append((kp1[match.queryIdx][0], kp1[match.queryIdx][1]))

				x, y = left.shape
				output_image = np.zeros((x,y), np.uint8)
						
				fTx = 255
				for temp in range(len(base_arr)):
					output_image[int(base_arr[temp][1])][int(base_arr[temp][0])] = np.abs(base_arr[temp][0] - arr_temp[temp][0]) * 8 
				cv2.imshow('orj', output_image)
				'''
						print(int(base_arr[i][1]), int(base_arr[i][0]), int(arr_temp[i][1]), int(arr_temp[i][0]), (base_arr[i][0] - arr_temp[i][0]) * 8)

					cv2.imshow("res", output_image)

					match_img = cv2.drawKeypoints(left, kp1, None, color=(0,255,0), flags=0)
					#match_img = cv2.drawMatches(left, kp1, right, kp2, matches[:600], None)
					cv2.imshow('Matches1', match_img)
					match_img = cv2.drawKeypoints(right, kp2, None, color=(0,255,0), flags=0)
					cv2.imshow('Matches2', match_img)

					im = left - right
					cv2.imshow("Left-Right", im)

					arr_temp = np.array(arr_temp)
					base_arr = np.array(base_arr)
					h, status = cv2.findHomography(arr_temp, base_arr)  
					if(status[0][0] == 1 and 
						status[1][0] == 1 and 
						status[2][0] == 1 and 
						status[3][0] == 1
						):
						oput = cv2.warpPerspective(right, h, (left.shape[1], left.shape[0]))
						cv2.imshow("output", oput)

					cv2.waitKey()
					cv2.destroyAllWindows()


					cv2.imshow("left", left)
					cv2.imshow("right", right)
				'''
				cv8uc = cv2.normalize(output_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
				cv2.imshow('normalized', cv8uc)

				disparity = stereo.compute(left, right)
				'''
					kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
					disparity = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)
					cv2.imshow("disparitys", disparity)  
				'''
				local_max = disparity.max()
				local_min = disparity.min()
				#disparity_visual = (disparity - local_min) * (1.0 / (local_max - local_min))
				
				disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
				x, y = gt.shape
				x1, y1 = disparity_visual.shape
				#gt = cv2.copyMakeBorder(gt,  y1 - y, 0, x1 - x, 0, cv2.BORDER_CONSTANT, value=(255,255,255))		
				cv2.imshow("disparity", disparity_visual)
				x, y = gt.shape
				error = np.zeros((x,y), np.uint8)
				for i in range(len(error)):
					for j in range(len(error[i])):
						#print(error[i][j], ' = ', gt[i][j], ' - ', disparity_visual[i][j])
						error[i][j] = np.abs(gt[i][j] - disparity[i][j] * 8) % 255


				cv2.imshow("error", error)  		
				'''
				gap = 6

				for y in range(left.shape[0] - gap, gap, -1):
					for x in range(gap, left.shape[1] - gap):
						patch = left[y - gap : y + gap, x - gap: x + gap]
						result = cv2.matchTemplate(right[y - gap: y + gap,], patch, cv2.TM_CCOEFF_NORMED)
						(a, b, _, position) = cv2.minMaxLoc(result)
						#if(b > 0.6):
						output_image[y][x] += (x - position[0]) * 8
					cv8uc = cv2.normalize(output_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
					cv2.imshow('Matches', output_image)
					cv2.imshow('Matches 2', cv8uc)
					cv2.waitKey(1)
				print(i)

				
				x, y = gt.shape
				error = np.zeros((x,y), np.uint8)
				for i in range(len(error)):
					for j in range(len(error[i])):
						#print(error[i][j], ' = ', gt[i][j], ' - ', disparity_visual[i][j])
						error[i][j] = np.abs(gt[i][j] - output_image[i][j]) % 255


				cv2.imshow("error", error)  
				cv2.imshow("gt", gt)
				
				
				'''
				cv2.waitKey(100)



				
