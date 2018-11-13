import cv2
import numpy as np

from rectangular_area import RectangularArea	

def main():
	capture	= cv2.VideoCapture(0)

	width = 600
	height = 337
	dim = (width, height)

	horizontal_divisions = 3
	vertical_divisions = 3

	slice_x = int(width/horizontal_divisions)
	slice_y = int(height/vertical_divisions)

	red = (0,0,255)
	green = (0,255,0)
	blue = (255,0,0)
	white = (255,255,255)

	# GREEN BALL
	lower_color_bound = np.array([25,40,0])
	upper_color_bound = np.array([80,130,255])

	# ORANGE BALL
	# lower_color_bound = np.array([0,150,30])
	# upper_color_bound = np.array([80,255,255])

	hd = horizontal_divisions-1
	vd = vertical_divisions-1

	areas = []
	areas.append(RectangularArea((0, 0), (slice_x, slice_y), red))
	areas.append(RectangularArea((slice_x, 0), (slice_x*hd,slice_y), red))
	areas.append(RectangularArea((slice_x*hd, 0), (width, slice_y), red))
	areas.append(RectangularArea((0, slice_y), (slice_x, slice_y*vd), red))
	areas.append(RectangularArea((slice_x, slice_y), (slice_x*hd, slice_y*vd), green))
	areas.append(RectangularArea((slice_x*hd, slice_y), (width, slice_y*vd), red))
	areas.append(RectangularArea((0, slice_y*vd), (slice_x, height), red))
	areas.append(RectangularArea((slice_x, slice_y*vd), (slice_x*hd, height), red))
	areas.append(RectangularArea((slice_x*hd, slice_y*vd), (width, height), red))

	while True:
		_, image = capture.read()
		frame = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

		frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(frame_hsv, lower_color_bound, upper_color_bound)
		frame_filtered = cv2.bitwise_and(frame, frame, mask = mask)

		median = cv2.medianBlur(frame_filtered, 15)
		thresh = cv2.threshold(median, 1, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

		img, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		frame = cv2.line(frame, (0, slice_y), (width, slice_y), blue, 1)
		frame = cv2.line(frame, (0, slice_y*vd), (width, slice_y*vd), blue, 1)
		frame = cv2.line(frame, (slice_x, 0), (slice_x, height), blue, 1)
		frame = cv2.line(frame, (slice_x*hd, 0), (slice_x*hd, height), blue, 1)

		areas_img = np.zeros((height, width, 3), dtype = np.uint8)

		areas_img = cv2.line(areas_img, (0, slice_y), (width, slice_y), blue, 1)
		areas_img = cv2.line(areas_img, (0, slice_y*vd), (width, slice_y*vd), blue, 1)
		areas_img = cv2.line(areas_img, (slice_x, 0), (slice_x, height), blue, 1)
		areas_img = cv2.line(areas_img, (slice_x*hd, 0), (slice_x*hd, height), blue, 1)

		length = len(contours)

		if length > 0:
			max_contour = contours[0]

			if length > 1:
				max_area = 0

				for contour in contours:
					area = cv2.contourArea(contour)
					if area > max_area:
						max_area = area
						max_contour = contour

			(x, y), radius = cv2.minEnclosingCircle(max_contour)
			center = (int(x), int(y))
			radius = int(radius)

			frame = cv2.circle(frame, center, 5, (0, 0, 255), -1)

			for area in areas:
				if area.covers(center):
					areas_img[area.point1[1]:area.point2[1], area.point1[0]:area.point2[0]] = area.color

					frame = cv2.circle(frame, center, radius, area.color,1)
					frame = cv2.circle(frame, center, 5, area.color, -1)

					areas_img = cv2.circle(areas_img, center, radius, white,1)
					areas_img = cv2.circle(areas_img, center, 5, white, -1)

					break

		cv2.imshow('frame', frame)
		cv2.imshow('mask', thresh)
		cv2.imshow('areas', areas_img)

		k = cv2.waitKey(5) & 0xFF

		if k == 27:
			break

	cv2.destroyAllWindows()
	cap.release()

if __name__ == '__main__':
	main()
