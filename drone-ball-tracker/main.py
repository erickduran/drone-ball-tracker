import cv2
import numpy as np
import sys
import time

sys.path.append('../bybop')

from PIL import ImageGrab
from pynput import keyboard

import Bybop_Device
from Bybop_Discovery import Discovery, DeviceID, get_name

from rectangular_area import RectangularArea	

drone = None

pi = 3.141592654
rotation = pi/12

def main():
	print('Searching for devices')

	discovery = Discovery(DeviceID.ALL)
	discovery.wait_for_change()

	devices = discovery.get_devices()

	discovery.stop()

	if not devices:
	    print('Oops ...')
	    sys.exit(1)

	device = next(iter(devices.values()))

	print('Will connect to ' + get_name(device))

	d2c_port = 54321
	controller_type = "PC"
	controller_name = "bybop shell"

	drone = Bybop_Device.create_and_connect(device, d2c_port, controller_type, controller_name)

	if drone is None:
	    print('Unable to connect to a product')
	    sys.exit(1)

	print('Connected to ' + get_name(device))
	drone.start_streaming()
	# end drone connection

	def print_battery():
	    battery = drone.get_battery()
	    print('\n Battery status: ' + str(battery))

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
	# lower_color_bound = np.array([25,40,0])
	# upper_color_bound = np.array([80,130,255])

	# ORANGE BALL
	lower_color_bound = np.array([0,150,30])
	upper_color_bound = np.array([80,255,255])

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

	last_time = 0

	while True:
		screen = np.array(ImageGrab.grab(bbox = (0,100, 1300, 734)))
		screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(screen, dim, interpolation = cv2.INTER_AREA)

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

		area_index = None

		radius = None

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
			area_index = None
			for index, area in enumerate(areas):
				if area.covers(center):
					areas_img[area.point1[1]:area.point2[1], area.point1[0]:area.point2[0]] = area.color
					
					frame = cv2.circle(frame, center, radius, area.color,1)
					frame = cv2.circle(frame, center, 5, area.color, -1)

					areas_img = cv2.circle(areas_img, center, radius, white,1)
					areas_img = cv2.circle(areas_img, center, 5, white, -1)
					area_index = index

					break
			if area_index is None:
				print('no ball')
			else:
				print('ball in area '+ str(area_index))

		cv2.imshow('frame', frame)
		# cv2.imshow('mask', thresh)
		# cv2.imshow('areas', areas_img)

		current_time = time.time()
		k = cv2.waitKey(5) & 0xFF

		update_position = False

		if (current_time - last_time) > 1:
			update_position = True
			last_time = current_time	

		if k == 27:
			drone.land()
			break
		elif k == ord('t'):
			print('t: take off')
			drone.take_off()
		elif k == ord('f') or k == ord('j') or k == ord('h'):
			print('f|j|h: EMERGENCY KEY PRESSED')
			drone.emergency()
		elif k == ord('l'):
			print('l: land')
			drone.land()
		elif k == ord('u') or update_position is True:
			if radius:
				if radius < 10:
					print('go forward')
					drone.move(1, 0, 0, 0)
					time.sleep(0.5)
					drone.move(0, 0, 0, 0)
				elif radius > 30:
					print('go backward')
					drone.move(-1, 0, 0, 0)
					time.sleep(0.5)
					drone.move(0, 0, 0, 0)
				radius = None
			if area_index == 0 or area_index == 1 or area_index == 2:
				print('drone goes down')
				drone.move(0, 0, -1, 0)
				time.sleep(0.5)
				drone.move(0, 0, 0, 0)
			elif area_index == 3:
				print('rotating left')
				drone.move(0, 0, 0, -rotation)
			elif area_index == 4:
				print('ball is in center')
			elif area_index == 5:
				print('rotating right')
				drone.move(0, 0, 0, rotation)
			elif area_index == 6 or area_index == 7 or area_index == 8:
				print('drone goes up')
				drone.move(0, 0, 1, 0)
				time.sleep(0.5)
				drone.move(0, 0, 0, 0)
			else:
				print('ball is out of range')
			update_position = False
		elif k == ord('b'):
			print_battery()

	cv2.destroyAllWindows()
	cap.release()

if __name__ == '__main__':
	main()
