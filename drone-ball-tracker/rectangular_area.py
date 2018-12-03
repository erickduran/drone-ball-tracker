class RectangularArea(object):
	def __init__(self, point1, point2, color):
		self.point1 = point1
		self.point2 = point2
		self.color = color

	def covers(self, point):
		if point[0] >= self.point1[0] and point[0] < self.point2[0]:
			if point[1] >= self.point1[1] and point[1] < self.point2[1]:
				return True
		return False