import numpy as np

class NuMixing:
	def __init__(self, theta_23, theta_13, theta_12, delta_cp, Delta_m221, Delta_m231):
		c23 = np.cos(theta_23)
		s23 = np.sin(theta_23)
		c13 = np.cos(theta_13)
		s13 = np.sin(theta_13)
		c12 = np.cos(theta_12)
		s12 = np.sin(theta_12)
		dm221 = Delta_m221
		dm231 = Delta_m231
		dcp = delta_cp

		R12 = self.R12(c12, s12)
		R23 = self.R23(c23, s23)
		R13d = self.R13d(c13, s13, dcp)

		d12_R12 = self.d12_R12(c12, s12)
		d23_R23 = self.d23_R23(c23, s23)
		d13_R13d = self.d13_R13d(c13, s13, dcp)
		dd_R13d = self.dd_R13d(c13, s13, dcp)

		self.U = R23 @ R13d @ R12
		self.d23_U = d23_R23 @ R13d @ R12
		self.d12_U = R23 @ R13d @ d12_R12
		self.d13_U = R23 @ d13_R13d @ R12
		self.dd_U = R23 @ dd_R13d @ R12

		self.print_matrix("PMNS", self.U)
		self.print_matrix("Derivative of PMNS w.r.t. theta_23", self.d23_U)
		self.print_matrix("Derivative of PMNS w.r.t. theta_12", self.d12_U)
		self.print_matrix("Derivative of PMNS w.r.t. theta_13", self.d13_U)
		self.print_matrix("Derivative of PMNS w.r.t. delta_cp", self.dd_U)

	def R12(self, c12, s12):
		return np.array([
			[c12, s12, 0],
			[-s12, c12, 0],
			[0, 0, 1]
			])

	def d12_R12(self, c12, s12):
		return np.array([
			[-s12, c12, 0],
			[-c12, -s12, 0],
			[0, 0, 0]
			])

	def R23(self, c23, s23):
		return np.array([
			[1, 0, 0],
			[0, c23, s23],
			[0, -s23, c23],
			])

	def d23_R23(self, c23, s23):
		return np.array([
			[0, 0, 0],
			[0, -s23, c23],
			[0, -c23, -s23],
			])

	def R13d(self, c13, s13, dcp):
		return np.array([
			[c13, 0, s13*np.exp(-dcp*1j)],
			[0, 1, 0],
			[-s13*np.exp(dcp*1j), 0, c13],
			])

	def dd_R13d(self, c13, s13, dcp):
		return np.array([
			[0, 0, -1j*s13*np.exp(-dcp*1j)],
			[0, 0, 0],
			[-1j*s13*np.exp(dcp*1j), 0, 0],
			])

	def d13_R13d(self, c13, s13, dcp):
		return np.array([
			[-s13, 0, c13*np.exp(-dcp*1j)],
			[0, 0, 0],
			[-c13*np.exp(dcp*1j), 0, -s13],
			])

	def print_matrix(self, name, matrix):
		print(f"Printing {name} matrix:")
		print(np.around(matrix,decimals=1))


if __name__ == "__main__":
	theta_23 = 0.86 # mixing angle in radians
	theta_12 = 0.58 # mixing angle in radians
	theta_13 = 0.15 # mixing angle in radians
	delta_cp = 4.1 # cp phase in radians
	Delta_m221 = 1e-5
	Delta_m231 = 1e-3
	nu_mixing = NuMixing(theta_23, theta_13, theta_12, delta_cp, Delta_m221, Delta_m231)