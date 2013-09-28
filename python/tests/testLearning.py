import unittest
import numpy as np
import minisom_nao
import pdb

from similar_vec import get_similar_data
from random import choice

class TestSOMs(unittest.TestCase):
	
	def setUp(self):
		path = "/home/ivana/babbling_KB_left_arm.dat" 
		data = minisom_nao.read_data(path)
		self.som_hands = minisom_nao.train_som(data['hands'][:10])
		self.som_joints = minisom_nao.train_som(data['joints'][:10])
	
	def testWeightRanges(self):
		"""
		Make sure all weights are within 0 and 1 range if normalization minmax
		is used
		"""
		
		self.assertTrue(self.som_hands.norm_type == 'minmax')
		
		self.assertTrue(np.all(np.logical_and(self.som_hands.weights > 0.0, \
			self.som_hands.weights < 1.0)))
			
		self.assertTrue(np.all(np.logical_and(self.som_joints.weights > 0.0, \
			self.som_joints.weights < 1.0)))		
	
	def testDataPointAssignmentHands(self):
		"""
		Pick randomly 10 points from hand coordinates and check if the closest 
		nodes in SOM are indeed activated
		"""
		
		npoints = 10
		_, w = self.som_hands.get_weights()
	
		closest_dps = np.zeros((npoints, self.som_hands.data.shape[1]))
		for i in xrange(npoints):
			closest_dps[i, :] = choice(self.som_hands.data)
	
		true = self.som_hands.quantization(closest_dps)
	
		# check if closest neurons are really the closest ones
		est = np.zeros((npoints, self.som_hands.data.shape[1]))
		for i, dp in enumerate(closest_dps):
			v, q = get_similar_data(w, dp, eta=0.2)
			est[i, :] = v[0]

		np.testing.assert_array_almost_equal(est, true, decimal=4)
		
	def testDataPointAssignmentJoints(self):
		"""
		Pick randomly 10 points from joint coordinates and check if the closest 
		nodes in SOM are indeed activated
		"""	
		npoints = 10
		_, w = self.som_joints.get_weights()
	
		closest_dps = np.zeros((npoints, self.som_joints.data.shape[1]))
		for i in xrange(npoints):
			closest_dps[i, :] = choice(self.som_joints.data)
	
		true = self.som_joints.quantization(closest_dps)
	
		# check if closest neurons are really the closest ones
		est = np.zeros((npoints, self.som_joints.data.shape[1]))
		for i, dp in enumerate(closest_dps):
			v, q = get_similar_data(w, dp, eta=0.2)
			est[i, :] = v[0]

		np.testing.assert_array_almost_equal(est, true, decimal=4)	
		
if __name__ == "__main__":
	unittest.main()
