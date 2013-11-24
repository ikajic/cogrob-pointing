import unittest
import numpy as np
import minisom_nao
import pdb
import random

from similar_vec import get_similar_vector


class TestSOMs(unittest.TestCase):
	
	def setUp(self):
		path = "../data/34min_65k.dat" 
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
			closest_dps[i, :] = random.choice(self.som_hands.data)
	
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
			closest_dps[i, :] = random.choice(self.som_joints.data)
	
		true = self.som_joints.quantization(closest_dps)
	
		# check if closest neurons are really the closest ones
		est = np.zeros((npoints, self.som_joints.data.shape[1]))
		for i, dp in enumerate(closest_dps):
			v, q = get_similar_data(w, dp, eta=0.2)
			est[i, :] = v[0]

		np.testing.assert_array_almost_equal(est, true, decimal=4)	
		
	def testInactivatedNodesReallyUseless(self):
		"""
		Since a small portion of trained nodes remains inactive in every trial,
		check whether there really aren't any data points to which this inactivated
		nodes are closest
		"""
		i_nodes = self.som_hands.activation_response(self.som_hands.data)
		idx_inact = np.where(i_nodes.flatten()==0)[0]
		_, w = self.som_hands.get_weights()
		
		act = np.setdiff1d(np.arange(w.shape[0]), idx_inact)		
		i_w = w[idx_inact, :]
		a_w = w[act, :]

		for i, dp in enumerate(self.som_hands.data):
			s, _ = get_similar_data(w, dp)
			
			# make sure the closest vector is not in i_w
			# too many for loops here.. ugly!
			_, qi = get_similar_data(i_w, s[0])
			_, qa = get_similar_data(a_w, s[0])
			self.assertGreater(qi[0], qa[0])
				
if __name__ == "__main__":
	unittest.main()
