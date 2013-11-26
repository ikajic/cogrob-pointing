import unittest
import numpy as np
import minisom_nao
import pdb
import random

class TestHebb(unittest.TestCase):
	
	def setUp(self):
		path = "../../data/r_37min_74k.dat" 
		data = minisom_nao.read_data(path, 100)
		self.som_hands = minisom_nao.train_som(data['hands'])
		self.som_joints = minisom_nao.train_som(data['joints'])
		self.hebb = minisom_nao.hebbian_learning(self.som_hands, \
			self.som_joints)
		
	def testHebbWeightsBetterThanRandomWeights(self):
		"""
		Test if activated joints in the second map are closer to true hand-joint pairs
		than randomly choosen joints 
		"""
		nr_pts = 10
		nr_runs = 100

		all_hebb_better = []
		for run in xrange(nr_runs):
			mseRand, mseHebb = 0, 0
			for i in xrange(nr_pts):
				idx = random.randint(0, len(self.som_hands.data)-1) 
				hands_view = self.som_hands.data[idx, :]
		
				win_1 = self.som_hands.winner(hands_view)
				win_2 = np.unravel_index(self.hebb[win_1[0], win_1[1], :, :].argmax(), \
					self.som_hands.weights.shape[:2])
		
				joints = self.som_joints.weights[win_2[0], win_2[1], :]
				joints_random = random.choice(random.choice(self.som_joints.weights))
		
				mseHebb += np.linalg.norm(self.som_joints.data[idx, :] - joints)
				mseRand += np.linalg.norm(self.som_joints.data[idx, :] - joints_random)
			all_hebb_better.append(mseHebb < mseRand)
		
		self.assertTrue(all(all_hebb_better))

			
if __name__ == "__main__":
	unittest.main()
