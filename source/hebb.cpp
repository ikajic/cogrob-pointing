#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>


using namespace std;
using namespace boost::algorithm;

typedef vector<float> v_f;
typedef v_f::const_iterator v_f_it;


/* 
Functions used for debugging
*/
void debug_info(string info, vector <float> w, int dim){
	cout << info << endl;
	cout << "Number of weights: " << w.size()/dim << endl;
	cout << "Number of dimensions: " << dim << endl;
	cout << endl;
}

void debug_print_vector(v_f vec){
	for(v_f_it it = vec.begin(); it<vec.end(); ++it){
		cout << *it << " ";
	}
	cout << endl;
}

/*
End debug
*/

v_f wsplit(string str, char delim) {
       v_f tokens;
       string buffer;

       for(int i = 0; i < str.size(); i++) {
               if(str[i] == delim && (i == 0 || str[i-1] != '\\')) {
                       tokens.push_back(atof(buffer.c_str()));
                       buffer = "";
               } else {
                       buffer += str[i];
               }
       }
       
       if(buffer != "") {
               tokens.push_back(atof(buffer.c_str()));
       }
       return tokens;
}

v_f read_csv(string name){
	ifstream file ( name.c_str() );
	string value;

	v_f data;
	while ( file.good() )
	{
		 getline ( file, value, '\n' );
		 v_f tokens = wsplit(value, ',');
		
		for(v_f_it i = tokens.begin(); i != tokens.end(); ++i) {
			data.push_back(*i); 
		}
		
	}
	return data;
}

v_f get_row(v_f vec, int idx, int dim){
	v_f res;
	
	v_f_it it = vec.begin() + (idx*dim);
	for(int i = 0; i < dim; ++i){
		res.push_back(*it);
		it++;
	}
	
	return res;
}

float euclid_dist(v_f v1, v_f v2){
	assert(v1.size()==v2.size());
	
	float dist = 0.;
	
	for(int i = 0; i < v1.size(); ++i){
		dist += (v1[i]-v2[i])*(v1[i]-v2[i]);
	}
	
	return sqrt(dist);
}

int get_winning_node(v_f data, v_f weights, int dim){
	int idx = 0;
	float min_dist = 1000.;
	v_f w;
	
	for(int i = 0; i<int(weights.size()/dim); ++i){
		w = get_row(weights, i, dim);
		float dist = euclid_dist(data, w);
		if (dist < min_dist){
			min_dist = dist;
			idx = i; 
		}
	}
	
	return idx;
}

int get_strongest_connection(const int w, v_f hebb, const int n){
	const int offset = w*n; 
	int idx = 0;
	float max_w = -1;
	v_f_it iter = hebb.begin()+offset;
	
	for(int i = 0; i < n; ++i){
		if (*(iter + i) > max_w){
			max_w = *(iter + i);
			idx = i;
		}
	}
	
	return idx;
}

int main(){
	const int nr_rows = 7;
	const int d1 = 3;
	const int d2 = 4;
	bool shw_dbg = false;
	
	// Read weights for the 1st SOM		
	v_f w1(nr_rows*nr_rows*d1);
	w1  = read_csv("som1.csv");
	if (shw_dbg){
		debug_info("\nSOM 1", w1, d1);
		debug_print_vector(w1);
	}
	
	// Read weights for the 2nd SOM		
	v_f w2(nr_rows*nr_rows*d1);
	w2  = read_csv("som2.csv");
	if (shw_dbg){ 
		debug_info("SOM 2", w2, d2);
		debug_print_vector(w2);
	}
	
	// Read Hebbian weights connecting these two
	v_f hebb(nr_rows*nr_rows*nr_rows*nr_rows);	
	hebb = read_csv("hebb.csv");
	if (shw_dbg)
		debug_info("Hebbian weights", hebb, 1);
	
	//// Simulation of motor babbling
	// This code will be replaced by the real-time sensory data from robot	
	v_f d_hands, d_joints;
	d_hands = read_csv("hands.csv");
	d_joints = read_csv("joints.csv");
	
	const int nr_runs = 50000;
	const int offset = 500;
	
	for (int i = 0; i < nr_runs; i+=offset){
		v_f hands = get_row(d_hands, i, d1);
		v_f true_joints = get_row(d_joints, i, d2);
		
		// 1D idx of act. neuron in the first network
		int win1 = get_winning_node(hands, w1, d1); 

		// Let's see how close we get...
		if (shw_dbg) {
			v_f node = get_row(w1, win1, d1);
			cout << "Diff neuron and hand conf:" << euclid_dist(node, hands) << endl;
			debug_print_vector(hands);
			debug_print_vector(node);
		}

		// 1D idx of a neuron in the second network
		int win2 = get_strongest_connection(win1, hebb, w1.size()/d1); 
		
		// Winning neuron's weights
		v_f joints = get_row(w2, win2, d2);
		
		if (shw_dbg) {
			cout << "Err joint conf: " << euclid_dist(true_joints, joints) << endl;
			debug_print_vector(true_joints);
			debug_print_vector(joints);
			cout << endl;
		}
		//move_joints(joints);
	} 
	

    return 0;
}
