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

vector<float> wsplit(string str, char delim) {
       vector<float> tokens;
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

vector<float> read_csv(string name){
	ifstream file ( name.c_str() );
	string value;

	vector<float> weights;
	while ( file.good() )
	{
		 getline ( file, value, '\n' );
		 vector<float> tokens = wsplit(value, ',');
		
		for(vector<float>::const_iterator i = tokens.begin(); i != tokens.end(); ++i) {
			weights.push_back(*i); 
		}
		
	}
	return weights;
}

void debug_info(string info, vector <float> w, int dim){
	cout << info << endl;
	cout << "Number of weights: " << w.size()/dim << endl;
	cout << "Number of dimensions: " << dim << endl;
}

vector<float> get_hands(vector<float> d_hands, int idx, int dim){
	vector<float> res;
	
	vector<float>::const_iterator it = d_hands.begin() + (idx*dim);
	for(int i = 0; i < dim; i++){
		res.push_back(*it);
		it++;
	}
	
	return res;
}

int main()
{
	const int nr_rows = 7;
	const int d_som1 = 3;
	const int d_som2 = 4;
	
	typedef vector<float> v_f;
	
	// Read weights for the 1st SOM		
	v_f w1(nr_rows*nr_rows*d_som1);
	w1  = read_csv("som1.csv");
	debug_info("SOM 1", w1, d_som1);
	
	// Read weights for the 2nd SOM		
	v_f w2(nr_rows*nr_rows*d_som1);
	w2  = read_csv("som2.csv");
	debug_info("SOM 2", w2, d_som2);
	
	// Read Hebbian weights connecting these two
	v_f hebb(nr_rows*nr_rows*nr_rows*nr_rows);	
	hebb = read_csv("hebb.csv");
	debug_info("Hebbian weights", hebb, 1);
	
	//// Simulation of motor babbling
	// This code will be replaced by the real-time sensory data from robot	
	v_f d_hands, d_joints;
	d_hands = read_csv("hands.csv");
	d_joints = read_csv("joints.csv");
	
	int nr_runs = 10;
	for (int i = 0; i < nr_runs; ++i){
		v_f hands = get_hands(d_hands, i, d_som1);
		cout << hands[0] << " "<<  hands[1] << " "<<hands[2] << endl;
		//int winner1 =  activate_node(hands, w1); // 1D coordinate of activated neuron in SOM 1
		//int winner2 = find_strongest(winner1, hebb); // 1D coordinate of of activated neuron in SOM 2
		//v_f joints = w2[w1*nr_rows*d_som2+w2*d_som2]
		//move_joints(joints);
	} 
	
	cout << d_hands[1] << endl;
	int idx = 2*7*3+4*3;
	cout << w1[idx] << " " << w1[idx+1] << " " << w1[idx+2] << endl;
	cout << w2[idx] << " " << w2[idx+1] << " " << w2[idx+2] << endl;

    return 0;
}
