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

int main()
{
	const int nr_rows = 7;
	const int d_som1 = 3;
	const int d_som2 = 4;
	
	// Read weights for the 1st SOM		
	vector<float> w1(nr_rows*nr_rows*d_som1);
	w1  = read_csv("som1.csv");
	debug_info("SOM 1", w1, d_som1);
	
	// Read weights for the 2nd SOM		
	vector<float> w2(nr_rows*nr_rows*d_som1);
	w2  = read_csv("som2.csv");
	debug_info("SOM 2", w2, d_som2);
	
	// Read Hebbian weights connecting these two
	vector<float> hebb(nr_rows*nr_rows*nr_rows*nr_rows);	
	hebb = read_csv("hebb.csv");
	debug_info("Hebbian weights", hebb, 1);
	
	int idx = 2*7*3+4*3;
	cout << w1[idx] << " " << w1[idx+1] << " " << w1[idx+2] << endl;
	cout << w2[idx] << " " << w2[idx+1] << " " << w2[idx+2] << endl;

    return 0;
}
