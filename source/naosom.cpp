#include <boost/cstdint.hpp>
#include <string.h>
#include <csignal> 

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "../include/debugger.hpp"

#include "../include/std_headers.hpp"
#include "../include/neural_net_headers.hpp"
#include "../include/data_parser.hpp"

#include "configuration.hpp"

using namespace std;
using namespace boost;
using namespace neural_net;
using namespace distance;

typedef double data_type;
typedef vector < data_type > V_d;
typedef vector < V_d > V_v_d;
typedef vector <double> hebb_out;

typedef Cauchy_function < V_d::value_type, V_d::value_type, int32_t > C_a_f;
typedef Gauss_function < V_d::value_type, V_d::value_type, int32_t > G_a_f;
typedef Gauss_function < V_d::value_type, V_d::value_type, int32_t > G_f_space;
typedef Gauss_function < int32_t, V_d::value_type, int32_t > G_f_net;

typedef Euclidean_distance_function < V_d > E_d_t;
typedef Weighted_euclidean_distance_function < V_d, V_d > We_d_t;

typedef Basic_neuron < C_a_f, E_d_t > Kohonen_neuron;
typedef Rectangular_container < Kohonen_neuron > Kohonen_network;

typedef Wta_proportional_training_functional < V_d, double, int32_t > Wta_train_func;
typedef Wta_training_algorithm < 
                    Kohonen_network, V_d, V_v_d::iterator, Wta_train_func > Learning_algorithm;

typedef Hexagonal_topology <int32_t > Hex_top;

typedef const unsigned int cuint;
typedef G_f_space Space_func;
typedef G_f_net Net_func;
typedef Hex_top Net_top;
typedef E_d_t Space_top;
E_d_t e_d; // euclidian distance

typedef Classic_training_weight<V_d, int32_t, Net_func, Space_func, Net_top, Space_top, int32_t> Classic_weight;
typedef Wtm_classical_training_functional <V_d, double, int32_t, int32_t, Classic_weight> Wtm_c_l_f;
typedef Wtm_training_algorithm <Kohonen_network, V_d, V_v_d::iterator, Wtm_c_l_f, int32_t> Wtm_c_training_alg;


void read_input(V_v_d &data, string file_name){
	::data_parser::Data_parser < V_v_d > data_parser;	
	stringstream tmp_stream;
	
	istream *file_ptr = static_cast <istream * > ( 0 );
	file_ptr = new ifstream (file_name.c_str());
	if ( !file_ptr ) {
		cout << "Error in reading file." << endl; 
		exit ( EXIT_FAILURE );
	}

        tmp_stream << file_ptr->rdbuf();
        
        
        delete file_ptr;       
        
        data_parser ( tmp_stream, data );
}

void train_network(Kohonen_network &som, V_v_d &data, int nr_epochs){
		
	// training parameters
	Hex_top hex_top ( som.get_no_rows() );
        Wta_train_func wta_train_func ( 0.2, 0 );
	G_f_space g_f_space ( 100, 1 );
	G_f_net g_f_net ( 10, 1 );
	
	// define if
	Classic_weight classic_weight ( g_f_net, g_f_space, hex_top, e_d );
	Wtm_c_l_f wtm_c_l_f ( classic_weight, 0.3 );
	Wtm_c_training_alg wtm_c_train_alg ( wtm_c_l_f );

	// train SOM for mapping of hand coordinates
	for ( int32_t i = 0; i < nr_epochs; ++i )
	{
		// train network using data
		wtm_c_train_alg ( data.begin(), data.end(), &som);

		// decrease sigma parameter in network will make training proces more sharpen with each epoch,
		// but it have to be done slowly :-)
		wtm_c_train_alg.training_functional.generalized_training_weight.network_function.sigma *= 2.0/3.0;

		// shuffle data
		random_shuffle ( data.begin(), data.end() );
	}
	
}

void Print(const vector<data_type>& v){
  for (unsigned int i=0; i<v.size();i++){
    cout << v[i] << " ";
  }
  cout << endl;
}

void save_weights(string file_name, Kohonen_network &som_hands){
	stringstream output;
	print_network_weights(output, som_hands);
   	
   	ofstream filePtr(file_name.c_str());

   	filePtr << output.rdbuf() << std::flush;
	
   	filePtr.close();
   	
}

void get_activations(Kohonen_network &som, V_d &data, hebb_out &wta){
	
	double max_x=0.0, max_y=0.0, max_A = 0.0;
	
	for (int i=0; i < som.objects.size(); ++i){
		for (int j=0; j < som.objects[0].size(); ++j){
			double act = som.objects[i][j] (data);
			if (act > max_A){
				max_x = i; max_y = j; max_A = act;
				}
		}
	}

	wta.push_back(max_x);
	wta.push_back(max_y);
	wta.push_back(max_A);			
	
}

int main(int argc, char *argv[]){	
	
	if (argc < 1) {
		cout << "Please specify network dimensions as one integers... exiting" << endl;
		return 0;
	}
	
	// network configuration
	const int nr_rows = atoi(argv[1]);
	const int nr_cols = atoi(argv[1]);
	const int32_t nr_epochs = 100; 
	const string data_path = "data/";

	
	Internal_randomize IR;
	G_a_f g_a_f ( 2.0, 1 ); // Activation function
	C_a_f c_a_f ( 2.0, 1 );
	
        V_v_d data_hands, data_joints;  
	read_input(data_hands, data_path + "hands"); 
	read_input(data_joints, data_path + "joints"); 
	
	Kohonen_network som_hands;
	Kohonen_network som_joints;
	
	generate_kohonen_network(nr_rows, nr_cols, c_a_f, e_d, data_hands, som_hands, IR);
	generate_kohonen_network(nr_rows, nr_cols, c_a_f, e_d, data_joints, som_joints, IR);

	
	// print network weights
	cout << "saving SOM weights before training" << endl;
	save_weights(data_path + "w_init_hands", som_hands);
	save_weights(data_path + "w_init_joints", som_joints);
	
	cout << "Training...";
	train_network(som_hands, data_hands,  nr_epochs);
	train_network(som_joints, data_joints,  nr_epochs);
	cout << "done!" << endl;
	
	cout << "saving SOM weights after training" << endl;
	save_weights(data_path + "w_final_hands", som_hands);
	save_weights(data_path + "w_final_joints", som_joints);	


	double hebb_weights[nr_rows][nr_cols][nr_rows][nr_cols];// = {0};

	for(size_t i = 0; i < nr_rows; ++i)
	    for(size_t j = 0; j < nr_cols; ++j)
		for(size_t k = 0; k < nr_rows; ++k)
			for(size_t l = 0; l < nr_cols; ++l)
				hebb_weights[i][j][k][l] = 0;
	   
	double alpha = 0.3;
		
	for (unsigned int idx = 0; idx < data_hands.size(); ++idx)
	{
		hebb_out w1, w2;
		get_activations(som_hands, data_hands[idx], w1);
		get_activations(som_joints, data_joints[idx], w2);
		
		hebb_weights[(int)w1[0]][(int)w1[1]][(int)w2[0]][(int)w2[1]] = alpha * w1[2] * w2[2];
		Print(w1);
		Print(w2);
		//cout << hebb_weights[(int)w1[0]][(int)w1[1]][(int)w2[0]][(int)w2[1]] << endl;

	}
	
	/* prediction */
	/*
		
	cout << "SOM hands after training: " << endl;
	print_network_weights(cout, som_hands);
	cout << endl;
	
	cout << "SOM joints after training: " << endl;
	print_network_weights(cout, som_joints);
	cout << endl;

	cout << "saving SOM weights after training..." << endl;
	save_weights("weights_after_hands", som_hands);
	save_weights("weights_after_joints", som_joints);	
	*/
	return 0;
}	
