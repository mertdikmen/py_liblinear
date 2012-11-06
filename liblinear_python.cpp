#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include "linear.h"

#define INF HUGE_VAL
#define REAL_SMALL_NUMBER 0.0001f

#include <boost/python.hpp>

//#define PY_ARRAY_UNIQUE_SYMBOL tv
#include <numpy/arrayobject.h>

using namespace boost::python;

float bias_val = 10.0f;

//SVM INITIALIZATION
struct feature_node *x_space;
struct parameter param;
struct problem prob;
struct model* model_;
int flag_cross_validation ;
int nr_fold;

void initialize_svm()
{
//    param.solver_type = L2R_L2LOSS_SVC_DUAL;
    param.solver_type = L2R_L1LOSS_SVC_DUAL;
    param.C = 1;
    param.eps = INF; // see setting below
    
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;

    flag_cross_validation = 0;

    if(param.eps == INF)
    {
        if(param.solver_type == L2R_LR ||
           param.solver_type == L2R_L2LOSS_SVC)
            param.eps = 0.0001;
        else if(param.solver_type == L2R_L2LOSS_SVC_DUAL ||
                param.solver_type == L2R_L1LOSS_SVC_DUAL ||
                param.solver_type == MCSVM_CS ||
                param.solver_type == L2R_LR_DUAL)
            param.eps = 0.0001;
        else if(param.solver_type == L1R_L2LOSS_SVC ||
                param.solver_type == L1R_LR)
            param.eps = 0.0001;
    }
}

class LibLinearPy
{
    std::vector<feature_node*> feature_nodes_vec;
    std::vector<feature_node*> allocation_heads;

    std::vector<int> labels_vec;
    int feature_dim;

    public:
        LibLinearPy(){
            initialize_svm();
            feature_dim = -1;
        }

        void set_bias(const float bias){
            bias_val = bias;
        }   

        void set_weights(const double weight1, const double weight2)
        {
            param.nr_weight = 2;
		    param.weight_label = (int *) realloc(param.weight_label,
                                                 sizeof(int)*param.nr_weight);
			param.weight = (double *) realloc(param.weight,
                                              sizeof(double)*param.nr_weight);
			param.weight_label[0] = 1;
            param.weight_label[1] =-1;

            param.weight[0] = weight1;
            param.weight[1] = weight2;

        }   

        void set_solver(int solver_type){
            param.solver_type = solver_type;
            std::cout << "solver set: " << solver_type << std::endl;
        }
   
        void add_data(boost::python::object& source, int normalize_flag, int label)
        { 
            PyObject* contig = PyArray_FromAny(source.ptr(), 
                                               PyArray_DescrFromType(PyArray_FLOAT),
                                               1, 3, NPY_CARRAY, NULL);
            handle<> temp(contig);
            boost::python::object array;
            array = object(temp);

            float* data = (float*) PyArray_DATA(array.ptr());

            const int num_samples = (int) PyArray_DIM(array.ptr(),0);
            const int fea_dim = (int) PyArray_DIM(array.ptr(),1);

            if ((feature_dim != -1) && (feature_dim != fea_dim) ){
                std::cout << "Feature dimensionality mismatch" << std::endl;
            }
            feature_dim = fea_dim;

            std::cout << num_samples << "x" << fea_dim << std::endl;

            int nnz = 0;
            int data_total = num_samples*fea_dim;
            for (int di=0; data_total--; di++){
                if (data[di] != 0) nnz++;
            }

            std::cout << nnz << "/" << num_samples*fea_dim << std::endl;

            int num_nodes = nnz + 2*num_samples;

            feature_node* all_nodes = (feature_node*) malloc(sizeof(feature_node) *
                                               num_nodes);

            allocation_heads.push_back(all_nodes);

            int node_counter = 0;
            for (int si=0;si<num_samples;si++){
                feature_nodes_vec.push_back(all_nodes + node_counter);
                labels_vec.push_back(label);

                int base_ind = si * fea_dim;
                float feature_mag = 1.0f;

                if (normalize_flag){
                    feature_mag = 0.0f;
                    for (int fi=0; fi<fea_dim; fi++){
                        feature_mag += data[base_ind+fi] * data[base_ind+fi];
                    }
                    feature_mag = sqrt(feature_mag);                    
                    if (feature_mag == 0){
                        feature_mag = REAL_SMALL_NUMBER;
                    }
                }

                for (int fi=0; fi<fea_dim; fi++){
                    float fea_val = data[base_ind+fi];
                    if (fea_val != 0){
                        all_nodes[node_counter].value = fea_val / feature_mag;
                        all_nodes[node_counter].index = fi+1;
                        node_counter++;
                    }
                }
                
                if (bias_val > 0){
                    //bias
                    all_nodes[node_counter].value = bias_val;
                    all_nodes[node_counter].index = fea_dim+1;
                    node_counter++;
                }
                
                //terminator
                all_nodes[node_counter].value = 0;
                all_nodes[node_counter].index = -1;
                node_counter++;  
            }
        }

        void train_svm(const std::string model_file_name, float c_val){

            int total_samples = labels_vec.size();

            prob.x = (feature_node**) malloc( sizeof(feature_node*) * total_samples);
            prob.y = (double*) malloc(sizeof(double) * total_samples);
            std::cout << "Bias: " << bias_val << std::endl;
            prob.bias = bias_val;
            prob.l = total_samples;
            prob.n = feature_dim;
            if (bias_val > 0) prob.n++;

            for (int i=0; i<total_samples; i++){
                prob.x[i] = feature_nodes_vec[i];
                prob.y[i] = labels_vec[i];
            }

            param.C = c_val;

            model_=train(&prob, &param);

            std::cout << "Saving: " << model_file_name.c_str() << std::endl;

            save_model(model_file_name.c_str(), model_);

            for (int i = 0; i<allocation_heads.size(); i++){
                free(allocation_heads[i]);
            }
            free(prob.x);
            free(prob.y);
        }
} ;

BOOST_PYTHON_MODULE(liblinear_python){
    import_array();
    class_<LibLinearPy>("LibLinearPy", init<>())
        .def("add_data", &LibLinearPy::add_data)
        .def("set_solver", &LibLinearPy::set_solver)
        .def("set_weights", &LibLinearPy::set_weights)
        .def("set_bias", &LibLinearPy::set_bias)
        .def("train_svm", &LibLinearPy::train_svm)
    ;
}
