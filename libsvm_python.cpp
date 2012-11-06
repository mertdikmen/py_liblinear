#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include "svm.h"

#define INF HUGE_VAL
#define REAL_SMALL_NUMBER 0.0001f

#include <boost/python.hpp>

//#define PY_ARRAY_UNIQUE_SYMBOL tv
#include <numpy/arrayobject.h>

using namespace boost::python;

//SVM INITIALIZATION
struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model_;
struct svm_node *x_space;
int cross_validation;
int nr_fold;

int sample_ct = 0;
  
void initialize_svm()
{
    // default values
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR;
    param.degree = 3;
    param.gamma = 0;	// 1/num_features
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 2000;
    param.C = 1;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    cross_validation = 0;
    sample_ct = 1;
}

class LibSvmPy
{
    std::vector<svm_node*> feature_nodes_vec;
    std::vector<double> labels_vec;
    int feature_dim;

    public:
        LibSvmPy(){
            initialize_svm();
            feature_dim = -1;
        }

    void add_data(boost::python::object& source, int normalize_flag, double label)
    { 
        PyObject* contig = PyArray_FromAny(source.ptr(), PyArray_DescrFromType(PyArray_FLOAT), 
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
        if (param.kernel_type == PRECOMPUTED){
            nnz = data_total;
        }
        else {
            for (int di=0; di<data_total;di++){
                if (data[di] != 0) nnz++;
            }
        }
        std::cout << nnz << "/" << data_total << std::endl;

        int num_nodes = nnz + num_samples; //for end node
 
        if (param.kernel_type == PRECOMPUTED){
            num_nodes += num_samples;  //for index node
        }

        svm_node* all_nodes = (svm_node*) malloc(sizeof(svm_node) * num_nodes);
 
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

            if (param.kernel_type==PRECOMPUTED){
                all_nodes[node_counter].value = sample_ct;
                all_nodes[node_counter].index = 0;
                node_counter++;
                sample_ct++;
            }

            for (int fi=0; fi<fea_dim; fi++){
                float fea_val = data[base_ind+fi];
                if ( (fea_val != 0) || (param.kernel_type==PRECOMPUTED) ){
                    all_nodes[node_counter].value = fea_val / feature_mag;
                    all_nodes[node_counter].index = fi+1;
                    node_counter++;
                }
            }
           
            //terminator
            all_nodes[node_counter].value = 0;
            all_nodes[node_counter].index = -1;
            node_counter++;  
        }
        std::cout << "nodes written: " << node_counter << std::endl;
    }

    void train_svm(const std::string model_file_name, float c_val){

        int total_samples = labels_vec.size();

        prob.x = (svm_node**) malloc( sizeof(svm_node*) * total_samples);
        prob.y = (double*) malloc(sizeof(double) * total_samples);
        prob.l = total_samples;

        for (int i=0; i<total_samples; i++){
            prob.x[i] = feature_nodes_vec[i];
            prob.y[i] = labels_vec[i];
        }

        param.C = c_val;

        model_=svm_train(&prob, &param);

        std::cout << "Saving: " << model_file_name.c_str() << std::endl;

        svm_save_model(model_file_name.c_str(), model_);
    }

    void set_kernel(const int kernel_id){
        param.kernel_type = kernel_id;
    }   

};

    BOOST_PYTHON_MODULE(libsvm_python){
    import_array();
    class_<LibSvmPy>("LibSvmPy", init<>())
        .def("add_data", &LibSvmPy::add_data)
        .def("train_svm", &LibSvmPy::train_svm)
        .def("set_kernel", &LibSvmPy::set_kernel)
    ;
};


