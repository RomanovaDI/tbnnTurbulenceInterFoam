#ifndef addTBNN_H
#define addTBNN_H
#include <iostream>
#include "tensorflow/c/c_api.h"
#include <fstream>
#include <vector>


class Network
{
    public:
      Network();
      ~Network();
      void LoadGraph(std::string &modelPath);
      std::vector<std::string> get_operations() const;
      std::vector<int64_t> get_operation_shape(
        const std::string& operation) const;
      void Run(double* data_1,  double* data_2, std::vector<double> &tbnn_out,
                int input_size_0, int input_size_1,
                std::string input_name_0, std::string input_name_1) const;//std::pair<double*, double*> &data);
      static void Deallocator(void* data, size_t length, void* arg);
    private:
      TF_Session* session;
      TF_Graph* graph;
};
#endif
