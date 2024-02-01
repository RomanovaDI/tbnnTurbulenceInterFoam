#ifndef addNN_H
#define addNN_H

#include <iostream>
#include "tensorflow/c/c_api.h"
#include <fstream>
#include <vector>

struct InputFields
{
    std::vector<double> fields;

    InputFields() = default;
    InputFields(size_t size)
    {
        fields.reserve(size);
    }
};

class NN
{
    public:
      NN();
      ~NN();

      void LoadGraph(std::string &modelPath);

      void Run(std::vector<double> &tbnn_out,
               std::vector<InputFields> &input_data,
               const std::vector<std::string> &inputs_names) const;
      
      static void Deallocator(void* data, size_t length, void* arg);


      // std::vector<std::string> get_operations() const;
      // std::vector<int64_t> get_operation_shape(
      //   const std::string& operation) const;

    private:
      TF_Session* session;
      TF_Graph* graph;
};

#endif
