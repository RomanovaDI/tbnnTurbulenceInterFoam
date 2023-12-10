#include "addTBNN.h"

Network::Network()
{
    graph = nullptr;
    session = nullptr;
}

void Network::LoadGraph(std::string &modelPath)
{
    graph = TF_NewGraph();
    TF_Status* Status = TF_NewStatus();
    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    TF_Buffer* RunOpts = NULL;
    const char* tags = "serve";
    int ntags = 1;
    char* path = const_cast<char*>(modelPath.c_str());

    session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, path, &tags, ntags, graph, NULL, Status);
    if (TF_GetCode(Status) == TF_OK) 
    {
        printf("Tensorflow 2x Model loaded OK\n");
    }
    else
    {
        printf("%s", TF_Message(Status));
    }

    /*
  std::vector<std::string> operations;
  operations = this->get_operations();

  for (int i = 0; i < operations.size(); ++i)
  {
    if (operations[i] == "NoOp") {
      continue;
    }
    std::vector<int64_t> operation_shape = this->get_operation_shape(operations[i]);
    if (operation_shape.size() > 0 && operation_shape[0] == -1) {
      std::cout << operations[i] << std::endl;
      std::cout << "shape: " << std::endl;
      for (int j = 0; j < operation_shape.size(); ++j) {
        std::cout << operation_shape[j] << " ";
      }
      std::cout << std::endl;
    }
  }
*/

    TF_DeleteSessionOptions(SessionOpts);
    TF_DeleteStatus(Status);
}

std::vector<std::string> Network::get_operations() const
{
    std::vector<std::string> result;
    size_t pos = 0;
    TF_Operation* oper;

    // Iterate through the operations of a graph
    while ((oper = TF_GraphNextOperation(this->graph, &pos)) != nullptr)
        result.emplace_back(TF_OperationName(oper));

    return result;
}

inline bool status_check(TF_Status* status)
{
    if (TF_GetCode(status) != TF_OK)
        throw std::runtime_error(TF_Message(status));

    return true;
}

std::vector<int64_t> Network::get_operation_shape(
        const std::string& operation) const
{
    // Get operation by the name
    TF_Output out_op;
    out_op.oper = TF_GraphOperationByName(this->graph, operation.c_str());
    out_op.index = 0;

    std::vector<int64_t> shape;

    // Operation does not exist
    if (!out_op.oper)
        throw std::runtime_error("No operation named \"" + operation + "\" exists");

    if (operation == "NoOp")
        throw std::runtime_error("NoOp doesn't have a shape");


    // Get number of dimensions
    TF_Status* status = TF_NewStatus();
    int n_dims = TF_GraphGetTensorNumDims(this->graph, out_op,
                                          status);

    // If is not a scalar
    if (n_dims > 0)
    {
        // Get dimensions
        auto* dims = new int64_t[n_dims];
        TF_GraphGetTensorShape(this->graph, out_op, dims, n_dims,
                               status);
        // Check error on Model Status
        status_check(status);
        shape = std::vector<int64_t>(dims, dims + n_dims);
        delete[] dims;
    }

    TF_DeleteStatus(status);
    return shape;
}

void Network::Run(double* data_1,  double* data_2, std::vector<double> &tbnn_out,
                  int input_size_0, int input_size_1,
                  std::string input_name_0, std::string input_name_1) const
{
    ///only for nn with 2 inputs, 1 output (double weights)

    TF_Output input_tensors[2];
    TF_Output output_tensors[1];
    TF_Tensor* input_values[2];
    TF_Tensor* output_values[1];

    //input tensor shape.
    int input_tensor_dims_0 = 2;
    int input_tensor_dims_1 = 2;
    std::int64_t input_dims_0[] = {1, input_size_0};
    std::int64_t input_dims_1[] = {1, input_size_1};
    int num_bytes_in_0 = input_size_0 * sizeof(double);
    int num_bytes_in_1 = input_size_1 * sizeof(double);


    TF_Output t0 = {TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};
    if (t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");

    std::string full_input_name_0 = "serving_default_" + input_name_0;
    TF_Output t1 = {TF_GraphOperationByName(graph, full_input_name_0.c_str()), 0};
    if (t1.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName %s \n", full_input_name_0);

    std::string full_input_name_1 = "serving_default_" + input_name_1;
    TF_Output t2 = {TF_GraphOperationByName(graph, full_input_name_1.c_str()), 0};
    if (t2.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName %s \n", full_input_name_1);


    input_tensors[0] = t1;
    input_tensors[1] = t2;
    input_values[0] = TF_NewTensor(TF_DOUBLE, input_dims_0, input_tensor_dims_0, data_1, num_bytes_in_0, &Deallocator, 0);
    input_values[1] = TF_NewTensor(TF_DOUBLE, input_dims_1, input_tensor_dims_1, data_2, num_bytes_in_1, &Deallocator, 0);

    output_tensors[0] = t0;

    TF_Status* status = TF_NewStatus();
    TF_SessionRun(session, nullptr,
                  &input_tensors[0], &input_values[0], 2,
                  &output_tensors[0], &output_values[0], 1,
                  nullptr, 0, nullptr, status
                  );

    if (TF_GetCode(status) != TF_OK)
    {
        printf("ERROR: SessionRun: %s", TF_Message(status));
        for (int i = 0; i < tbnn_out.size(); ++i)
            tbnn_out[i] = 0;
    }
    else 
    {
        auto output = static_cast<double*>(TF_TensorData(output_values[0]));
        for (int i = 0; i < tbnn_out.size(); ++i)
            tbnn_out[i] = output[i];
    }

    //free memory
    TF_DeleteStatus(status);

    for (auto& t : output_values)
            TF_DeleteTensor(t);

    for (auto& t : input_values)
            TF_DeleteTensor(t);
}

void Network::Deallocator(void* data, size_t length, void* arg)
{
    // printf("Dellocation of the input tensor\n");
}

Network::~Network()
{
    TF_DeleteGraph(graph);
    TF_Status* status = TF_NewStatus();
    TF_CloseSession(session, status);
    if (TF_GetCode(status) != TF_OK) {
            printf("Error close session");
        }
    TF_DeleteSession(session, status);
    TF_DeleteStatus(status);
}


