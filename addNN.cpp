#include "addNN.h"

NN::NN()
{
    graph = nullptr;
    session = nullptr;
}

void NN::LoadGraph(std::string &modelPath)
{
    graph = TF_NewGraph();
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    TF_Buffer* run_opts = NULL;
    const char* tags = "serve";
    int ntags = 1;
    char* path = const_cast<char*>(modelPath.c_str());

    session = TF_LoadSessionFromSavedModel(session_opts,
                                           run_opts,
                                           path,
                                           &tags,
                                           ntags,
                                           graph,
                                           NULL,
                                           status);
    if (TF_GetCode(status) == TF_OK) 
    {
        printf("Tensorflow 2x Model loaded OK\n");
    }
    else
    {
        printf("%s", TF_Message(status));
        throw std::runtime_error("Failed to load Tensorflow 2x saved_model");
    }

    TF_DeleteSessionOptions(session_opts);
    TF_DeleteStatus(status);
}

void NN::Run(std::vector<double> &tbnn_out,
             std::vector<InputFields> &input_data,
             const std::vector<std::string> &inputs_names) const
{
    size_t branches_num = input_data.size();

    const int input_tensor_dims = 2;

    TF_Output t_out = {TF_GraphOperationByName(graph, "StatefulPartitionedCall"),
                      0};
    if (t_out.oper == NULL)
        throw std::runtime_error(
            "Failed TF_GraphOperationByName: StatefulPartitionedCall");

    std::vector<TF_Output> input_tensors(branches_num);
    std::vector<TF_Output> output_tensors(1);
    std::vector<TF_Tensor*> input_values(branches_num, nullptr);
    std::vector<TF_Tensor*> output_values(1, nullptr);

    output_tensors[0] = t_out;

    for (size_t i = 0; i < branches_num; ++i)
    {
        std::int64_t input_dim[] = {1, input_data[i].fields.size()};
        int bytes_num = input_dim[1] * sizeof(double);

        std::string full_branch_name = "serving_default_" + inputs_names[i];
        TF_Output t_input = 
             {TF_GraphOperationByName(graph, full_branch_name.c_str()), 0};
        if (t_input.oper == NULL)
            throw std::runtime_error("Failed TF_GraphOperationByName: " +
             full_branch_name);

        double* data = input_data[i].fields.data();

        input_tensors[i] = t_input;
        input_values[i] = TF_NewTensor(TF_DOUBLE,
                                       input_dim,
                                       input_tensor_dims,
                                       data,
                                       bytes_num,
                                       &Deallocator,
                                       0); 
    }

    TF_Status* status = TF_NewStatus();
    TF_SessionRun(session, nullptr,
                  &input_tensors[0],
                  &input_values[0],
                  branches_num,
                  &output_tensors[0],
                  &output_values[0],
                  1,
                  nullptr,
                  0,
                  nullptr,
                  status
                  );

    if (TF_GetCode(status) != TF_OK)
    {
        for (size_t i = 0; i < tbnn_out.size(); ++i)
            tbnn_out[i] = 0;

        std::string message = "SessionRun: " + std::string(TF_Message(status));
        throw std::runtime_error(message.c_str());
    }
    else 
    {
        auto output = static_cast<double*>(TF_TensorData(output_values.front()));
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

void NN::Deallocator(void* data, size_t length, void* arg)
{
    // printf("Dellocation of the input tensor\n");
}

NN::~NN()
{
    TF_DeleteGraph(graph);
    TF_Status* status = TF_NewStatus();
    TF_CloseSession(session, status);
    if (TF_GetCode(status) != TF_OK) {
            printf("Error while session closing");
        }
    TF_DeleteSession(session, status);
    TF_DeleteStatus(status);
}

/*std::vector<std::string> NN::get_operations() const
{
    std::vector<std::string> result;
    size_t pos = 0;
    TF_Operation* oper;

    // Iterate through the operations of a graph
    while ((oper = TF_GraphNextOperation(this->graph, &pos)) != nullptr)
        result.emplace_back(TF_OperationName(oper));

    return result;
}
*/

/*std::vector<int64_t> NN::get_operation_shape(
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
        if (TF_GetCode(status) != TF_OK)
            throw std::runtime_error(TF_Message(status));
        shape = std::vector<int64_t>(dims, dims + n_dims);
        delete[] dims;
    }

    TF_DeleteStatus(status);
    return shape;
}
*/
