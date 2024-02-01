
bool correct_fields_with_nn = true;

NN nn;

std::vector<Branch> nn_inputs;
Branch nn_output;

std::ifstream nn_f("/home/nargiza/repo_1_fork/tbnnTurbulenceInterFoam/nn_info.json");
try
{
    json nn_info = json::parse(nn_f);

    std::string nn_path = nn_info["nn_path"];
    size_t inputs_num = nn_info["inputs"].size();

    nn_inputs.resize(inputs_num);

    for (size_t i = 0; i < inputs_num; ++i)
        nn_inputs[i] = Branch(nn_info["inputs"][i]["name"],
                              nn_info["inputs"][i]["fields"].size());

    nn_output = Branch("output", nn_info["output"]["fields"].size());

    for (size_t i = 0; i < inputs_num; ++i)
    {
        for (size_t j = 0; j < nn_inputs[i].branch_size; ++j)
        {
            std::string input_field_name = nn_info["inputs"][i]["fields"][j];
            ParseRawFieldName(nn_inputs[i].fields[j].info, input_field_name);
        }
    }

    // for (size_t j = 0; j < nn_output.branch_size; ++j)
    // {
    //     std::string output_field_name = nn_info["output"]["fields"][j];
    //     ParseRawFieldName(nn_output.fields[j].info, output_field_name);
    // }

    nn.LoadGraph(nn_path);
}
catch (const std::exception &e)
{
    std::cerr << "Failed to initialize NN from given configuration file:"
              << e.what() << std::endl;
    correct_fields_with_nn = false;
}
