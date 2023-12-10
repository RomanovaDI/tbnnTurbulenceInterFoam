std::ifstream nn_f("/home/nargiza/repo_1_fork/tbnnTurbulenceInterFoam/nn_info.json");
json nn_info = json::parse(nn_f);

std::string nn_path = nn_info["nn_path"];
int inputs_num = nn_info["inputs"].size();
std::vector<std::string> tbnn_inputs_names(inputs_num, "");

for (int i = 0; i < tbnn_inputs_names.size(); ++i)
    tbnn_inputs_names[i] = nn_info["inputs"][i]["name"];

std::vector<InputFieldInfo> inputs_0_info(nn_info["inputs"][0]["fields"].size());
std::vector<InputFieldInfo> inputs_1_info(nn_info["inputs"][1]["fields"].size());
std::vector<InputFieldInfo> output_info(nn_info["outputs"][0]["fields"].size());

for (int i = 0; i < inputs_0_info.size(); ++i)
{
    std::string in_field_info = nn_info["inputs"][0]["fields"][i];
    InputFieldInfo info;
    ParseString(info, in_field_info);
    inputs_0_info[i] = info;
}

for (int i = 0; i < inputs_1_info.size(); ++i)
{
    std::string in_field_info = nn_info["inputs"][1]["fields"][i];
    InputFieldInfo info;
    ParseString(info, in_field_info);
    inputs_1_info[i] = info;
}

Network tbnn;
tbnn.LoadGraph(nn_path);
