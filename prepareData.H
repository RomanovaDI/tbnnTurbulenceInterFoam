auto FillFieldValues = [&](FieldWithInfo &field, size_t field_size) {
            void* data = nullptr;
            std::string name = field.info.name;
            int idx = field.info.idx;
            bool prev_time_step = field.info.prev_time_step;
            assert(idx >= 0);
            assert(idx <= 9);

            if (name == "p_rgh") {
                    data = prev_time_step ? 
                           p_rgh.oldTime().data() : 
                           p_rgh.data();

                    // if (prev_time_step)
                    // {
                    //     Info << "p_rgh oldtime: " << p_rgh.oldTime().data()[0] << endl;
                    // }
                    // else
                    // {
                    //     Info << "p_rgh: " << p_rgh.data()[0] << endl;
                    // }
                }
            else if (name == "U") {
                    data = prev_time_step ? 
                           U.oldTime().component(idx).get()->data() :
                           U.component(idx).get()->data();
                }
            else if (name == "alpha1") {
                    data = prev_time_step ? alpha1.oldTime().data() : alpha1.data();
                }
            else if (name == "gradP") {
                    data = prev_time_step ?
                           gradP.oldTime().component(idx).get()->data() : 
                           gradP.component(idx).get()->data();
                }
            else if (name == "gradU") {
                    data = prev_time_step ? 
                           gradU.oldTime().component(idx).get()->data() : 
                           gradU.component(idx).get()->data();
                }
            else if (name == "gradAW") {
                    data = prev_time_step ? 
                           gradAW.oldTime().component(idx).get()->data() : 
                           gradAW.component(idx).get()->data();
                }
            else if (name == "inv1GradU") {
                    data = inv1GradU.data();
                }
            else if (name == "inv2GradU") {
                    data = inv2GradU.data();
                }
            else if (name == "strainRateTensor") {
                    data = strainRateTensor.component(idx).get()->data();
                }
            else if (name == "rotationRateTensor") {
                    data = rotationRateTensor.component(idx).get()->data();
                }
            else if (name == "I0") {
                    data = I0.data();
                }
            else if (name == "I1") {
                    data = I1.data();
                }
            else if (name == "I2") {
                    data = I2.data();
                }
            else if (name == "I3") {
                    data = I3.data();
                }
            else if (name == "I4") {
                    data = I4.data();
                }
            else if (name == "magGradP") {
                    data = magGradP.data();
                }
            else if (name == "magGradAW") {
                    data = magGradAW.data();
                }

            //my little crutch

            if (data != nullptr)
            {
                memcpy(field.field.data(), data, sizeof(double) * field_size);
                return;
            }

            if (name == "T0") {
                    memcpy(field.field.data(), T0.component(idx).get()->data(),
                           sizeof(double) * field_size);
                }
            else if (name == "T1") {
                    memcpy(field.field.data(), T1.component(idx).get()->data(),
                           sizeof(double) * field_size);
                }
            else if (name == "T2") {
                    memcpy(field.field.data(), T2.component(idx).get()->data(),
                           sizeof(double) * field_size);
                }
            else if (name == "T3") {
                    memcpy(field.field.data(), T3.component(idx).get()->data(),
                           sizeof(double) * field_size);
                }
            else if (name == "T4") {
                    memcpy(field.field.data(), T4.component(idx).get()->data(),
                           sizeof(double) * field_size);
                }
            else if (name == "T5") {
                    memcpy(field.field.data(), T5.component(idx).get()->data(),
                           sizeof(double) * field_size);
                }
            else if (name == "T6") {
                    memcpy(field.field.data(), T6.component(idx).get()->data(),
                           sizeof(double) * field_size);
                }
            else if (name == "T7") {
                    memcpy(field.field.data(), T7.component(idx).get()->data(),
                           sizeof(double) * field_size);
                }
            else if (name == "T8") {
                    memcpy(field.field.data(), T8.component(idx).get()->data(),
                           sizeof(double) * field_size);
                }
            else if (name == "T9") {
                    memcpy(field.field.data(), T9.component(idx).get()->data(),
                           sizeof(double) * field_size);
                }
            else {
                    throw std::runtime_error("Unknown field name");
                }
};

int field_size = U.size();

size_t max_branch_size = 0;
for (size_t i = 0; i < nn_inputs.size(); ++i)
{
    if (nn_inputs[i].branch_size > max_branch_size)
        max_branch_size = nn_inputs[i].branch_size;
}

std::vector<std::vector<InputFields>> input_data(
    field_size, std::vector<InputFields> (nn_inputs.size(), InputFields(max_branch_size)));
std::vector<std::string> inputs_names(nn_inputs.size(), "");

for (size_t j = 0; j < input_data.size(); ++j)
    for (size_t k = 0; k < input_data[j].size(); ++k)
        input_data[j][k].fields.resize(nn_inputs[k].branch_size);

for (size_t i = 0; i < nn_inputs.size(); ++i)
{
    inputs_names[i] = nn_inputs[i].name;

    for (size_t j = 0; j < nn_inputs[i].branch_size; ++j)
    {
        nn_inputs[i].fields[j].field.resize(field_size);
        FillFieldValues(nn_inputs[i].fields[j], field_size);

        for (size_t k = 0; k < nn_inputs[i].fields[j].field.size(); ++k)
            input_data[k][i].fields[j] = nn_inputs[i].fields[j].field[k];

    }
}

// Info << "input_data:" << endl;
// for (size_t i = 0; i < input_data[0].size(); ++i)
// {
//     Info << "branch size: " << input_data[0][i].fields.size() << endl;
//     for (size_t j = 0; j < input_data[0][i].fields.size(); ++j)
//     {
//         Info << input_data[0][i].fields[j] << endl;
//     }   
//     Info << endl; 
// }
















