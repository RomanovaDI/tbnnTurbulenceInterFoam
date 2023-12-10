int U_size = U.size();
std::vector<std::vector<double>> inputs_0_values(inputs_0_info.size(), std::vector<double> (U_size, 0));
std::vector<std::vector<double>> inputs_1_values(inputs_1_info.size(), std::vector<double> (U_size, 0));

//include this part of code as .H file
//add check if idx lies between 0 and 8
auto FillValues = [&](std::vector<std::vector<double>> &values, std::vector<InputFieldInfo> &info) {
    for (int i = 0; i < values.size(); ++i) {
            scalar* data= nullptr;
            std::string name = info[i].name;
            int idx = info[i].idx;
            bool old_time = info[i].old_time;
            assert(idx >= 0);
            assert(idx <= 9);

            if (name == "p_rgh") {
                    data = old_time ? p_rgh.oldTime().data() : p_rgh.data();
                }
            else if (name == "U") {
                    data = old_time ? U.oldTime()[idx].data() : U[idx].data();
                }
            else if (name == "alpha1") {
                    data = old_time ? alpha1.oldTime().data() : alpha1.data();
                }
            else if (name == "gradP") {
                    data = old_time ? gradP.oldTime()[idx].data() : gradP[idx].data();
                }
            else if (name == "gradU") {
                    data = old_time ? gradU.oldTime()[idx].data() : gradU[idx].data();
                }
            else if (name == "gradAW") {
                    data = old_time ? gradAW.oldTime()[idx].data() : gradAW[idx].data();
                }
            else if (name == "inv1GradU") {
                    data = inv1GradU.data();
                }
            else if (name == "inv2GradU") {
                    data = inv2GradU.data();
                }
            else if (name == "strainRateTensor") {
                    data = strainRateTensor[idx].data();
                }
            else if (name == "rotationRateTensor") {
                    data = rotationRateTensor[idx].data();
                }
            else if (name == "T0") {
                    data = T0[idx].data();
                }
            else if (name == "T1") {
                    data = T1[idx].data();
                }
            else if (name == "T2") {
                    data = T2[idx].data();
                }
            else if (name == "T3") {
                    data = T3[idx].data();
                }
            else if (name == "T4") {
                    data = T4[idx].data();
                }
            else if (name == "T5") {
                    data = T5[idx].data();
                }
            else if (name == "T6") {
                    data = T6[idx].data();
                }
            else if (name == "T7") {
                    data = T7[idx].data();
                }
            else if (name == "T8") {
                    data = T8[idx].data();
                }
            else if (name == "T9") {
                    data = T9[idx].data();
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
            else {
                    assert(data);
                }
            memcpy(&(values[i][0]), data, sizeof(double) * U_size);
        }
};

FillValues(inputs_0_values, inputs_0_info);
FillValues(inputs_1_values, inputs_1_info);

std::vector<std::vector<double>> inputs_0_values_t(U_size, std::vector<double> (inputs_0_info.size(), 0));
std::vector<std::vector<double>> inputs_1_values_t(U_size, std::vector<double> (inputs_1_info.size(), 0));

auto Transpose = [&](std::vector<std::vector<double>> &v_transposed, const std::vector<std::vector<double>> &v) {
    assert(v.size() > 0);
    assert(v[0].size() > 0);
    assert(v_transposed.size() == v[0].size());
    assert(v_transposed[0].size() == v.size());

    for (int i = 0; i < v_transposed.size(); ++i)
        for (int j = 0; j < v_transposed[0].size(); ++j)
            v_transposed[i][j] = v[j][i];

};

Transpose(inputs_0_values_t, inputs_0_values);
Transpose(inputs_1_values_t, inputs_1_values);
















