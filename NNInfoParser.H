#include <fstream>
#include <cassert>
#include "json.hpp"

using json = nlohmann::json;

using Field_ = std::vector<double>;

struct FieldInfo
{
    std::string name;
    int idx;
    bool prev_time_step;

    FieldInfo(): name(""), idx(0), prev_time_step(false) {}
    FieldInfo(const std::string &name, int idx, bool prev_time_step):
              name(name), idx(idx), prev_time_step(prev_time_step) {}
};

struct FieldWithInfo
{
    Field_ field;
    FieldInfo info;
};

struct Branch
{
    std::string name;

    std::vector<FieldWithInfo> fields;
    size_t branch_size;

    Branch(): name(""), fields(0), branch_size(0) {}
    Branch(std::string name, size_t size): name(name), fields(size), branch_size(size) {}
    Branch(size_t size): name(""), fields(size), branch_size(size) {}
    Branch(const std::vector<FieldWithInfo> &fields):
        name(""), fields(fields), branch_size(fields.size()) {}
};

void ParseRawFieldName(FieldInfo &info, const std::string &raw_field_name)
{
    if (raw_field_name.empty())
        throw std::runtime_error("ParseString: input string is empty!");

    std::string field_name = raw_field_name;
    int idx = 0;
    bool prev_time_step = false;

    //check if the field corresponds to the previuos time step
    size_t pos = field_name.find("res/");
    if (pos == std::string::npos)
        prev_time_step = true;
    else
        field_name.erase(0, 4);

    //check if the last symbol denotes vector or tensor component index
    char last_sym = field_name[field_name.length() - 1];
    if ('0' <= last_sym && last_sym <= '9' && field_name[0] != 'I')
    {
        idx = last_sym - '0';
        field_name.pop_back();
    }

    //special case with alpha.water field name
    if (field_name == "alpha.water")
      field_name = "alpha1";

    info = FieldInfo(field_name, idx, prev_time_step);
}

