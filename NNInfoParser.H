#include <fstream>
#include <cassert>
#include "json.hpp"

using json = nlohmann::json;

struct InputFieldInfo
{
    std::string name;
    int idx;
    bool old_time;

    InputFieldInfo(): name(""), idx(0), old_time(false) {}
};

void ParseString(InputFieldInfo &info, const std::string &in_str)
{
    std::string out_str = in_str;
    int idx = 0;
    bool old_time = false;

    size_t pos = out_str.find("res/");
    if (pos == std::string::npos)
    {
        old_time = true;
    }
    else
    {
        out_str.erase(0, 4);
    }

    char last_sym = out_str[out_str.length() - 1];
    if ('0' <= last_sym && last_sym <= '9' && out_str[0] != 'I')
    {
        idx = last_sym - '0';
        out_str.pop_back();
    }
    //special case
    if (out_str == "alpha.water")
      out_str = "alpha1";

    info.name = out_str;
    info.idx = idx;
    info.old_time = old_time;
}

