#ifndef COMMON_UTIL_H_
#define COMMON_UTIL_H_

#include <string>

inline std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) elems.push_back(item);
    return elems;
}

inline std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

inline std::string get_file_extension(const std::string& filename) {
    std::vector<std::string> components = split(filename, '.');

    if ( components.size() == 0 ) {
        return filename;
    }
    return components[components.size() - 1];
}

inline int next_pow_2(int val)
{
    return pow(2, ceil(log(val) / log(2)));
}

#endif //COMMON_UTIL_H_
