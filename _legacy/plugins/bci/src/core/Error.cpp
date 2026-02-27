/**
 * @file Error.cpp
 * @brief Implementation of Error::format().
 * @author MasterLaplace
 */

#include "core/Error.hpp"

#include <sstream>

namespace bci {

std::string Error::format() const
{
    std::ostringstream os;
    os << '[' << errorCodeName(code) << "] " << message
       << " (" << location.file_name() << ':' << location.line() << ')';
    return os.str();
}

} // namespace bci
