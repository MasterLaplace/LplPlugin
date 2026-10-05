/**************************************************************************
 * VkWrapper v0.0.0
 *
 * VkWrapper is a software package, part of LplPlugin.
 *
 * This file is part of the VkWrapper project that is under the MIT License.
 * https://opensource.org/license/mit
 * Copyright © 2024 by @MasterLaplace, All rights reserved.
 *
 * VkWrapper is free software: you can use, copy, modify, merge, publish and
 * distribute it under the terms of the MIT License, provided this copyright
 * notice and the permission notice are kept. See the LICENSE file.
 *
 * @file vkwrapper_export.h
 * @brief Compile-Time exportation of the project path.
 *
 * @author @MasterLaplace
 * @version 0.0.0
 * @date 2024-10-15
 **************************************************************************/

// clang-format off
#ifndef EXPORT_H_
    #define EXPORT_H_

#ifdef __cplusplus
#include <filesystem>

#define PROJECT_SOURCE_DIR std::filesystem::current_path().string() + "/"
#else
#include <stdlib.h>
#include <string.h>

#define PROJECT_SOURCE_DIR getenv("PWD")
#endif

#endif /* !EXPORT_H_ */
// clang-format on
