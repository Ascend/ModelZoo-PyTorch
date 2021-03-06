/*
* BSD 3-Clause License
*
* Copyright (c) 2017 xxxx
* All rights reserved.
* Copyright 2021 Huawei Technologies Co., Ltd
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice, this
*   list of conditions and the following disclaimer.
*
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
* * Neither the name of the copyright holder nor the names of its
*   contributors may be used to endorse or promote products derived from
*   this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* ============================================================================
*//*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "cudaUtils.h"
#include "engineCache.h"
#include "logging.h"
#include "waveGlowBuilder.h"

#include "NvInfer.h"

#include <iostream>
#include <memory>

using namespace nvinfer1;
using namespace tts;

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

bool matches(const std::string& arg, const std::string& flag)
{
  return arg.length() >= flag.length() && arg.substr(0, flag.length()) == flag;
}

int parseNumFlag(
    const int argc, const char** argv, const std::string& flag, int* i)
{
  int value;
  const std::string arg(argv[*i]);
  if (arg.length() > flag.length()) {
    value = std::stol(arg.substr(flag.length()));
  } else if (*i + 1 < argc) {
    ++(*i);
    value = std::stol(argv[*i]);
  } else {
    throw std::runtime_error("Missing argument for '" + flag + "'.");
  }
  return value;
}

int parseAmpFlag(
    const int argc, const char** argv, const std::string& flag, int* i)
{
  std::string str;
  const std::string arg(argv[*i]);
  if (arg.length() > flag.length()) {
    str = arg.substr(flag.length());
  } else if (*i + 1 < argc) {
    ++(*i);
    str = argv[*i];
  } else {
    throw std::runtime_error("Missing argument for '" + flag + "'.");
  }

  int value;
  if (str == "fp32") {
    value = 0;
  } else if (str == "amp") {
    value = 1;
  } else {
    throw std::runtime_error(
        "Invalid argument for precision (amp|fp32): " + str);
  }

  return value;
}

void usage(const std::string& binName)
{
  std::cerr << "usage: " << std::endl;
  std::cerr << "    " << binName << " <model file> <engine file> [options]\n";
  std::cerr << "options:" << std::endl;
  std::cerr << "  -B<batch size>" << std::endl;
  std::cerr << "  -F<precision (fp32|amp)>" << std::endl;
  std::cerr << "  -h" << std::endl;
}

void parseArgs(
    const int argc,
    const char** const argv,
    std::string* model,
    std::string* enginePath,
    int* batchSize,
    int* useAMP)
{
  bool modelSet = false;
  bool enginePathSet = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (matches(arg, "-B")) {
      *batchSize = parseNumFlag(argc, argv, "-B", &i);
    } else if (matches(arg, "-F")) {
      *useAMP = parseAmpFlag(argc, argv, "-F", &i);
    } else if (matches(arg, "-h")) {
      usage(argv[0]);
      exit(0);
    } else {
      if (!modelSet) {
        *model = arg;
        modelSet = true;
      } else if (!enginePathSet) {
        *enginePath = arg;
        enginePathSet = true;
      } else {
        throw std::runtime_error("Unknown extra argument '" + arg + "'.");
      }
    }
  }
}

/******************************************************************************
 * MAIN ***********************************************************************
 *****************************************************************************/

int main(int argc, const char* argv[])
{
  std::string waveglowModelPath;
  std::string enginePath;

  int batchSize = 1;
  int useFP16 = true;

  parseArgs(argc, argv, &waveglowModelPath, &enginePath, &batchSize, &useFP16);
  if (waveglowModelPath.empty() || enginePath.empty()) {
    usage(argv[0]);
    return 1;
  }

  CudaUtils::printDeviceInformation();

  try {
    std::shared_ptr<Logger> logger(new Logger(ILogger::Severity::kERROR));

    TRTPtr<IBuilder> builder(createInferBuilder(*logger));

    EngineCache cache(logger);

    WaveGlowBuilder waveglowBuilder(waveglowModelPath, logger);
    const TRTPtr<ICudaEngine> wgEng
        = waveglowBuilder.build(*builder, batchSize, useFP16);

    cache.save(*wgEng, enginePath);
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
