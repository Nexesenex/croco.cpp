#pragma once

#include "ggml.h"

// Keep KoboldCpp's process-wide GGML logger active when stable-diffusion.cpp
// installs its own callback during model or upscaler initialization.
void kcpp_sd_preserve_ggml_logger();
void kcpp_sd_ggml_log_callback(ggml_log_level level, const char* text, void* user_data);
