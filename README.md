# koboldcpp

KoboldCpp is an easy-to-use AI text-generation software for GGML and GGUF models, inspired by the original **KoboldAI**. It's a single self-contained distributable from Concedo, that builds off llama.cpp, and adds a versatile **KoboldAI API endpoint**, additional format support, Stable Diffusion image generation, speech-to-text, backward compatibility, as well as a fancy UI with persistent stories, editing tools, save formats, memory, world info, author's note, characters, scenarios and everything KoboldAI and KoboldAI Lite have to offer.

![Preview](media/preview.png)
![Preview](media/preview2.png)
![Preview](media/preview3.png)
![Preview](media/preview4.png)

## Windows Usage (Precompiled Binary, Recommended)
- Windows binaries are provided in the form of **koboldcpp.exe**, which is a pyinstaller wrapper containing all necessary files. **[Download the latest koboldcpp.exe release here](https://github.com/LostRuins/koboldcpp/releases/latest)**
- To run, simply execute **koboldcpp.exe**.
- Launching with no command line arguments displays a GUI containing a subset of configurable settings. Generally you dont have to change much besides the `Presets` and `GPU Layers`. Read the `--help` for more info about each settings.
- By default, you can connect to http://localhost:5001
- You can also run it using the command line. For info, please check `koboldcpp.exe --help`

## Linux Usage (Precompiled Binary, Recommended)
On modern Linux systems, you should download the `koboldcpp-linux-x64-cuda1150` prebuilt PyInstaller binary on the **[releases page](https://github.com/LostRuins/koboldcpp/releases/latest)**. Simply download and run the binary.

Alternatively, you can also install koboldcpp to the current directory by running the following terminal command:
```
curl -fLo koboldcpp https://github.com/LostRuins/koboldcpp/releases/latest/download/koboldcpp-linux-x64-cuda1150 && chmod +x koboldcpp
```
After running this command you can launch Koboldcpp from the current directory using `./koboldcpp` in the terminal (for CLI usage, run with `--help`).

## MacOS (Precompiled Binary)
- PyInstaller binaries for Modern ARM64 MacOS (M1, M2, M3) are now available! **[Simply download and run the MacOS binary](https://github.com/LostRuins/koboldcpp/releases/latest)**
- Alternatively, or for older x86 MacOS computers, you can clone the repo and compile from source code, see Compiling for MacOS below.

## Run on Colab
- KoboldCpp now has an **official Colab GPU Notebook**! This is an easy way to get started without installing anything in a minute or two. [Try it here!](https://colab.research.google.com/github/LostRuins/koboldcpp/blob/concedo/colab.ipynb).
- Note that KoboldCpp is not responsible for your usage of this Colab Notebook, you should ensure that your own usage complies with Google Colab's terms of use.

## Run on RunPod
- KoboldCpp can now be used on RunPod cloud GPUs! This is an easy way to get started without installing anything in a minute or two, and is very scalable, capable of running 70B+ models at afforable cost. [Try our RunPod image here!](https://koboldai.org/runpodcpp).

## Docker
- The official docker can be found at https://hub.docker.com/r/koboldai/koboldcpp
- If you're building your own docker, remember to set CUDA_DOCKER_ARCH or enable LLAMA_PORTABLE

## Obtaining a GGUF model
- KoboldCpp uses GGUF models. They are not included here, but you can download GGUF files from other places such as [TheBloke's Huggingface](https://huggingface.co/TheBloke). Search for "GGUF" on huggingface.co for plenty of compatible models in the `.gguf` format.
- For beginners, we recommend the models [Airoboros Mistral](https://huggingface.co/TheBloke/airoboros-mistral2.2-7B-GGUF/resolve/main/airoboros-mistral2.2-7b.Q4_K_S.gguf) or [Tiefighter 13B](https://huggingface.co/KoboldAI/LLaMA2-13B-Tiefighter-GGUF/resolve/main/LLaMA2-13B-Tiefighter.Q4_K_S.gguf) (larger model).
- [Alternatively, you can download the tools to convert models to the GGUF format yourself here](https://kcpptools.concedo.workers.dev). Run `convert-hf-to-gguf.py` to convert them, then `quantize_gguf.exe` to quantize the result.

## Improving Performance
- **GPU Acceleration**: If you're on Windows with an Nvidia GPU you can get CUDA support out of the box using the `--usecublas`  flag (Nvidia Only), or `--usevulkan` (Any GPU), make sure you select the correct .exe with CUDA support.
- **GPU Layer Offloading**: Add `--gpulayers` to offload model layers to the GPU. The more layers you offload to VRAM, the faster generation speed will become. Experiment to determine number of layers to offload, and reduce by a few if you run out of memory.
- **Increasing Context Size**: Use `--contextsize (number)` to increase context size, allowing the model to read more text. Note that you may also need to increase the max context in the KoboldAI Lite UI as well (click and edit the number text field).
- **Old CPU Compatibility**: If you are having crashes or issues, you can try turning off BLAS with the `--noblas` flag. You can also try running in a non-avx2 compatibility mode with `--noavx2`. Lastly, you can try turning off mmap with `--nommap`.

For more information, be sure to run the program with the `--help` flag, or **[check the wiki](https://github.com/LostRuins/koboldcpp/wiki).**

## Compiling KoboldCpp From Source Code

### Compiling on Linux (Using koboldcpp.sh automated compiler script)
when you can't use the precompiled binary directly, we provide an automated build script which uses conda to obtain all dependencies, and generates (from source) a ready-to-use a pyinstaller binary for linux users.
- Clone the repo with `git clone https://github.com/LostRuins/koboldcpp.git`
- Simply execute the build script with `./koboldcpp.sh dist` and run the generated binary. (Not recomended for systems that already have an existing installation of conda. Dependencies: curl, bzip2)
```
./koboldcpp.sh # This launches the GUI for easy configuration and launching (X11 required).
./koboldcpp.sh --help # List all available terminal commands for using Koboldcpp, you can use koboldcpp.sh the same way as our python script and binaries.
./koboldcpp.sh rebuild # Automatically generates a new conda runtime and compiles a fresh copy of the libraries. Do this after updating Koboldcpp to keep everything functional.
./koboldcpp.sh dist # Generate your own precompiled binary (Due to the nature of Linux compiling these will only work on distributions equal or newer than your own.)
```

### Compiling on Linux (Manual Method)
- To compile your binaries from source, clone the repo with `git clone https://github.com/LostRuins/koboldcpp.git`
- A makefile is provided, simply run `make`.
- Optional OpenBLAS: Link your own install of OpenBLAS manually with `make LLAMA_OPENBLAS=1`
- Optional CLBlast: Link your own install of CLBlast manually with `make LLAMA_CLBLAST=1`
- Note: for these you will need to obtain and link OpenCL and CLBlast libraries.
  - For Arch Linux: Install `cblas` `openblas` and `clblast`.
  - For Debian: Install `libclblast-dev` and `libopenblas-dev`.
- You can attempt a CuBLAS build with `LLAMA_CUBLAS=1`, (or `LLAMA_HIPBLAS=1` for AMD). You will need CUDA Toolkit installed. Some have also reported success with the CMake file, though that is more for windows.
- For a full featured build (all backends), do `make LLAMA_OPENBLAS=1 LLAMA_CLBLAST=1 LLAMA_CUBLAS=1 LLAMA_VULKAN=1`. (Note that `LLAMA_CUBLAS=1` will not work on windows, you need visual studio)
- After all binaries are built, you can run the python script with the command `koboldcpp.py [ggml_model.gguf] [port]`

### Compiling on Windows
- You're encouraged to use the .exe released, but if you want to compile your binaries from source at Windows, the easiest way is:
  - Get the latest release of w64devkit (https://github.com/skeeto/w64devkit). Be sure to use the "vanilla one", not i686 or other different stuff. If you try they will conflit with the precompiled libs!
  - Clone the repo with `git clone https://github.com/LostRuins/koboldcpp.git`
  - Make sure you are using the w64devkit integrated terminal, then run `make` at the KoboldCpp source folder. This will create the .dll files.
  - If you want to generate the .exe file, make sure you have the python module PyInstaller installed with pip (`pip install PyInstaller`). Then run the script `make_pyinstaller.bat`
  - The koboldcpp.exe file will be at your dist folder.
- **Building with CUDA**: Visual Studio, CMake and CUDA Toolkit is required. Clone the repo, then open the CMake file and compile it in Visual Studio. Copy the `koboldcpp_cublas.dll` generated into the same directory as the `koboldcpp.py` file. If you are bundling executables, you may need to include CUDA dynamic libraries (such as `cublasLt64_11.dll` and `cublas64_11.dll`) in order for the executable to work correctly on a different PC.
- **Replacing Libraries (Not Recommended)**: If you wish to use your own version of the additional Windows libraries (OpenCL, CLBlast and OpenBLAS), you can do it with:
  - OpenCL - tested with https://github.com/KhronosGroup/OpenCL-SDK . If you wish to compile it, follow the repository instructions. You will need vcpkg.
  - CLBlast - tested with https://github.com/CNugteren/CLBlast . If you wish to compile it you will need to reference the OpenCL files. It will only generate the ".lib" file if you compile using MSVC.
  - OpenBLAS - tested with https://github.com/xianyi/OpenBLAS .
  - Move the respectives .lib files to the /lib folder of your project, overwriting the older files.
  - Also, replace the existing versions of the corresponding .dll files located in the project directory root (e.g. libopenblas.dll).
  - Make the KoboldCpp project using the instructions above.

### Compiling on MacOS
- You can compile your binaries from source. You can clone the repo with `git clone https://github.com/LostRuins/koboldcpp.git`
- A makefile is provided, simply run `make`.
- If you want Metal GPU support, instead run `make LLAMA_METAL=1`, note that MacOS metal libraries need to be installed.
- After all binaries are built, you can run the python script with the command `koboldcpp.py --model [ggml_model.gguf]` (and add `--gpulayers (number of layer)` if you wish to offload layers to GPU).

### Compiling on Android (Termux Installation)
- [Install and run Termux from F-Droid](https://f-droid.org/en/packages/com.termux/)
- Enter the command `termux-change-repo` and choose `Mirror by BFSU`
- Install dependencies with `pkg install wget git python` (plus any other missing packages)
- Install dependencies `apt install openssl` (if needed)
- Clone the repo `git clone https://github.com/LostRuins/koboldcpp.git`
- Navigate to the koboldcpp folder `cd koboldcpp`
- Build the project `make`
- Grab a small GGUF model, such as `wget https://huggingface.co/concedo/KobbleTinyV2-1.1B-GGUF/resolve/main/KobbleTiny-Q4_K.gguf`
- Start the python server `python koboldcpp.py --model KobbleTiny-Q4_K.gguf`
- Connect to `http://localhost:5001` on your mobile browser
- If you encounter any errors, make sure your packages are up-to-date with `pkg up`
- GPU acceleration for Termux may be possible but I have not explored it. If you find a good cross-device solution, do share or PR it.

## AMD Users
- Please check out https://github.com/YellowRoseCx/koboldcpp-rocm

## Third Party Resources
- These unofficial resources have been contributed by the community, and may be outdated or unmaintained. No official support will be provided for them!
  - Arch Linux Packages: [CUBLAS](https://aur.archlinux.org/packages/koboldcpp-cuda), and [HIPBLAS](https://aur.archlinux.org/packages/koboldcpp-hipblas).
  - Unofficial Dockers: [korewaChino](https://github.com/korewaChino/koboldCppDocker) and [noneabove1182](https://github.com/noneabove1182/koboldcpp-docker)
  - Nix & NixOS: KoboldCpp is available on Nixpkgs and can be installed by adding just `koboldcpp` to your `environment.systemPackages`.
    - Make sure to have `nixpkgs.config.allowUnfree`, `hardware.opengl.enable` *(`hardware.graphics.enable` if you're using unstable channel)* and `nixpkgs.config.cudaSupport` set to `true` to enable CUDA.
    - Metal is enabled by default on macOS, Vulkan support is enabled by default on both Linux and macOS, ROCm support isn't available yet.
    - You can also use `nix3-run` to use KoboldCpp: `nix run --expr ``with import <nixpkgs> { config = { allowUnfree = true; cudaSupport = true; }; }; koboldcpp`` --impure`
    - Or use `nix-shell`: `nix-shell --expr 'with import <nixpkgs> { config = { allowUnfree = true; cudaSupport = true; }; }; koboldcpp' --run "koboldcpp" --impure`
    - Packages (like OpenBlast, CLBLast, Vulkan, etc.) can be overridden, please refer to the [17th Nix Pill - Nixpkgs Overriding Packages](https://nixos.org/guides/nix-pills/17-nixpkgs-overriding-packages)

## Questions and Help Wiki
- **First, please check out [The KoboldCpp FAQ and Knowledgebase](https://github.com/LostRuins/koboldcpp/wiki) which may already have answers to your questions! Also please search through past issues and discussions.**
- If you cannot find an answer, open an issue on this github, or find us on the [KoboldAI Discord](https://koboldai.org/discord).

## KoboldCpp and KoboldAI API Documentation
- [Documentation for KoboldAI and KoboldCpp endpoints can be found here](https://lite.koboldai.net/koboldcpp_api)

## KoboldCpp Public Demo
- [A public KoboldCpp demo can be found at our Huggingface Space. Please do not abuse it.](https://koboldai-koboldcpp-tiefighter.hf.space/)

## Considerations
- For Windows: No installation, single file executable, (It Just Works)
- Since v1.0.6, requires libopenblas, the prebuilt windows binaries are included in this repo. If not found, it will fall back to a mode without BLAS.
- Since v1.15, requires CLBlast if enabled, the prebuilt windows binaries are included in this repo. If not found, it will fall back to a mode without CLBlast.
- Since v1.33, you can set the context size to be above what the model supports officially. It does increases perplexity but should still work well below 4096 even on untuned models. (For GPT-NeoX, GPT-J, and Llama models) Customize this with `--ropeconfig`.
- Since v1.42, supports GGUF models for LLAMA and Falcon
- Since v1.55, lcuda paths on Linux are hardcoded and may require manual changes to the makefile if you do not use koboldcpp.sh for the compilation.
- Since v1.60, provides native image generation with StableDiffusion.cpp, you can load any SD1.5 or SDXL .safetensors model and it will provide an A1111 compatible API to use.
- **I try to keep backwards compatibility with ALL past llama.cpp models**. But you are also encouraged to reconvert/update your models if possible for best results.

## License
- The original GGML library and llama.cpp by ggerganov are licensed under the MIT License
- However, KoboldAI Lite is licensed under the AGPL v3.0 License
- KoboldCpp code and other files are also under the AGPL v3.0 License unless otherwise stated

## Notes
- If you wish, after building the koboldcpp libraries with `make`, you can rebuild the exe yourself with pyinstaller by using `make_pyinstaller.bat`
- API documentation available at `/api` (e.g. `http://localhost:5001/api`) and https://lite.koboldai.net/koboldcpp_api. An OpenAI compatible API is also provided at `/v1` route (e.g. `http://localhost:5001/v1`).
- **All up-to-date GGUF models are supported**, and KoboldCpp also includes backward compatibility for older versions/legacy GGML `.bin` models, though some newer features might be unavailable.
- An incomplete list of models and architectures is listed, but there are *many hundreds of other GGUF models*. In general, if it's GGUF, it should work.
  - Llama / Llama2 / Llama3 / Alpaca / GPT4All / Vicuna / Koala / Pygmalion / Metharme / WizardLM
  - Mistral / Mixtral / Miqu
  - Qwen / Qwen2 / Yi
  - Gemma / Gemma2
  - GPT-2 / Cerebras
  - Phi-2 / Phi-3
  - GPT-NeoX / Pythia / StableLM / Dolly / RedPajama
  - GPT-J / RWKV4 / MPT / Falcon / Starcoder / Deepseek and many more
  - [Stable Diffusion 1.5 and SDXL safetensor models](https://github.com/LostRuins/koboldcpp/wiki#can-i-generate-images-with-koboldcpp)
  - [LLaVA based Vision models and multimodal projectors (mmproj)](https://github.com/LostRuins/koboldcpp/wiki#what-is-llava-and-mmproj)
  - [Whisper models for Speech-To-Text](https://huggingface.co/koboldcpp/whisper/tree/main)


# IK_Llama, a llama.cpp clone with better CPU performance

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## TL;DR

This repository is a clone of [llama.cpp](https://github.com/ggerganov/llama.cpp) with the following improvements
* Better implementation of CPU matrix multiplications (`AVX2` and `ARM_NEON`) for `fp16/fp32` and all k-, i-, and legacy `llama.cpp` quants, that leads to a significant improvement in prompt processing (PP) speed, typically in the range of 2X, but up to 4X for some quantization types. Token generation (TG) also benefits, but to a lesser extent due to TG being memory bound
* Faster CPU inference for MoE models with similar performance gains
* Implementation of the [Bitnet b1.58](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) model for the CPU (`AVX2` and `ARM_NEON`) and GPU (`CUDA` and `Metal`). This implementation is much faster than the unmerged `llama.cpp` [PR-8151](https://github.com/ggerganov/llama.cpp/pull/8151)

If you are not already familiar with [llama.cpp](https://github.com/ggerganov/llama.cpp), it is better to start there. For those familiar with `llama.cpp`, everything here works the same as in `llama.cpp` (or at least the way `llama.cpp` worked when I last synced on Aug 12 2024).

Note that I have published some, but not all, of the code in this repository in a series of [llamafile](https://github.com/Mozilla-Ocho/llamafile) PRs ([394](https://github.com/Mozilla-Ocho/llamafile/pull/394), [405](https://github.com/Mozilla-Ocho/llamafile/pull/405), [428](https://github.com/Mozilla-Ocho/llamafile/pull/428), [435](https://github.com/Mozilla-Ocho/llamafile/pull/435), [453](https://github.com/Mozilla-Ocho/llamafile/pull/453), and [464](https://github.com/Mozilla-Ocho/llamafile/pull/464))

The implementation of matrix-matrix and matrix-vector multiplications is in a single C++ source file (`iqk_mul_mat.cpp`) with just two interface functions `iqk_mul_mat` (`fp16/fp32` and quantized matrix multiplications) and `iqk_mul_mat_moe` (as `iqk_mul_mat` but meant to be used for the FFN part of a MoE model). Under the hood `iqk_mul_mat_moe` uses the same implementation as `iqk_mul_mat`, with the only difference being where results are stored in memory. Bitnet quantization related stuff is in `iqk-quantize.cpp`.   

## Why?

Mostly out of curiosity:
* Justine Tunney's `tinyBLAS`, which she contributed to `llama.cpp` in [PR 6414](https://github.com/ggerganov/llama.cpp/pull/6414), only works for `Q4_0`, `Q8_0` and `fp16/bf16` models. In the surrounding discussion about possibly extending `tinyBLAS` to k- and i-quants, she felt that k-quants are [not amenable to block-tiling](https://github.com/ggerganov/llama.cpp/pull/6840#issuecomment-2072995387), which is required to improve performance. This statement piqued my curiosity, so here we are.
* Bitnet-1.58b has been one of the [most discussed topics](https://github.com/ggerganov/llama.cpp/issues/5761#issuecomment-2198380366) in the `llama.cpp` project, so eventually I decided to see how efficiently one can implement a ternary model

Curiosity aside, improved CPU performance may be (or may become) important in practice. According to The Register, 70% of AI inference [is done on the CPU of mobile phones](https://www.theregister.com/2024/05/30/arm_cortex_x925_ai_cores/?td=rt-3a), at least in the Android world (but I haven't come around to actually comparing performance on a phone). With ever increasing number of LLM model parameters, and with Meta's 400B model just released, the CPU may become the only viable option for people not willing (or not able to) rent/buy uber expensive GPU instances capable of running such models. Granted, one would need a pretty beefy computer to run a 400B model, and inference speed will be sluggish, but at least one will not need to spend the equivalent of a luxury apartment in the downtown of the city where I live to buy the GPU system capable of running the model.

## Performance comparison to llama.cpp

The results in the following tables are obtained with these parameters:
* Model is LLaMA-v3-8B for `AVX2` and LLaMA-v2-7B for `ARM_NEON`
* The `AVX2` CPU is a 16-core Ryzen-7950X
* The `ARM_NEON` CPU is M2-Max
* `tinyBLAS` is enabled in `llama.cpp`
* `llama.cpp` results are for `build: 081fe431 (3441)`, which was the current `llama.cpp` master branch when I pulled on July 23 2024.
* The projects are built without `CUDA` support, no `BLAS`, and Accelerate framework disabled

### Prompt processing

Here I set the number of threads to be equal to the number of (performance) cores of the CPU, so 16 threads for the Ryzen-7950X and 8 threads for the M2-Max. The following table summarizes the results. To not make the table too long, I have listed only quantized models containing predominantly one quantization type (i.e., excluded the `QX_K - Medium/Large` variants, which are typically a mix of `QX_K` and `Q(X+1)_K`, as well as `IQ2_S` and `IQ3_XS`).  

The command line to generate the benchmark data is
```
./bin/llama-bench -m $model -p 512 -n 0 -t $num_threads -ngl 0
```

| Quantization|       size | backend    | threads | t/s (llama.cpp)  | t/s (iqk_mul_mat)| Speedup |
| ----------- | ---------: | ---------- | ------: | ---------------: | ---------------: | ------: |
| 8B F16      |  14.96 GiB | AVX2       |      16 |    112.37 ± 0.40 |    131.27 ± 0.38 |  1.168  |
| 7B F16      |  12.55 GiB | NEON       |       8 |     90.28 ± 1.25 |     95.34 ± 0.15 |  1.056  |
| 8B Q8_0     |   7.95 GiB | AVX2       |      16 |    118.07 ± 0.53 |    134.00 ± 0.47 |  1.135  |
| 7B Q8_0     |   6.67 GiB | NEON       |       8 |     77.25 ± 1.81 |     94.14 ± 1.15 |  1.219  |
| 8B Q4_0     |   4.35 GiB | AVX2       |      16 |    104.46 ± 0.33 |    130.20 ± 0.29 |  1.246  |
| 7B Q4_0     |   3.57 GiB | NEON       |       8 |     65.46 ± 0.79 |     76.22 ± 0.71 |  1.164  |
| 8B Q4_1     |   4.77 GiB | AVX2       |      16 |     57.83 ± 0.24 |    160.69 ± 0.49 |  2.779  |
| 7B Q4_1     |   3.95 GiB | NEON       |       8 |     37.40 ± 0.50 |     65.83 ± 0.98 |  1.760  |
| 8B Q5_0     |   5.22 GiB | AVX2       |      16 |     53.50 ± 0.35 |    122.62 ± 0.48 |  2.292  |
| 7B Q5_0     |   4.34 GiB | NEON       |       8 |     29.31 ± 0.51 |     67.51 ± 1.17 |  2.303  |
| 8B Q5_1     |   5.64 GiB | AVX2       |      16 |     50.85 ± 0.36 |    147.15 ± 0.47 |  2.894  |
| 7B Q5_1     |   4.72 GiB | NEON       |       8 |     26.02 ± 0.37 |     58.49 ± 0.85 |  2.248  |
| 8B Q2_K_S   |   2.78 GiB | AVX2       |      16 |    110.11 ± 0.28 |    192.47 ± 1.35 |  1.748  |
| 7B Q2_K_S   |   2.16 GiB | NEON       |       8 |     35.44 ± 0.06 |     77.93 ± 1.64 |  2.199  |
| 8B Q3_K_S   |   3.41 GiB | AVX2       |      16 |     77.42 ± 0.36 |    181.64 ± 0.44 |  2.346  |
| 7B Q3_K_S   |   2.75 GiB | NEON       |       8 |     26.79 ± 0.03 |     59.38 ± 1.08 |  2.216  |
| 8B Q4_K_S   |   4.36 GiB | AVX2       |      16 |     98.92 ± 0.34 |    185.35 ± 0.39 |  1.874  |
| 7B Q4_K_S   |   3.59 GiB | NEON       |       8 |     46.55 ± 0.67 |     76.31 ± 0.38 |  1.639  |
| 8B Q5_K_S   |   5.21 GiB | AVX2       |      16 |     69.44 ± 0.31 |    179.62 ± 0.69 |  2.587  |
| 7B Q5_K_S   |   4.33 GiB | NEON       |       8 |     30.18 ± 0.23 |     65.34 ± 0.79 |  2.165  |
| 8B Q6_K     |   6.14 GiB | AVX2       |      16 |     74.89 ± 0.26 |    181.86 ± 0.55 |  2.428  |
| 7B Q6_K     |   5.15 GiB | NEON       |       8 |     28.12 ± 1.24 |     60.75 ± 1.15 |  2.160  |
| 8B IQ2_XXS  |   2.23 GiB | AVX2       |      16 |     42.57 ± 0.16 |    126.63 ± 0.55 |  2.975  |
| 7B IQ2_XXS  |   1.73 GiB | NEON       |       8 |     20.87 ± 0.20 |     64.29 ± 1.12 |  3.080  |
| 8B IQ2_XS   |   2.42 GiB | AVX2       |      16 |     46.45 ± 0.27 |    125.46 ± 0.43 |  2.701  |
| 7B IQ2_XS   |   1.89 GiB | NEON       |       8 |     22.77 ± 0.21 |     51.15 ± 0.24 |  2.246  |
| 8B IQ2_M    |   2.74 GiB | AVX2       |      16 |     40.76 ± 0.18 |    113.07 ± 0.48 |  2.774  |
| 7B IQ2_M    |   2.20 GiB | NEON       |       8 |     14.95 ± 0.26 |     44.87 ± 0.50 |  3.001  |
| 8B IQ3_XXS  |   3.04 GiB | AVX2       |      16 |     31.95 ± 0.20 |    109.86 ± 0.45 |  3.438  |
| 7B IQ3_XXS  |   2.41 GiB | NEON       |       8 |     14.40 ± 0.10 |     53.58 ± 0.85 |  3.721  |
| 8B IQ3_S    |   3.42 GiB | AVX2       |      16 |     28.04 ± 0.08 |     96.28 ± 0.45 |  3.434  |
| 7B IQ3_S    |   2.75 GiB | NEON       |       8 |     12.08 ± 0.30 |     49.72 ± 0.06 |  4.116  |
| 8B IQ4_XS   |   4.13 GiB | AVX2       |      16 |     68.98 ± 0.31 |    180.34 ± 0.55 |  2.614  |
| 7B IQ4_XS   |   3.37 GiB | NEON       |       8 |     40.67 ± 1.97 |     75.11 ± 1.97 |  1.847  |
| 8B IQ4_NL   |   4.35 GiB | AVX2       |      16 |     59.94 ± 0.21 |    129.06 ± 0.43 |  2.153  |
| 7B IQ4_NL   |   3.56 GiB | NEON       |       8 |     34.36 ± 0.81 |     76.02 ± 1.36 |  2.212  |

We see that `llama.cpp` achieves respectable performance for `fp16`, `Q8_0`, and `Q4_0`, being only up to 25% slower than this implementation. This is thanks to the use of Justine Tunney's `tinyBLAS`, which is utilized for these quantization types. For all other quants we observe performance gains in the `1.75X - 4X` range, which is not a small feat considering that the `ggml` matrix multiplication functions has been rewritten several times since `llama.cpp` was first published. Performance gains are larger for i-quants due to the higher quant unpacking cost (see discussion in "To tile or not to tile")

### Token generation

On the Ryzen-7950X TG is memory bound, and for many quantization types peak performance is achieved at just 4 threads. Hence, only results for 2 and 4 threads are shown for `AVX2`. The M2-Max has a much more capable memory subsystem and as a result performance keep increasing up to 8 threads. Thus, results are given for up to 8 threads for `ARM_NEON`.

The command line to generate the data was
```
./bin/llama-bench -m $model -p 0 -n 128 -t $num_threads -ngl 0
```

| Quantization|       size | backend    | threads | t/s (llama.cpp)  | t/s (iqk_mul_mat)| Speedup |
| ---------- | ---------: | ---------- | ------: | ---------------: | ---------------: | ------: |
| 8B F16     |  14.96 GiB | AVX2       |       1 |      2.20 ± 0.00 |      2.25 ± 0.00 |  1.023  |
|            |            |            |       2 |      3.63 ± 0.00 |      3.68 ± 0.00 |  1.014  |
|            |            |            |       4 |      4.20 ± 0.00 |      4.20 ± 0.00 |  1.000  |
| 7B F16     |  12.55 GiB | NEON       |       2 |      6.94 ± 0.27 |      7.40 ± 0.01 |  1.066  |
|            |            |            |       4 |      8.73 ± 0.01 |      8.83 ± 0.01 |  1.011  |
|            |            |            |       6 |      9.05 ± 0.02 |      9.05 ± 0.01 |  1.000  |
| 8B Q8_0    |   7.95 GiB | AVX2       |       2 |      5.03 ± 0.00 |      7.87 ± 0.00 |  1.565  |
|            |            |            |       4 |      7.40 ± 0.00 |      7.82 ± 0.00 |  1.057  |
| 7B Q8_0    |   6.67 GiB | NEON       |       2 |      8.29 ± 0.44 |     12.07 ± 0.10 |  1.456  |
|            |            |            |       4 |     13.53 ± 0.03 |     15.77 ± 0.08 |  1.166  |
|            |            |            |       8 |     16.24 ± 0.10 |     16.94 ± 0.04 |  1.043  |
| 8B Q4_0    |   4.35 GiB | AVX2       |       2 |      6.36 ± 0.00 |     10.28 ± 0.00 |  1.616  |
|            |            |            |       4 |     10.97 ± 0.06 |     13.55 ± 0.07 |  1.235  |
| 7B Q4_0    |   3.57 GiB | NEON       |       2 |      9.77 ± 0.02 |     13.69 ± 0.03 |  1.401  |
|            |            |            |       4 |     17.82 ± 0.06 |     23.98 ± 0.11 |  1.346  |
|            |            |            |       8 |     26.63 ± 0.41 |     29.86 ± 0.04 |  1.121  |
| 8B Q4_1    |   4.77 GiB | AVX2       |       2 |      5.11 ± 0.00 |     11.45 ± 0.00 |  2.241  |
|            |            |            |       4 |      9.08 ± 0.02 |     12.58 ± 0.00 |  1.385  |
| 7B Q4_1    |   3.95 GiB | NEON       |       2 |      9.11 ± 0.06 |     14.62 ± 0.04 |  1.605  |
|            |            |            |       4 |     17.04 ± 0.09 |     24.08 ± 0.28 |  1.413  |
|            |            |            |       8 |     25.26 ± 0.24 |     27.23 ± 0.14 |  1.078  |
| 8B Q5_0    |   5.22 GiB | AVX2       |       2 |      5.31 ± 0.01 |      8.30 ± 0.01 |  1.563  |
|            |            |            |       4 |      9.40 ± 0.01 |     11.47 ± 0.00 |  1.220  |
| 7B Q5_0    |   4.34 GiB | NEON       |       2 |      7.26 ± 0.06 |      7.52 ± 0.00 |  1.036  |
|            |            |            |       4 |     13.63 ± 0.18 |     14.16 ± 0.10 |  1.039  |
|            |            |            |       8 |     22.55 ± 0.35 |     24.34 ± 0.22 |  1.079  |
| 8B Q5_1    |   5.64 GiB | AVX2       |       2 |      4.52 ± 0.00 |      8.86 ± 0.00 |  1.960  |
|            |            |            |       4 |      7.72 ± 0.05 |     10.68 ± 0.03 |  1.383  |
| 7B Q5_1    |   4.72 GiB | NEON       |       2 |      6.51 ± 0.01 |      6.42 ± 0.03 |  0.986  |
|            |            |            |       4 |     12.26 ± 0.18 |     12.21 ± 0.14 |  0.996  |
|            |            |            |       8 |     20.33 ± 0.52 |     21.85 ± 0.22 |  1.075  |
| 8B Q2_K_S  |   2.78 GiB | AVX2       |       2 |     11.30 ± 0.00 |     13.06 ± 0.01 |  1.156  |
|            |            |            |       4 |     18.70 ± 0.00 |     19.04 ± 0.65 |  1.014  |
| 7B Q2_K_S  |   2.16 GiB | NEON       |       2 |      8.42 ± 0.05 |     11.97 ± 0.10 |  1.422  |
|            |            |            |       4 |     15.74 ± 0.01 |     22.09 ± 0.08 |  1.403  |
|            |            |            |       8 |     27.35 ± 0.05 |     38.32 ± 0.05 |  1.401  |
| 8B Q3_K_S  |   3.41 GiB | AVX2       |       2 |      8.58 ± 0.00 |     10.82 ± 0.00 |  1.261  |
|            |            |            |       4 |     15.26 ± 0.01 |     16.25 ± 0.01 |  1.065  |
| 7B Q3_K_S  |   2.75 GiB | NEON       |       2 |      6.40 ± 0.02 |      9.12 ± 0.09 |  1.425  |
|            |            |            |       4 |     12.17 ± 0.00 |     17.11 ± 0.03 |  1.406  |
|            |            |            |       8 |     22.04 ± 0.08 |     31.39 ± 0.31 |  1.424  |
| 8B Q4_K_S  |   4.36 GiB | AVX2       |       2 |      9.61 ± 0.00 |     10.72 ± 0.01 |  1.116  |
|            |            |            |       4 |     13.24 ± 0.31 |     13.28 ± 0.01 |  1.003  |
| 7B Q4_K_S  |   3.59 GiB | NEON       |       2 |     11.15 ± 0.05 |     12.93 ± 0.09 |  1.160  |
|            |            |            |       4 |     20.24 ± 0.16 |     23.49 ± 0.29 |  1.161  |
|            |            |            |       8 |     25.76 ± 0.07 |     28.31 ± 0.22 |  1.099  |
| 8B Q5_K_S  |   5.21 GiB | AVX2       |       2 |      7.45 ± 0.00 |      9.73 ± 0.00 |  1.306  |
|            |            |            |       4 |     11.05 ± 0.33 |     11.43 ± 0.02 |  1.034  |
| 7B Q5_K_S  |   4.33 GiB | NEON       |       2 |      7.20 ± 0.04 |      8.81 ± 0.04 |  1.224  |
|            |            |            |       4 |     13.62 ± 0.15 |     16.81 ± 0.16 |  1.234  |
|            |            |            |       8 |     20.56 ± 0.19 |     23.96 ± 0.14 |  1.165  |
| 8B Q6_K    |   6.14 GiB | AVX2       |       2 |      7.53 ± 0.00 |      9.42 ± 0.00 |  1.251  |
|            |            |            |       4 |      9.74 ± 0.00 |      9.97 ± 0.01 |  1.024  |
| 7B Q6_K    |   5.15 GiB | NEON       |       2 |      6.85 ± 0.04 |      8.30 ± 0.06 |  1.212  |
|            |            |            |       4 |     13.03 ± 0.05 |     15.47 ± 0.17 |  1.187  |
|            |            |            |       8 |     18.52 ± 0.07 |     20.67 ± 0.08 |  1.116  |
| 8B IQ2_XXS |   2.23 GiB | AVX2       |       2 |      5.33 ± 0.01 |      6.40 ± 0.00 |  1.201  |
|            |            |            |       4 |     10.06 ± 0.03 |     11.76 ± 0.03 |  1.169  |
| 7B IQ2_XXS |   1.73 GiB | NEON       |       2 |      5.07 ± 0.04 |      5.22 ± 0.05 |  1.030  |
|            |            |            |       4 |      9.63 ± 0.00 |      9.91 ± 0.07 |  1.029  |
|            |            |            |       8 |     17.40 ± 0.50 |     18.65 ± 0.22 |  1.072  |
| 8B IQ2_XS  |   2.42 GiB | AVX2       |       2 |      5.83 ± 0.00 |      6.55 ± 0.00 |  1.123  |
|            |            |            |       4 |     10.88 ± 0.09 |     12.07 ± 0.07 |  1.109  |
| 7B IQ2_XS  |   1.89 GiB | NEON       |       2 |      5.52 ± 0.01 |      5.60 ± 0.00 |  1.014  |
|            |            |            |       4 |     10.50 ± 0.01 |     11.15 ± 0.00 |  1.062  |
|            |            |            |       8 |     18.19 ± 1.30 |     20.94 ± 0.19 |  1.151  |
| 8B IQ2_M   |   2.74 GiB | AVX2       |       2 |      5.12 ± 0.01 |      5.17 ± 0.00 |  1.010  |
|            |            |            |       4 |      9.60 ± 0.28 |      9.68 ± 0.16 |  1.008  |
| 7B IQ2_M   |   2.20 GiB | NEON       |       2 |      3.73 ± 0.02 |      4.53 ± 0.00 |  1.214  |
|            |            |            |       4 |      7.14 ± 0.05 |      8.70 ± 0.06 |  1.218  |
|            |            |            |       8 |     11.99 ± 0.48 |     16.41 ± 0.05 |  1.369  |
| 8B IQ3_XXS |   3.04 GiB | AVX2       |       2 |      4.06 ± 0.01 |      5.00 ± 0.00 |  1.232  |
|            |            |            |       4 |      7.75 ± 0.02 |      9.13 ± 0.45 |  1.178  |
| 7B IQ3_XXS |   2.41 GiB | NEON       |       2 |      3.53 ± 0.00 |      3.82 ± 0.00 |  1.082  |
|            |            |            |       4 |      6.74 ± 0.04 |      7.42 ± 0.07 |  1.103  |
|            |            |            |       8 |     11.96 ± 0.40 |     13.19 ± 0.29 |  1.103  |
| 8B IQ3_S   |   3.42 GiB | AVX2       |       2 |      3.62 ± 0.00 |      4.06 ± 0.00 |  1.122  |
|            |            |            |       4 |      6.80 ± 0.01 |      7.62 ± 0.10 |  1.121  |
| 7B IQ3_S   |   2.75 GiB | NEON       |       2 |      2.96 ± 0.01 |      3.21 ± 0.03 |  1.084  |
|            |            |            |       4 |      5.68 ± 0.01 |      6.25 ± 0.05 |  1.100  |
|            |            |            |       8 |     10.32 ± 0.25 |     11.11 ± 0.37 |  1.077  |
| 8B IQ4_XS  |   4.13 GiB | AVX2       |       2 |      8.08 ± 0.00 |     11.35 ± 0.00 |  1.405  |
|            |            |            |       4 |     13.36 ± 0.72 |     14.32 ± 0.24 |  1.072  |
| 7B IQ4_XS  |   3.37 GiB | NEON       |       2 |      9.87 ± 0.03 |     12.06 ± 0.00 |  1.222  |
|            |            |            |       4 |     17.78 ± 0.23 |     22.06 ± 0.28 |  1.241  |
|            |            |            |       8 |     27.62 ± 0.09 |     29.70 ± 0.39 |  1.075  |
| 8B IQ4_NL  |   4.35 GiB | AVX2       |       2 |      5.52 ± 0.00 |     10.26 ± 0.00 |  1.859  |
|            |            |            |       4 |     10.78 ± 0.01 |     13.69 ± 0.08 |  1.270  |
| 7B IQ4_NL  |   3.56 GiB | NEON       |       2 |      8.32 ± 0.01 |     13.54 ± 0.01 |  1.627  |
|            |            |            |       4 |     15.89 ± 0.00 |     24.28 ± 0.29 |  1.528  |
|            |            |            |       8 |     26.56 ± 0.36 |     29.87 ± 0.08 |  1.125  |

Here gains are generally lower compared to PP due to TG performance being limited by memory bandwidth. Nevertheless, for some quants/architectures/threads the speedup is quite remarkable (e.g., almost a factor of 2 for `Q5_1` on `AVX2` with 2 threads).  

## MoE models

There is [PR-6840](https://github.com/ggerganov/llama.cpp/pull/6840) from Justine Tunney in `llama.cpp`, but it has not been merged since April 23, so I'll compare performance to the master branch for Mixtral-8x7B. As Mixtral8x7B quantization is quite a lengthy process, the following table shows data only for `Q4_K_S` (a commonly used k-quant, 4 bit), `Q5_0` (a legacy quant, 5 bit), and `IQ4_XXS` (a 3-bit i-quant)

| model        |       size | backend    | threads |     test |  t/s (llama.cpp) | t/s (iqk_mul_mat)| Speedup |
| ------------ | ---------: | ---------- | ------: | -------: | ---------------: | ---------------: | ------: |
| 8x7B Q4_K_S  |  48.75 GiB | AVX2       |      16 |    pp512 |     54.92 ± 0.23 |    102.94 ± 0.37 |  1.874  |
|              |            | NEON       |       8 |    pp512 |     23.54 ± 1.56 |     38.32 ± 0.54 |  1.628  |
|              |            | AVX2       |       4 |    tg128 |      7.80 ± 0.07 |      7.83 ± 0.09 |  1.004  |
|              |            | NEON       |       8 |    tg128 |     14.95 ± 0.25 |     15.28 ± 0.24 |  2.022  |
| 8x7B IQ3_XXS |  33.07 GiB | AVX2       |      16 |    pp512 |     17.58 ± 0.04 |     68.45 ± 0.22 |  3.894  |
|              |            | NEON       |       8 |    pp512 |      7.75 ± 0.04 |     34.67 ± 0.40 |  4.474  |
|              |            | AVX2       |       4 |    tg128 |      4.60 ± 0.01 |      5.45 ± 0.09 |  1.185  |
|              |            | AVX2       |       8 |    tg128 |      8.04 ± 0.65 |      9.83 ± 0.06 |  1.223  |
|              |            | AVX2       |      16 |    tg128 |     10.42 ± 0.01 |     10.57 ± 0.01 |  1.014  |
|              |            | NEON       |       8 |    tg128 |      6.19 ± 1.16 |      7.27 ± 0.14 |  1.174  |
| 8x7B Q5_0    |  59.11 GiB | AVX2       |      16 |    pp512 |     29.06 ± 0.43 |     62.67 ± 0.32 |  2.157  |
|              |            | NEON       |       8 |    pp512 |     15.17 ± 0.51 |     27.36 ± 1.03 |  1.804  |
|              |            | AVX2       |       4 |    tg128 |      5.44 ± 0.10 |      6.81 ± 0.06 |  1.252  |
|              |            | NEON       |       8 |    tg128 |     12.03 ± 0.77 |     12.41 ± 1.27 |  1.032  |


## Bitnet-1.58B

Two implementations are provided
* `IQ1_BN` - uses 1.625 bits-per-weight (bpw)
* `IQ2_BN` - uses 2.0 bpw

`IQ2_BN` is faster for PP (CPU and GPU, although the PP performance difference on CUDA is very minor). `IQ1_BN` can arrive at a higher TG performance on the Ryzen-7950X (given enough threads) because of the smaller model size, but it is always slower on the GPU and on the M2-Max CPU.

There is the unmerged [PR 8151](https://github.com/ggerganov/llama.cpp/pull/8151) in `llama.cpp` that implements Bitnet-1.58B for the CPU (`AVX` and `ARM_NEON`, no GPU implementation). The following table compares performance between this repo and `PR-8151` in `llama.cpp`. The CUDA results were obtained on an RTX-4080, the Metal results on a 30-core M2-Max GPU.

| model       |       size | backend    | threads |   test | t/s (llama.cpp)  | t/s (this repo)| Speedup |
| ----------- | ---------: | ---------- | ------: | -----: | ---------------: | -------------: | ------: |
| 3B - IQ1_BN | 729.64 MiB | AVX2       |      16 |  pp512 |    120.61 ± 0.48 | 423.19 ± 1.28  |  3.509  |
|             |            | NEON       |       8 |  pp512 |     46.64 ± 0.02 | 205.90 ± 0.88  |  4.415  |
|             |            | CUDA       |       8 |  pp512 |           -      | 10660 ± 170    |    -    |
|             |            | Metal      |       8 |  pp512 |           -      | 698.25 ± 1.91  |    -    |
|             |            | AVX2       |       2 |  tg128 |     15.79 ± 0.01 |  22.13 ± 0.02  |  1.402  |
|             |            | AVX2       |       4 |  tg128 |     28.64 ± 1.72 |  40.14 ± 0.04  |  1.402  |
|             |            | AVX2       |       8 |  tg128 |     48.91 ± 0.08 |  61.79 ± 0.09  |  1.263  |
|             |            | AVX2       |      16 |  tg128 |     57.73 ± 0.05 |  60.79 ± 0.05  |  1.053  |
|             |            | NEON       |       2 |  tg128 |     11.43 ± 0.04 |  16.87 ± 0.02  |  1.476  |
|             |            | NEON       |       4 |  tg128 |     21.11 ± 0.05 |  30.66 ± 0.11  |  1.452  |
|             |            | NEON       |       8 |  tg128 |     37.36 ± 0.07 |  55.21 ± 0.16  |  1.478  |
|             |            | CUDA       |       8 |  tg128 |           -      | 301.44 ± 0.12  |    -    |
|             |            | Metal      |       8 |  tg128 |           -      |  76.70 ± 0.07  |    -    |
| 3B - IQ2_BN | 873.65 MiB | AVX2       |      16 |  pp512 |    151.39 ± 0.35 | 540.82 ± 2.48  |  3.572  |
|             |            | NEON       |       8 |  pp512 |     46.54 ± 0.03 | 242.05 ± 0.34  |  5.201  |
|             |            | CUDA       |       8 |  pp512 |           -      | 10800 ± 160    |    -    |
|             |            | Metal      |       8 |  pp512 |           -      | 723.19 ± 0.53  |    -    |
|             |            | AVX2       |       2 |  tg128 |     18.93 ± 0.02 |  38.34 ± 0.08  |  2.026  |
|             |            | AVX2       |       4 |  tg128 |     34.54 ± 0.06 |  56.29 ± 0.07  |  1.630  |
|             |            | AVX2       |       8 |  tg128 |     52.97 ± 0.07 |  53.44 ± 0.08  |  1.009  |
|             |            | AVX2       |      16 |  tg128 |     51.84 ± 0.25 |  53.46 ± 0.07  |  1.031  |
|             |            | NEON       |       2 |  tg128 |     11.40 ± 0.02 |  32.01 ± 0.27  |  2.808  |
|             |            | NEON       |       4 |  tg128 |     20.99 ± 0.00 |  56.45 ± 0.11  |  2.689  |
|             |            | NEON       |       8 |  tg128 |     37.28 ± 0.08 |  89.77 ± 0.70  |  2.408  |
|             |            | CUDA       |       8 |  tg128 |           -      | 322.10 ± 0.07  |    -    |
|             |            | Metal      |       8 |  tg128 |           -      | 110.39 ± 0.13  |    -    |

We can make the following observations:
* For prompt processing this Bitnet-1.58b implementation is massively better than PR-8151 in `llama.cpp`, with gains between 3.4X and 5.2X!
* We get `PP-512 = 520 t/s` for the 2.0 bpw variant on the Ryzen-7950X, which costs less than $500. Hey, who needs a GPU?  
* For low number of threads (2), this implementation is also much faster than PR-8151 for TG, where speed gains are between 1.4X and 2.8X. As we become memory bound on the Ryzen-7950X, the speed advantage goes away there for sufficiently high number of threads. But on the M2-Max this implementation is 1.4X (1.625 bpw) or 2.4X faster even at 8 threads
* Looking at TG on the M2-Max, the GPU looks a bit like wasted silicon (90 vs 110 t/s for TG-128 and the 2.0 bpw variant). If the GPU transistors had been spent to double the M2 number of CPU cores (and all memory bandwidth is given to the CPU), the CPU would be wiping the floor with the GPU.
* I'm of course kidding with the above. Still, it seems there are massive inefficiencies in the `llama.cpp` Metal implementation that start showing up when matrix multiplications become very fast as is the case here. The difference between CPU and GPU prompt processing speed is typically at least a factor of 7 in favor of the GPU on the M2-Max, but it is only around a factor of 3 here.
* It is worth noting that one needs to offload the token embeddings tensor to the GPU, else performance on CUDA/Metal is significantly lower. Bitnet uses the same tensor for token embeddings and for output. Mainline `llama.cpp` currently puts the token embeddings tensor on the CPU, and this results in running the matrix multiplication with the output tensor on the CPU. This most likely affects other models as well (e.g., Gemma), but I haven't yet looked into this.

To reproduce these results:
* Clone https://huggingface.co/1bitLLM/bitnet_b1_58-3B
* Run `python3 --outtype f16 path_to_bitnet` to convert to GGUF
* Run `./bin/llama-quantize path_to_bitnet/ggml-model-f16.gguf quantized.gguf [iq1_bn | iq2_bn]`. Note: no imatrix is required (and, if you provide one, it is ignored)
* Caveat: only the 3B Bitnet variant works. The smaller Bitnet models contain tensors with number of columns that are not even a multiple of 32, so basically no `llama.cpp` quant will work for these.  

## To tile or not to tile

The common wisdom for efficient matrix multiplications is to use block tiling, and this is also used here for `fp16/fp32` matrices. But block tiling does not somehow magically reduce the amount of computation that needs to get done. Performance gains are simply due to the better utilization of memory caches. When dealing with quantized matrix multiplications, there is an additional factor that comes into play: the quantized data needs to be unpacked to 8-bit integers before being used in the matrix multiplication multiply-add operations. Depending on quantization type, this unpacking can represent a significant fraction of the overall computation cost. Hence, for best performance, one would want to reuse the unpacked quants as much as possible, thus spending some fraction of the available vector registers to hold the unpacked data. But when using block tiling, one also needs a certain number of vector registers for accumulating results. For instance, on `AVX2` (16 vector registers available), for `fp16/fp32` models best performance is achieved with `2 x 6` tiles (where the `2` refers to rows in the left matrix and is measured in units of the vector register size, so 16/8 floats for `fp16/fp32`, and `6` is for the number of columns in the right matrix). Unpacking quantized data works best when done in blocks of 128 or 256 quants so that, if we wanted to keep unpacked quants for 2 rows, we would need at least 8 vector registers, thus being left with less than 8 registers for result accumulation, so at best `2 x 3` tiles. In practice one needs addition vector registers for various constants that are typically needed for de-quantization, so that, at the end, it becomes better to use `1 x N` "tiles", i.e., a row-wise multiplication where each row in the left matrix is multiplied with `N` columns in the right matrix, thus reusing the unpacked data `N` times. This (i.e., amortizing de-quantization cost) is the main mechanism for seeding up quantized matrix multiplications. Having started with quantized matrices, and having gone from tiles to a row-wise implementation after some experimentation, I did try row-wise multiplication for float matrices first. Performance was not quite as good as for block-tiling, but I did get up to 90-95% of the speed of `tinyBLAS` that way before switching the `fp16/fp32` implementation to `2 x 6` (`AVX2`) or `5 x 5` (`AVX512` and `ARM_NEON`) block-tiles. But even for for `Q8_0 x Q8_0` multiplications, where there is basically no de-quantization cost, row-wise multiplication is faster than tiling (and hence this implemeintation beats `tinyBLAS`, which uses block-tiling also for `Q8_0`).

