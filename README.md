# Croco.Cpp (CCPP), a KoboldCPP mod - Esobold Powered version - Readme to be updated :

<details>
<summary>Unroll DISCLAIMER:</summary>

Croco.Cpp is a fork/mod of KoboldCPP, already known as KoboldCPP Frankenstein, Frankenfork, or shortened in KCPP-F.
The namechange is due to my boredom with the Frankenstein marker I myself initiated a year ago.
As usual, the Croco.Cpp builds are NOT supported by the KoboldCPP (KCPP) team, Github, or Discord channel.
They are for greedy-test and amusement only.
Any potential support found them is a courtesy, not a due.
My CCPP version number bumps as soon as the version number in the official experimental branch bumps in the following way x.xxx (ex : 1.80.1) : (KCPP)x.xxx.(CCPP)xx.
They are not "upgrades" over the official version. And they might be bugged at time: only the official KCPP releases are to be considered correctly numbered, reliable and "fixed".
The LllamaCPP version + the additional PRs integrated follow my CCPP versioning in the title, so everybody knows what version they deal with.
Important : New models sometimes integrated in my builds (like recently Mistral Nemo, which posed problems for several users) are for personal testing only, and CAN'T be fixed if they fail because their support come from third party PRs coming from LlamaCPP merged "savagely" in my builds, sometimes before even being merged on LlamaCPP master.
</details>

Presentation :

Croco.Cpp (CCPP) is a fork/mod of the experimental branch of KoboldCPP (KCPP) enhanced with Esobold, mainly aimed at NVidia Cuda users (I'm myself using Ampere GPUs and it doesn't support the other backends as of now except a CPU failsafe for the mainline Llama.cpp quants (for now) since v1.93040.. it might also support Hipblas/ROCm, but it's not tested for a year now..), this mod having a few modifications accordingly to my own needs, among which :
- A more cluttered GUI that I had to enlarge to put all my mess.
- More context steps in GUI, as well as more Blas Batch Size (supports MMVQ 1-8 for example).
- Physical Blas Batch Size Exposed and configurable (might be obsolete now).
- 22 or so different modes of quantization for the context cache (F16, around 15 KV modes with Flash Attention, BF16, 7 quantum legacy K cache modes without Flash Attention for models like Gemma).
- KV cache supports IQ4_NL and Q6_0 (except for Gemma), thanks to Ikawrakow. IQ4_NL gives 2% perplexity gain over q4_0, and q6_0 0.1-0.2% over Q5_1.
- Configurable KV cache for the draft model in case of speculative decoding.
- Shrunk Blas Batch size on the draft model compared to the main model BBS, due to the draft's logically smaller size and thus higher PP.
- Supports inference for BF16 models in Cuda (thanks Justine Tuney, Ikawrakow, and Johannes Gaessler).
- Supports inference for the IQ_K quants (first generation) made by Ikawrakow (Q6_0 legacy for irregularly shaped tensors ; IQ_2K, 3K, 4K, 5K, 6K).
- Supports in Cuda MMQ Mode : Ikawrakow's IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ6_K, IQ2_KS, IQ4_KS, IQ5_KS, IQ4_KS_R4, IQ5_KS_R4. WIP : IQ1_S_R4.
- Supports in Cuda Cublas mode (including those supported in MMQ as well) : Ikawrakow's IQ4_KSS, and the unmerged IQ3_KS. WIP: IQ1_S_R4, IQ1_M_R4, IQ2_K_R4, IQ3_K_R4, IQ4_K_R4, IQ5_K_R4 (the last 4 probably work).
- A dozen or so commits taken from Ikawrakow's IK_Llama.CPP for performances or quant quality. That includes a few more GGML ops.
- The NXS_Llama.cpp tool in order to quantize with Ikawrakow's new quants but with a mainline base. 
- 64 Stories slots instead of 10 or so in the web-interface (KLite).
- More infos displayed in the CLI, without activating debug mode.
- Smartcontext instead of contextshift by default in GUI for compatibility with Gemma.
- Support the edition of NORM_EPS_RMS value to improve the inference of 1.5bpw to 2bpw quants.
- More logging out of debug mode.
- A --contextshift parameter replaces the -noshift parameter (I'm confused by negative parameters opposing another positive parameter, in our case --smartcontext).
- More recent dependencies (that's nerdy, isn't it?).

- NEED TO BE FIXED : A slightly different benchmark (one flag per column instead of a single flag space).
- NEED TO BE FIXED : Since 1.71010, an enhanced model layers autoloader on GPU (which is less and less cluttered and bugged lol), based on Concedo's code and Pyroserenus formulas, but different from Henky's subsequent commit on KCPP-official. It's compatible with KV_Quants, accounts for FA, MMQ, LowVram, works in single and multi-GPU (up to 16?), is accessible in CLI and GUI modes, and can be configured easily in tandem with tensor split for an entirely customized loading accordingly to one's rig and needs.
- NEED TO BE FIXED : Support to Ikawrakow's IQ2_KT, 3_KT, 4_KT quants (Trellis, usually working with a restored DMMV kernel).

- BROKEN : - EmphasisFSM by Yoshku to handle the "" and ** formatting in KCPP and SillyTavern (mostly, if you have troubles of chat (thoughts, actions, dialogues) formatting, and anti-slop doesn't cut it for your needs somehow).

Recommanded settings for Commande Line Interface / GUI :
```
--flashattention (except for Gemma?)
--blastbatchsize 128 (256 for Gemma)
--usecublas mmq (for NVidia users, MMQ mode is faster)
--contexshift (to avoid to reprocess the whole context once it's full. Might not be compatible with K/V cache q6_0)
```
Check the help section (koboldcpp.exe --help or python koboldcpp.py --help) for more infos.

## Croco.Cpp specifics :

<details>
<summary>Unroll the 26 KV cache options (all should be considered experimental except F16, KV Q8_0, and KV Q4_0)</summary>

With Flash Attention :
- F16 -> Fullproof (the usual KV quant since the beginning of LCPP/KCPP)
- BF16 (experimental)
- K F16 with : V Q8_0, Q6_0 (experimental), Q5_0, iq4_nl.
- K Q8_0 with : V Q8_0 (stable, part of the LCPP/KCPP main triplet), Q6_0 (experimental),  Q5_0 (maybe unstable), iq4_nl (maybe stable).
- K Q6_0 with : V Q6_0, Q5_0, iq4_nl.
- K Q5_1 with : V Q5_0, iq4_nl.
- K Q5_0 with : V iq4_nl.
- KV Q4_0 (quite stable, if we consider that it's part of the LCPP/KCPP main triplet)
Works in command line, normally also via the GUI, and normally saves on .KCPPS config files.
- KV iq4_nl (with -2% perplexity compared to Q4_0).

Without Flash Attention nor MMQ (for models like Gemma) :
- V F16 with K Q8_0, Q5_1, Q5_0, Q4_1, and Q4_0.
- K Q6_0 and IQ4_NL to be tested, might not work.
</details>

<details>
<summary>Unroll the options to set KV Quants (obsolete)</summary>

KCPP official KV quantized modes (modes 1 and 2 require Flash Attention) :

0 = 1616/F16 (16 BPW),
1 = FA8080/KVq8_0 (8.5 BPW),
2 = FA4040/KVq4_0 (4.5BPW),

CCPP unofficial KV quantized modes (require flash attention) :

    "1 - q8_0 - (8.5BPW) - FA",
    "2 - q4_0 - (4.5BPW) - FA - possibly faulty on some models",
    "3* - K F16 - V q8_0 (12.25BPW) - FA",
    "4* - K F16 - V q6_0 (11.25BPW) - FA. Doesn't work on Gemma 2 FA.",   
    "5 - K q8_0 - V q6_0 (7.5BPW) - FA. Doesn't work on Gemma 2 FA.",
    "6* - K q8_0 - V q5_0 (7BPW) - FA",
    "7 - K q8_0 - V iq4_nl (6.5BPW) - FA. Doesn't work on Gemma 2 FA.",
    "8* - K q6_0 - V q6_0 (6.5BPW) - FA. Doesn't work on Gemma 2 FA.",
    "9 - K q6_0 - V q5_0 (6BPW) - FA, best game in FA town. Doesn't work on Gemma 2 FA.",
    "10* - K q6_0 - V iq4_nl (5.5BPW) - FA - faulty on some models (Gemma 2 FA. Qwen 2.5 1.5b?)",
    "11 - K q5_1 - V q5_0 (5.5BPW) - FA - possibly faulty on some models (Qwen 2.5 1.5b?)",
    "12* - K q5_1 - V iq4_nl (5.25BPW) - FA",
    "13 - K q5_0 - V iq4_nl (5BPW) - FA - possibly faulty on some models (Qwen 2.5 1.5b?)",
    "14 - K iq4_nl - V iq4_nl (4.5BPW) - FA",
    "15 - BF16 (16BPW) - no FA, experimental for Cuda, not tested on other backends.",
    "16 - K q8_0 - V F16 (12.25BPW) - NO FA, slower",
    "17 - K q6_0 - V F16 (11.25BPW) - NO FA, slower, best game in non-FA town.",
    "18 - K q5_1 - V F16 (11BPW) - NO FA, slower - possibly faulty on some models (Qwen 2.5 1.5b?)",
    "19 - K q5_0 - V F16 (11.75BPW) - NO FA, slower - possibly faulty on some models (Qwen 2.5 1.5b?)",
    "20 - K q4_1 - V F16 (10.5BPW) - NO FA, slower - possibly faulty on some models (Qwen 2.5 1.5b?)",
    "21 - K q4-0 - V F16 (10.25BPW) - NO FA, slower - possibly faulty on some models (Qwen 2.5 1.5b?)",
    "22 - K iq4_nl - V F16 (10.25BPW) - NO FA, slower"]

choices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22], default=0)
</details>

<details>
<summary>Unroll the details about Emphasisfsm by Yoshqu</summary>

The common problem during text generiation are misplaced emphasis characters.

    *looks at you "why* this is here?"

while it should be

    *looks at you* "why this is here?"

This emphasisfsm solves this by simple (and fast) grammar expressed by deterministic finite state machine.

![Letters](emphasis-dfsm-letters.png)

Single letters are not practical in LLMs as tokens often contains more than one.

Emphasisfsm uses LLM tokens as its alphabet making it very fast.

![Tokens](emphasis-dfsm-tokens.png)

Those are only most obvious examples. There are more, eg. ' "***' is a valid token to transition from qout to star. and '*this' is vaild for quot->none or none->quot.

### Usage

To support variety of GUIs this extension shamefully exploits GBNF grammar string. *This is not a proper GBNF grammar, it only uses the field which is easily editable in most GUIs*

![KoboldCpp hack](gbnf-kob.png) ![SillyTavern hack](gbnf-st.png)

    emphasisfsm "_bias_[D][_emph1_][,_emphn_]"

Empty string emphasisfsm is disabled. The easiest way to enable is to

    emphasisfsm "-20"

which defaults to

    emphasisfsm "-20 \" \" * *"

(no debug, only * and " are considered)


### how it works

Main loop is extended from:

- retrieve logits
- sample logits, select token (top_k and friends)
- output token

to

- retrieve logits
- ban forbidden emphasisfsm transitions from current state (stetting their logits low)
- sample logits, select token (top_k and friends)
- emphasisfsm trasition on selected token
- output token


### TODO

- find split utf8 letters over more than one token (i don't plant to support it, but warning would be nice)
- banning end tokens generation inside of emphasis - forcing LLM to finsh his 'thought' ?


### Meta-Llama-3-8B stats for default (" *) emphasisfsm

    empcats_gen: ban bias: -17.500000
    empcats_gen: emphasis indifferent tokens: 126802
    empcats_gen: tokens for emphasis '"' '"': 1137
    empcats_gen: tokens for emphasis '*' '*': 315
    empcats_gen: always banned tokens: 2
    empcats_gen: total tokens: 128256

Always banned tokens are :

<pre>' "*"',  ' "*"'</pre>

### Tests

    emphasisfsm "-20 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 0 0"

This forces that every digit is a citation, so example text completion looks like:


```
Give me math vector of random numbers.Here is a 3-dimensional math vector with random numbers:
Vector:
[
    3.445,
    -5.117,
    7.992
]
```

There is no other digit between two 3, two 4, two 5 and so on....
</details>

<details>
<summary>Unroll the Dry sampler recommanded settings</summary>

Multiplier : 0.8
Base : 1.75
Allowed length : 2
Range : Uses the repetition penalty range (usual parameter is 2048)
Usual sequence breakers : '\n', ':', '"', '*'
</details>

#Croco.Cpp notes :

- I often mislabel the Cuda specifics of the builds. Here's the right nomenclature :
Cuda 12.2 arch 60617075 : Cu12.2_SMC2_Ar60617075_DmmvX64Y2_MMY2_KQIt2
Cuda 12.1 arch 52617075 : CuCML_ArCML_SMC2_DmmvX32Y1 (CML : CMakeList)
Cuda 11.4.4/11.5 arch 35375052 : CuCML_ArCML_SMC2_DmmvX32Y1

# koboldcpp-experimental

KoboldCpp-experimental is a sligthly extended KoboldCpp with [custom](experimental/README.md) functionality.


-----

Croco is based on KoboldCPP

# koboldcpp (for information)

KoboldCpp is an easy-to-use AI text-generation software for GGML and GGUF models, inspired by the original **KoboldAI**. It's a single self-contained distributable that builds off **llama.cpp** and adds many additional powerful features.

![Preview](media/preview.png)
![Preview](media/preview2.png)
![Preview](media/preview3.png)
![Preview](media/preview4.png)
![Preview](media/preview5.png)
![Preview](media/preview6.png)

### Features
- Single file executable, with no installation required and no external dependencies
- Runs on CPU or GPU, supports full or partial offloaded
- LLM text generation (Supports all GGML and GGUF models, backwards compatibility with ALL past models)
- Image Generation (Stable Diffusion 1.5, SDXL, SD3, Flux)
- Speech-To-Text (Voice Recognition) via Whisper
- Text-To-Speech (Voice Generation) via OuteTTS
- Provides many compatible APIs endpoints for many popular webservices (KoboldCppApi OpenAiApi OllamaApi A1111ForgeApi ComfyUiApi WhisperTranscribeApi XttsApi OpenAiSpeechApi)
- Bundled KoboldAI Lite UI with editing tools, save formats, memory, world info, author's note, characters, scenarios.
- Includes multiple modes (chat, adventure, instruct, storywriter) and UI Themes (aesthetic roleplay, classic writer, corporate assistant, messsenger)
- Supports loading Tavern Character Cards, importing many different data formats from various sites, reading or exporting JSON savefiles and persistent stories.
- Many other features including new samplers, regex support, websearch, RAG via TextDB and more.
- Ready-to-use binaries for Windows, MacOS, Linux. Runs directly with Colab, Docker, also supports other platforms if self-compiled (like  Android (via Termux) and Raspberry PI).
- [Need help finding a model? Read this!](https://github.com/LostRuins/koboldcpp/wiki#getting-an-ai-model-file)

## Windows Usage (Precompiled Binary, Recommended)
- Windows binaries are provided in the form of **koboldcpp.exe**, which is a pyinstaller wrapper containing all necessary files. **[Download the latest koboldcpp.exe release here](https://github.com/LostRuins/koboldcpp/releases/latest)**
- To run, simply execute **koboldcpp.exe**.
- Launching with no command line arguments displays a GUI containing a subset of configurable settings. Generally you dont have to change much besides the `Presets` and `GPU Layers`. Read the `--help` for more info about each settings.
- Obtain and load a GGUF model. See [here](#Obtaining-a-GGUF-model)
- By default, you can connect to http://localhost:5001
- You can also run it using the command line. For info, please check `koboldcpp.exe --help`

## Linux Usage (Precompiled Binary, Recommended)
On modern Linux systems, you should download the `koboldcpp-linux-x64` prebuilt PyInstaller binary on the **[releases page](https://github.com/LostRuins/koboldcpp/releases/latest)**. Simply download and run the binary (You may have to `chmod +x` it first). If you have an older device, you can also try the `koboldcpp-linux-x64-oldpc` instead for greatest compatibility.

Alternatively, you can also install koboldcpp to the current directory by running the following terminal command:
```
curl -fLo koboldcpp https://github.com/LostRuins/koboldcpp/releases/latest/download/koboldcpp-linux-x64-oldpc && chmod +x koboldcpp
```
After running this command you can launch Koboldcpp from the current directory using `./koboldcpp` in the terminal (for CLI usage, run with `--help`).
Finally, obtain and load a GGUF model. See [here](#Obtaining-a-GGUF-model)

## MacOS (Precompiled Binary)
- PyInstaller binaries for Modern ARM64 MacOS (M1, M2, M3) are now available! **[Simply download the MacOS binary](https://github.com/LostRuins/koboldcpp/releases/latest)**
- In a MacOS terminal window, set the file to executable `chmod +x koboldcpp-mac-arm64` and run it with `./koboldcpp-mac-arm64`.
- In newer MacOS you may also have to whitelist it in security settings if it's blocked. [Here's a video guide](https://youtube.com/watch?v=NOW5dyA_JgY).
- Alternatively, or for older x86 MacOS computers, you can clone the repo and compile from source code, see Compiling for MacOS below.
- Finally, obtain and load a GGUF model. See [here](#Obtaining-a-GGUF-model)

## Run on Colab
- KoboldCpp now has an **official Colab GPU Notebook**! This is an easy way to get started without installing anything in a minute or two. [Try it here!](https://colab.research.google.com/github/LostRuins/koboldcpp/blob/concedo/colab.ipynb).
- Note that KoboldCpp is not responsible for your usage of this Colab Notebook, you should ensure that your own usage complies with Google Colab's terms of use.

## Run on RunPod
- KoboldCpp can now be used on RunPod cloud GPUs! This is an easy way to get started without installing anything in a minute or two, and is very scalable, capable of running 70B+ models at afforable cost. [Try our RunPod image here!](https://koboldai.org/runpodcpp).

## Run on Novita AI
KoboldCpp can now also be run on Novita AI, a newer alternative GPU cloud provider which has a quick launch KoboldCpp template for as well. [Check it out here!](https://koboldai.org/novitacpp)

## Docker
- The official docker can be found at https://hub.docker.com/r/koboldai/koboldcpp
- If you're building your own docker, remember to enable LLAMA_PORTABLE

## Obtaining a GGUF model
- KoboldCpp uses GGUF models. They are not included with KoboldCpp, but you can download GGUF files from other places such as [Bartowski's Huggingface](https://huggingface.co/bartowski). Search for "GGUF" on huggingface.co for plenty of compatible models in the `.gguf` format.
- For beginners, we recommend the models [L3-8B-Stheno-v3.2](https://huggingface.co/bartowski/L3-8B-Stheno-v3.2-GGUF/resolve/main/L3-8B-Stheno-v3.2-Q4_K_S.gguf) (smaller and weaker) or [Tiefighter 13B](https://huggingface.co/KoboldAI/LLaMA2-13B-Tiefighter-GGUF/resolve/main/LLaMA2-13B-Tiefighter.Q4_K_S.gguf) (old but very versatile model) or [Gemma-3-27B Abliterated](https://huggingface.co/mlabonne/gemma-3-27b-it-abliterated-GGUF/resolve/main/gemma-3-27b-it-abliterated.q4_k_m.gguf) (largest and most powerful)
- [Alternatively, you can download the tools to convert models to the GGUF format yourself here](https://kcpptools.concedo.workers.dev). Run `convert-hf-to-gguf.py` to convert them, then `quantize_gguf.exe` to quantize the result.
- Other models for Whisper (speech recognition), Image Generation, Text to Speech or Image Recognition [can be found on the Wiki](https://github.com/LostRuins/koboldcpp/wiki#what-models-does-koboldcpp-support-what-architectures-are-supported)

## Improving Performance
- **GPU Acceleration**: If you're on Windows with an Nvidia GPU you can get CUDA support out of the box using the `--usecuda`  flag (Nvidia Only), or `--usevulkan` (Any GPU), make sure you select the correct .exe with CUDA support.
- **GPU Layer Offloading**: Add `--gpulayers` to offload model layers to the GPU. The more layers you offload to VRAM, the faster generation speed will become. Experiment to determine number of layers to offload, and reduce by a few if you run out of memory.
- **Increasing Context Size**: Use `--contextsize (number)` to increase context size, allowing the model to read more text. Note that you may also need to increase the max context in the KoboldAI Lite UI as well (click and edit the number text field).
- **Old CPU Compatibility**: If you are having crashes or issues, you can try running in a non-avx2 compatibility mode by adding the `--noavx2` flag. You can also try reducing your `--blasbatchssize` (set -1 to avoid batching)

For more information, be sure to run the program with the `--help` flag, or **[check the wiki](https://github.com/LostRuins/koboldcpp/wiki).**

## Compiling KoboldCpp From Source Code

### Compiling on Linux (Using koboldcpp.sh automated compiler script)
when you can't use the precompiled binary directly, we provide an automated build script which uses conda to obtain all dependencies, and generates (from source) a ready-to-use a pyinstaller binary for linux users.
- Clone the repo with `git clone https://github.com/LostRuins/koboldcpp.git`
- Simply execute the build script with `./koboldcpp.sh dist` and run the generated binary. (Not recommended for systems that already have an existing installation of conda. Dependencies: curl, bzip2)
```
./koboldcpp.sh # This launches the GUI for easy configuration and launching (X11 required).
./koboldcpp.sh --help # List all available terminal commands for using Koboldcpp, you can use koboldcpp.sh the same way as our python script and binaries.
./koboldcpp.sh rebuild # Automatically generates a new conda runtime and compiles a fresh copy of the libraries. Do this after updating Koboldcpp to keep everything functional.
./koboldcpp.sh dist # Generate your own precompiled binary (Due to the nature of Linux compiling these will only work on distributions equal or newer than your own.)
```

### Compiling on Linux (Manual Method)
- To compile your binaries from source, clone the repo with `git clone https://github.com/LostRuins/koboldcpp.git`
- A makefile is provided, simply run `make`.
- Optional Vulkan: Link your own install of Vulkan SDK manually with `make LLAMA_VULKAN=1`
- Optional CLBlast: Link your own install of CLBlast manually with `make LLAMA_CLBLAST=1`
- Note: for these you will need to obtain and link OpenCL and CLBlast libraries.
  - For Arch Linux: Install `cblas` and `clblast`.
  - For Debian: Install `libclblast-dev`.
- You can attempt a CuBLAS build with `LLAMA_CUBLAS=1`, (or `LLAMA_HIPBLAS=1` for AMD). You will need CUDA Toolkit installed. Some have also reported success with the CMake file, though that is more for windows.
- For a full featured build (all backends), do `make LLAMA_CLBLAST=1 LLAMA_CUBLAS=1 LLAMA_VULKAN=1`. (Note that `LLAMA_CUBLAS=1` will not work on windows, you need visual studio)
- To make your build sharable and capable of working on other devices, you must use `LLAMA_PORTABLE=1`
- After all binaries are built, you can run the python script with the command `python koboldcpp.py [ggml_model.gguf] [port]`

### Compiling on Windows
- You're encouraged to use the .exe released, but if you want to compile your binaries from source at Windows, the easiest way is:
  - Get the latest release of w64devkit (https://github.com/skeeto/w64devkit). Be sure to use the "vanilla one", not i686 or other different stuff. If you try they will conflit with the precompiled libs!
  - Clone the repo with `git clone https://github.com/LostRuins/koboldcpp.git`
  - Make sure you are using the w64devkit integrated terminal, then run `make` at the KoboldCpp source folder. This will create the .dll files for a pure CPU native build.
  - For a full featured build (all backends), do `make LLAMA_CLBLAST=1 LLAMA_VULKAN=1`. (Note that `LLAMA_CUBLAS=1` will not work on windows, you need visual studio)
  - To make your build sharable and capable of working on other devices, you must use `LLAMA_PORTABLE=1`
  - If you want to generate the .exe file, make sure you have the python module PyInstaller installed with pip (`pip install PyInstaller`). Then run the script `make_pyinstaller.bat`
  - The koboldcpp.exe file will be at your dist folder.
- **Building with CUDA**: Visual Studio, CMake and CUDA Toolkit is required. Clone the repo, then open the CMake file and compile it in Visual Studio. Copy the `koboldcpp_cublas.dll` generated into the same directory as the `koboldcpp.py` file. If you are bundling executables, you may need to include CUDA dynamic libraries (such as `cublasLt64_11.dll` and `cublas64_11.dll`) in order for the executable to work correctly on a different PC.
- **Replacing Libraries (Not Recommended)**: If you wish to use your own version of the additional Windows libraries (OpenCL, CLBlast, Vulkan), you can do it with:
  - OpenCL - tested with https://github.com/KhronosGroup/OpenCL-SDK . If you wish to compile it, follow the repository instructions. You will need vcpkg.
  - CLBlast - tested with https://github.com/CNugteren/CLBlast . If you wish to compile it you will need to reference the OpenCL files. It will only generate the ".lib" file if you compile using MSVC.
  - Move the respectives .lib files to the /lib folder of your project, overwriting the older files.
  - Also, replace the existing versions of the corresponding .dll files located in the project directory root (e.g. clblast.dll).
  - Make the KoboldCpp project using the instructions above.

### Compiling on MacOS
- You can compile your binaries from source. You can clone the repo with `git clone https://github.com/LostRuins/koboldcpp.git`
- A makefile is provided, simply run `make`.
- If you want Metal GPU support, instead run `make LLAMA_METAL=1`, note that MacOS metal libraries need to be installed.
- To make your build sharable and capable of working on other devices, you must use `LLAMA_PORTABLE=1`
- After all binaries are built, you can run the python script with the command `python koboldcpp.py --model [ggml_model.gguf]` (and add `--gpulayers (number of layer)` if you wish to offload layers to GPU).

### Compiling on Android (Termux Installation)
- [First, Install and run Termux from F-Droid](https://f-droid.org/en/packages/com.termux/)
## Termux Quick Setup Script (Easy Setup)
- You can use this auto-installation script to quickly install and build everything and launch KoboldCpp with a model.
Simply run:
```
curl -sSL https://raw.githubusercontent.com/LostRuins/koboldcpp/concedo/android_install.sh | sh
```
and it will install everything required. Alternatively, you can download the above `android_install.sh` script to file, then do `chmod +x` and run it interactively.
## Termux Manual Instructions (DIY Setup)
- Open termux and run the command `apt update`
- Install dependency `apt install openssl`
- Install other dependencies with `pkg install wget git python`
- Run `pkg upgrade`
- Clone the repo `git clone https://github.com/LostRuins/koboldcpp.git`
- Navigate to the koboldcpp folder `cd koboldcpp`
- Build the project `make`
- To make your build sharable and capable of working on other devices, you must use `LLAMA_PORTABLE=1`, this disables usage of ARM instrinsics.
- Grab a small GGUF model, such as `wget https://huggingface.co/concedo/KobbleTinyV2-1.1B-GGUF/resolve/main/KobbleTiny-Q4_K.gguf`
- Start the python server `python koboldcpp.py --model KobbleTiny-Q4_K.gguf`
- Connect to `http://localhost:5001` on your mobile browser
- If you encounter any errors, make sure your packages are up-to-date with `pkg up` and `pkg upgrade`
- If you have trouble installing an dependency, you can try the command `termux-change-repo` and choose a different repo (e.g. `Mirror by BFSU`)
- GPU acceleration for Termux may be possible but I have not explored it. If you find a good cross-device solution, do share or PR it.

## AMD Users
- For most users, you can get very decent speeds by selecting the **Vulkan** option instead, which supports both Nvidia and AMD GPUs.
- Alternatively, you can try the ROCM fork at https://github.com/YellowRoseCx/koboldcpp-rocm though this may be outdated.

## Third Party Resources
- These unofficial resources have been contributed by the community, and may be outdated or unmaintained. No official support will be provided for them!
  - Arch Linux Packages: [CUBLAS](https://aur.archlinux.org/packages/koboldcpp-cuda), and [HIPBLAS](https://aur.archlinux.org/packages/koboldcpp-hipblas).
  - Unofficial Dockers: [korewaChino](https://github.com/korewaChino/koboldCppDocker) and [noneabove1182](https://github.com/noneabove1182/koboldcpp-docker)
  - Nix & NixOS: KoboldCpp is available on Nixpkgs and can be installed by adding just `koboldcpp` to your `environment.systemPackages` *(or it can also be placed in `home.packages`)*.
    - [Example Nix Setup and further information](examples/nix_example.md)
    - If you face any issues with running KoboldCpp on Nix, please open an issue [here](https://github.com/NixOS/nixpkgs/issues/new?assignees=&labels=0.kind%3A+bug&projects=&template=bug_report.md&title=).
- [GPTLocalhost](https://gptlocalhost.com/demo#KoboldCpp) - KoboldCpp is supported by GPTLocalhost, a local Word Add-in for you to use KoboldCpp in Microsoft Word. A local alternative to "Copilot in Word."

## Questions and Help Wiki
- **First, please check out [The KoboldCpp FAQ and Knowledgebase](https://github.com/LostRuins/koboldcpp/wiki) which may already have answers to your questions! Also please search through past issues and discussions.**
- If you cannot find an answer, open an issue on this github, or find us on the [KoboldAI Discord](https://koboldai.org/discord).

## KoboldCpp and KoboldAI API Documentation
- [Documentation for KoboldAI and KoboldCpp endpoints can be found here](https://lite.koboldai.net/koboldcpp_api)

## KoboldCpp Public Demo
- [A public KoboldCpp demo can be found at our Huggingface Space. Please do not abuse it.](https://koboldai-koboldcpp-tiefighter.hf.space/)

## Considerations
- For Windows: No installation, single file executable, (It Just Works)
- Since v1.15, requires CLBlast if enabled, the prebuilt windows binaries are included in this repo. If not found, it will fall back to a mode without CLBlast.
- Since v1.33, you can set the context size to be above what the model supports officially. It does increases perplexity but should still work well below 4096 even on untuned models. (For GPT-NeoX, GPT-J, and Llama models) Customize this with `--ropeconfig`.
- Since v1.42, supports GGUF models for LLAMA and Falcon
- Since v1.55, lcuda paths on Linux are hardcoded and may require manual changes to the makefile if you do not use koboldcpp.sh for the compilation.
- Since v1.60, provides native image generation with StableDiffusion.cpp, you can load any SD1.5 or SDXL .safetensors model and it will provide an A1111 compatible API to use.
- **I try to keep backwards compatibility with ALL past llama.cpp models**. But you are also encouraged to reconvert/update your models if possible for best results.
- Since v1.75, openblas has been deprecated and removed in favor of the native CPU implementation.

## License
- The original GGML library, stable-diffusion.cpp and llama.cpp by ggerganov are licensed under the MIT License
- However, KoboldAI Lite is licensed under the AGPL v3.0 License
- KoboldCpp code and other files are also under the AGPL v3.0 License unless otherwise stated
- Llama.cpp source repo is at https://github.com/ggml-org/llama.cpp (MIT)
- Stable-diffusion.cpp source repo is at https://github.com/leejet/stable-diffusion.cpp (MIT)
- KoboldCpp source repo is at https://github.com/LostRuins/koboldcpp (AGPL)
- KoboldAI Lite source repo is at https://github.com/LostRuins/lite.koboldai.net (AGPL)
- For any further enquiries, contact @concedo on discord, or LostRuins on github.

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

# Where can I download AI model files?
- The best place to get GGUF text models is huggingface. For image models, CivitAI has a good selection. Here are some to get started.
  - Text Generation: [L3-8B-Stheno-v3.2](https://huggingface.co/bartowski/L3-8B-Stheno-v3.2-GGUF/resolve/main/L3-8B-Stheno-v3.2-Q4_K_S.gguf) (smaller and weaker) or [Tiefighter 13B](https://huggingface.co/KoboldAI/LLaMA2-13B-Tiefighter-GGUF/resolve/main/LLaMA2-13B-Tiefighter.Q4_K_S.gguf) (old but very versatile model) or [Gemma-3-27B Abliterated](https://huggingface.co/mlabonne/gemma-3-27b-it-abliterated-GGUF/resolve/main/gemma-3-27b-it-abliterated.q4_k_m.gguf) (largest and most powerful)
  - Image Generation: [Anything v3](https://huggingface.co/admruul/anything-v3.0/resolve/main/Anything-V3.0-pruned-fp16.safetensors) or [Deliberate V2](https://huggingface.co/Yntec/Deliberate2/resolve/main/Deliberate_v2.safetensors) or [Dreamshaper SDXL](https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo/resolve/main/DreamShaperXL_Turbo_v2_1.safetensors)
  - Image Recognition MMproj: [Pick the correct one for your model architecture here](https://huggingface.co/koboldcpp/mmproj/tree/main)
  - Speech Recognition: [Whisper models for Speech-To-Text](https://huggingface.co/koboldcpp/whisper/tree/main)
  - Text-To-Speech: [TTS models for Narration](https://huggingface.co/koboldcpp/tts/tree/main)


-----

Croco is also using the following fork of KoboldCPP, Esobold.

# Esobold (Esolithe's fork of KoboldCPP) (for information)

![Kobold](https://github.com/user-attachments/assets/f5cb4087-bc72-47fc-91a5-29a4d6495648)

Welcome all, this fork focuses on enhanced remote management, server saving and integration in KoboldCPP and Kobold Lite.

It offers the following functionalities:
- Config reloading (Cedo's implementation) enhanced with an option to select a text model to override the config (useful to switch between models using a generic "8B" or "12B" config without needing one for each model).  There is also helper text in Lite to show you the current config and model in use, along with a waiting mechanism to only reload when it's ready.

![411112504-e4d6936f-234f-40d8-8bfe-578c4332089d](https://github.com/user-attachments/assets/de7d4105-ffdd-40c3-bc0e-26041b6ded36)
- Support for downloading models from HF through the UI (based on Cedo's implementation for the launcher) - To use this please tick the box under the admin tab in the launcher, and also ensure you run the exe from the directory where you store your models.

![e5c0d32a-d68f-4c8a-87f7-bdfade892c4f](https://github.com/user-attachments/assets/e5c0d32a-d68f-4c8a-87f7-bdfade892c4f)
![bda67a48-bc40-4815-8628-873c3136a415](https://github.com/user-attachments/assets/bda67a48-bc40-4815-8628-873c3136a415)

- Save / load from server - with or without admin password: It is possible to store saves, scenarios, character cards, lorebooks etc.  The difference with the main KCPP is this option integrates with the scenerio search, offers the ability to upload multiple types of content and does not limit the amount of save uploaded.  Both options can be used together if desired.

![413011334-de6d3101-5621-4c47-a45b-d71f1b436cce](https://github.com/user-attachments/assets/098f64c6-9f6e-4951-b4a9-b4b458ddb7d8)
- Data on the server can be assigned a thumbnail

![418191433-72b95f51-7868-4a4b-bb50-e92605e1a13c](https://github.com/user-attachments/assets/8618d216-42d8-4c34-a859-5c4a1e809136)
![413011254-30449652-2292-406c-8d27-bc04d2389a38](https://github.com/user-attachments/assets/63c3f6af-8695-4d4f-92b1-654ccf006fb4)
- Server save data can be accessed in the scenarios tab (allowing searching by types and names)

![418191036-d773e536-6c09-4b76-828e-6e542e075f0e](https://github.com/user-attachments/assets/81dfea9f-9862-4b52-ba3d-92472a1b758c)
![418190669-f839ed18-0fb9-4319-82d8-03952c5d3f20](https://github.com/user-attachments/assets/2ee0c202-8f68-44d6-aa6f-18ae35f33857)

- Agent thinking (based on prompts from this cool project [here](https://github.com/Wladastic/mini_autogpt))
- Improvements to TextDB, such as VectorDB (embedding) support and document support (including upload of text documents, PDFs (SevenOf9 wrote the parser), OCR using the vision model loaded, and transcription from audio)
- Export / Import of WI groups from files

![8d3e3a9d-08e0-4caf-9ad8-c2d806fec7e0](https://github.com/user-attachments/assets/8d3e3a9d-08e0-4caf-9ad8-c2d806fec7e0)

## Agent thinking mode (experimental)

An attempt to replicate tool usage / agent logic in Lite.  Essentially, the AI is provided the user input and a list of tools that it can use.  Should work on all UI modes for instruct, along with supporting chat names.

The currently supported options include:
- Sending messages / Asking for additional user input (including AI suggested options like a text adventure)

![2025-04-01 19_57_17-KoboldAI Lite — LibreWolf](https://github.com/user-attachments/assets/c46ce852-8d04-4451-a27e-7eb2cd1ab462)

![c46ce852-8d04-4451-a27e-7eb2cd1ab462](https://github.com/user-attachments/assets/c46ce852-8d04-4451-a27e-7eb2cd1ab462)

- Searching the web

![image](https://github.com/user-attachments/assets/1b5d066c-f71a-4b07-8e92-8a88870ed892)

- Evaluating mathematical formulas
- Rolling dice (can be used for random generation)
![image](https://github.com/user-attachments/assets/91529033-6a21-4cc3-a0c2-730fb68a99af)

- Generating images at different aspect ratios (in KCPP) - Both from text and another image
![image](https://github.com/user-attachments/assets/39bc5433-54f2-4c9e-8422-9e20d4383b71)
![image](https://github.com/user-attachments/assets/97bef072-06d9-47a1-8066-2bcb5c20bc6f)
- Analysing images (in KCPP)
- "Speaking" through TTS (in KCPP)
- Adding data to the TextDB and searching for information in it, along with the chat log
- Knowing the current date
- Enabling a word count on the AI responses
- Supports system prompts, both using and setting it automatically
- Supports setting a "state" parameter which is always inserted at the end of the text.  It is also possible to define the format that the response must use (i.e. {health: 10, mana: 20...})
- Support enforcing a specific action order (i.e. the agent can be set to always roll a dice, then send a response)
- Support for manually preventing the agent from taking specific actions:

```
[DOCUMENT BREAK][Forbidden agent commands]ask_user|roll_dice[DOCUMENT BREAK]
```

- Support for randomly selecting elements from a list of items.  The lists can be defined in the TextDB with:

```
[DOCUMENT BREAK][Table:Genres]Action Fantasy
Horror[DOCUMENT BREAK]
```

Using this function requires the following conditions to be met:
- Use an instruct model
- Use separate start and end tags for all roles (tick the option under the instruct settings and ensure they are all filled out, like ChatML)
- Ensure that if you wish the AI to use web searching, TTS or image gen that the respective options are configured and enabled in the UI

## Improvements to TextDB (probably will be upstreamed in the future)
- UI improvements

![image](https://github.com/user-attachments/assets/94b0f24e-360e-4c66-8d40-8b9185117902)
- Importing of lorebooks from the load button as Text DB entries
- Support for ```[DOCUMENT BREAK][Name of document]This is the content of the document``` which allows for user defined groupings of sections
- Support for embedding models running in KCPP - Embeddings are generated on the server based on the text DB content, and then stored in the browser (notification indicates progress - can take some time).

![image](https://github.com/user-attachments/assets/41ec4f1c-5698-4ef3-ba7c-6998cbc1d8f3)

- Upload document support (including upload of text documents, lorebooks, PDFs (SevenOf9 wrote the parser), OCR using the vision model loaded, and transcription from audio)
- Export / Import of WI groups from files

## Running the fork

Most of the settings are identical to KoboldCPP, but there's a couple of additional options to note in the launcher (admin tab) - admin must be enabled for these to work:
- Model directory: A folder with text model GGUFs you wish to allow switching between - this is an override on top of using a config, so please ensure a config directory is set as well.
- Data directory: A folder server side data is stored (for example saves, character cards etc).

![411112726-0ae7fed6-3baf-43fd-a1d6-bab52319a6ee](https://github.com/user-attachments/assets/8e80b8a3-11fc-4aa0-b4c8-5a9feacf02cb)

If you prefer the arguments:
- To turn on remote management: --admin
- To set a management password: --adminpassword "..."
- To set the reloadable configs directory --admindir "..."
- To set the reloadable models directory (overriding the config model): --admintextmodelsdir "..."
- To sets the data storage directory (where the database storing server side saves are stored): --admindatadir "..."

---