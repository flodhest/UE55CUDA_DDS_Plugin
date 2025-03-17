using UnrealBuildTool;
using System;
using System.IO;
using System.Linq;

public class GPGPUPlugin : ModuleRules
{
    // Explicitly use 'new' to avoid shadowing warnings if these properties overlap with base class
    private new string PluginDirectory => Path.GetFullPath(Path.Combine(ModuleDirectory, "..", ".."));
    private string ThirdPartyDirectory => Path.Combine(PluginDirectory, "ThirdParty");
    private string IncludeDirectory => Path.Combine(ThirdPartyDirectory, "Include");
    private string LibraryDirectory => Path.Combine(ThirdPartyDirectory, "Lib");
    private string BinaryDirectory => Path.Combine(ThirdPartyDirectory, "bin");
    private string CUDAKernelsDirectory => Path.Combine(ModuleDirectory, "Private");

    private class CudaArchInfo
    {
        public string ComputeArch { get; }
        public string SmArch { get; }
        public string Name { get; }
        public bool IsRTX { get; }
        public bool IsQuadro { get; }
        public CudaArchInfo(string compute, string sm, string name, bool isRtx = false, bool isQuadro = false)
        {
            ComputeArch = compute;
            SmArch = sm;
            Name = name;
            IsRTX = isRtx;
            IsQuadro = isQuadro;
        }
    }

    private readonly CudaArchInfo[] SupportedCudaArchs = new[]
    {
        new CudaArchInfo("compute_100", "sm_100", "Blackwell (RTX 50 series)", true, false),
        new CudaArchInfo("compute_100a", "sm_100a", "Blackwell (GB200, B100)", false, true),
        new CudaArchInfo("compute_90", "sm_90", "Hopper (H100)", false, true),
        new CudaArchInfo("compute_90a", "sm_90a", "Hopper (H200)", false, true),
        new CudaArchInfo("compute_89", "sm_89", "Ada Lovelace (RTX 40 series)", true, false),
        new CudaArchInfo("compute_86", "sm_86", "Ampere (RTX 30 series)", true, false),
        new CudaArchInfo("compute_80", "sm_80", "Ampere (A100, A40, A10, RTX A6000)", false, true),
        new CudaArchInfo("compute_75", "sm_75", "Turing (RTX 20 series, Quadro RTX)", true, true),
        new CudaArchInfo("compute_70", "sm_70", "Volta (Tesla V100)", false, true),
        new CudaArchInfo("compute_72", "sm_72", "Volta (Xavier)", false, false),
        new CudaArchInfo("compute_61", "sm_61", "Pascal (GTX 10 series, Quadro P-series)", false, true),
        new CudaArchInfo("compute_60", "sm_60", "Pascal (GP100, Tesla P100)", false, true),
        new CudaArchInfo("compute_52", "sm_52", "Maxwell (GTX 900 series, Quadro M-series)", false, true),
        new CudaArchInfo("compute_53", "sm_53", "Maxwell (Jetson TX1, Jetson Nano)", false, false),
    };

    private void SetupCudaArchitectureSupport()
    {
        var rtxArchitectures = SupportedCudaArchs.Where(arch => arch.IsRTX).ToList();
        var quadroArchitectures = SupportedCudaArchs.Where(arch => arch.IsQuadro).ToList();

        PublicDefinitions.Add($"CUDA_SUPPORTED_ARCH_COUNT={SupportedCudaArchs.Length}");
        var latestArch = SupportedCudaArchs.FirstOrDefault();
        if (latestArch != null)
        {
            PublicDefinitions.Add($"CUDA_LATEST_COMPUTE={latestArch.ComputeArch}");
            PublicDefinitions.Add($"CUDA_LATEST_SM={latestArch.SmArch}");
        }

        PublicDefinitions.Add($"CUDA_SUPPORTED_COMPUTE_ARCHS=\"{string.Join(",", SupportedCudaArchs.Select(a => a.ComputeArch))}\"");
        PublicDefinitions.Add($"CUDA_SUPPORTED_SM_ARCHS=\"{string.Join(",", SupportedCudaArchs.Select(a => a.SmArch))}\"");
        PublicDefinitions.Add($"CUDA_SUPPORTED_RTX_ARCHS=\"{string.Join(",", rtxArchitectures.Select(a => a.SmArch))}\"");
        PublicDefinitions.Add($"CUDA_SUPPORTED_QUADRO_ARCHS=\"{string.Join(",", quadroArchitectures.Select(a => a.SmArch))}\"");

        PublicDefinitions.Add($"CUDA_NVRTC_OPTIONS=\"{string.Join(" ", SupportedCudaArchs.Select(a => $"--gpu-architecture={a.ComputeArch}"))}\"");
        foreach (var arch in SupportedCudaArchs)
        {
            PublicDefinitions.Add($"CUDA_SUPPORTS_{arch.SmArch.Replace("sm_", "SM").ToUpper()}=1");
        }

        PublicDefinitions.Add("GPGPU_DYNAMIC_SM_DETECTION=1");
        PublicDefinitions.Add("GPGPU_SUPPORTS_RTX_MODELS=1");
        PublicDefinitions.Add("GPGPU_SUPPORTS_QUADRO_MODELS=1");

        Console.WriteLine($"CUDA Plugin configured with support for {SupportedCudaArchs.Length} GPU architectures");
        Console.WriteLine($"RTX Models: {rtxArchitectures.Count} architectures supported");
        Console.WriteLine($"Quadro/Professional Models: {quadroArchitectures.Count} architectures supported");
        Console.WriteLine($"Latest architecture: {latestArch?.Name} ({latestArch?.SmArch})");
    }

    private void SetupDefinitions()
    {
        PublicDefinitions.Add("WITH_CUDA=1");
        PublicDefinitions.Add("WITH_GPGPU=1");
    }

    public GPGPUPlugin(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;
        bEnableUndefinedIdentifierWarnings = false;
        CppStandard = CppStandardVersion.Cpp17;
        bEnableExceptions = true;
        bLegacyPublicIncludePaths = false;

        // Ensure all public headers are accessible
        PublicIncludePaths.Add(Path.Combine(ModuleDirectory, "Public"));
        PrivateIncludePaths.Add(Path.Combine(ModuleDirectory, "Private"));
        PublicSystemIncludePaths.AddRange(new string[] {
            Path.Combine(EngineDirectory, "Source/Runtime/Core/Public"),
            Path.Combine(EngineDirectory, "Source/Runtime/CoreUObject/Public"),
            IncludeDirectory
        });

        // Runtime dependencies
        PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "CoreUObject",
            "CycloneDDS",
            "Engine",
            "RenderCore",


            "RHI",
            "Projects",
            "InputCore",
            "UE53Chaos_Cust_Truck" // Ensure project module is included if needed
        });

        PrivateDependencyModuleNames.AddRange(new string[] {
            "Slate",
            "SlateCore"
        });

        // Add GPGPUPlugin as a self-dependency to ensure headers are visible
        PublicDependencyModuleNames.Add("GPGPUPlugin");

        // Platform-specific setup
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            SetupCudaDependencies();
        }
        else
        {
            PublicDefinitions.Add("GPGPU_PLATFORM_UNSUPPORTED=1");
        }

        SetupDefinitions();
    }

    private void SetupCudaDependencies()
    {
        string CudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
        if (string.IsNullOrEmpty(CudaPath))
        {
            Console.WriteLine("Warning: CUDA_PATH not found. Ensure CUDA Toolkit 12.8 is installed.");
            PublicDefinitions.Add("GPGPU_CUDA_TOOLKIT_NOT_FOUND=1");
            return;
        }

        string CudaLibPath = Path.Combine(CudaPath, "lib", "x64");
        if (!Directory.Exists(CudaLibPath))
        {
            Console.WriteLine($"Warning: CUDA library path not found: {CudaLibPath}");
            PublicDefinitions.Add("GPGPU_CUDA_LIBRARIES_NOT_FOUND=1");
            return;
        }

        PublicSystemIncludePaths.Add(Path.Combine(CudaPath, "include"));
        PublicSystemLibraryPaths.Add(CudaLibPath);

        SetupCudaArchitectureSupport();

        string[] NvidiaLibs = new[] {
            "cuda.lib", "cudart.lib", "cudart_static.lib", "nvrtc.lib", "nvrtc_static.lib",
            "nvrtc-builtins_static.lib", "nvJitLink.lib", "nvJitLink_static.lib",
            "cublas.lib", "cublasLt.lib", "cudadevrt.lib", "cufft.lib", "cufftw.lib",
            "cufilt.lib", "curand.lib", "cusolver.lib", "cusolverMg.lib", "cusparse.lib",
            "nppc.lib", "nppial.lib", "nppicc.lib", "nppidei.lib", "nppif.lib", "nppig.lib",
            "nppim.lib", "nppist.lib", "nppisu.lib", "nppitc.lib", "npps.lib", "nvblas.lib",
            "nvfatbin.lib", "nvfatbin_static.lib", "nvjpeg.lib", "nvml.lib",
            "nvptxcompiler_static.lib", "OpenCL.lib"
        };

        int foundLibraries = 0, missingLibraries = 0;
        foreach (string lib in NvidiaLibs)
        {
            string libPath = Path.Combine(CudaLibPath, lib);
            if (File.Exists(libPath))
            {
                PublicAdditionalLibraries.Add(libPath);
                foundLibraries++;
            }
            else
            {
                Console.WriteLine($"Warning: CUDA library not found: {libPath}");
                missingLibraries++;
            }
        }

        Console.WriteLine($"Found {foundLibraries} libraries in CUDA installation. Missing {missingLibraries} libraries.");

        string cudaDevRtPath = Path.Combine(CudaLibPath, "cudadevrt.lib");
        if (File.Exists(cudaDevRtPath))
        {
            PublicDefinitions.Add($"CUDA_DEVRT_PATH=TEXT(\"{cudaDevRtPath.Replace("\\", "/")}\")");
        }

        string mainKernelPath = Path.Combine(CUDAKernelsDirectory, "saxpy.cu");
        if (File.Exists(mainKernelPath))
        {
            RuntimeDependencies.Add(mainKernelPath, StagedFileType.NonUFS);
        }

        if (Directory.Exists(CUDAKernelsDirectory))
        {
            foreach (string kernelFile in Directory.GetFiles(CUDAKernelsDirectory, "*.cu"))
            {
                RuntimeDependencies.Add(kernelFile, StagedFileType.NonUFS);
            }
        }




    }
}
