#include "KernelContainer.h"

UKernelContainer::UKernelContainer()
    : KernelType(EKernelContainer::KernelType::kCUDA)
{
    KernelCode = FText::FromString(
        TEXT("extern \"C\" __global__ void rgbaToRgbKernel(const unsigned char* input, unsigned char* output, int width, int height)\n")
        TEXT("{\n")
        TEXT("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n")
        TEXT("    int totalPixels = width * height;\n")
        TEXT("    if (idx < totalPixels) {\n")
        TEXT("        int inIdx = idx * 4;\n")
        TEXT("        int outIdx = idx * 3;\n")
        TEXT("        output[outIdx] = input[inIdx + 2]; // B\n")
        TEXT("        output[outIdx + 1] = input[inIdx + 1]; // G\n")
        TEXT("        output[outIdx + 2] = input[inIdx]; // R\n")
        TEXT("    }\n")
        TEXT("}\n")
    );
    KernelEntryPoint = "rgbaToRgbKernel";
}