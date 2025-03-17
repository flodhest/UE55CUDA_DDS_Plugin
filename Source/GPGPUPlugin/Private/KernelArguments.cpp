// Plugins/CUDAPlugin/Source/GPGPUPlugin/Private/KernelArguments.cpp
#include "KernelArguments.h"

UKernelArguments* UKernelArguments::ParseKernelArgumentsFromStructure(UObject* KernelArgumentStructureIn)
{
    UKernelArguments* KernelArguments = NewObject<UKernelArguments>();
    if (!KernelArgumentStructureIn)
    {
        return KernelArguments;
    }

    if (FStructProperty* StructProp = FindFProperty<FStructProperty>(KernelArgumentStructureIn->GetClass(), TEXT("Structure")))
    {
        void* StructPtr = StructProp->ContainerPtrToValuePtr<void>(KernelArgumentStructureIn);
        IterateThroughStructProperty(StructProp, StructPtr, KernelArguments);
    }

    return KernelArguments;
}

FKernelArgumentValue UKernelArguments::SubParseProperty(FProperty* Property, void* ValuePtr)
{
    FKernelArgumentValue Result;

    if (FNumericProperty* NumericProperty = CastField<FNumericProperty>(Property))
    {
        if (NumericProperty->IsFloatingPoint())
        {
            if (NumericProperty->GetSize() == sizeof(double))
            {
                Result.SetDouble(NumericProperty->GetFloatingPointPropertyValue(ValuePtr));
            }
            else
            {
                Result.SetFloat(static_cast<float>(NumericProperty->GetFloatingPointPropertyValue(ValuePtr)));
            }
        }
        else if (NumericProperty->IsInteger())
        {
            if (NumericProperty->GetSize() == sizeof(int64))
            {
                Result.SetInt64(NumericProperty->GetSignedIntPropertyValue(ValuePtr));
            }
            else
            {
                Result.SetInt32(static_cast<int32>(NumericProperty->GetSignedIntPropertyValue(ValuePtr)));
            }
        }
    }

    return Result;
}

void UKernelArguments::IterateThroughStructProperty(FStructProperty* StructProperty, void* StructPtr, UKernelArguments* KernelArgs)
{
    if (!StructProperty || !StructPtr || !KernelArgs) return;

    UScriptStruct* Struct = StructProperty->Struct;
    if (!Struct) return;

    for (TFieldIterator<FProperty> It(Struct); It; ++It)
    {
        FProperty* Property = *It;
        if (!Property) continue;

        FString VariableName = Property->GetName();
        for (int32 ArrayIndex = 0; ArrayIndex < Property->ArrayDim; ArrayIndex++)
        {
            void* ValuePtr = Property->ContainerPtrToValuePtr<void>(StructPtr, ArrayIndex);
            ParseProperty(VariableName, Property, ValuePtr, KernelArgs);
        }
    }
    KernelArgs->PrepareForLaunch();
}

void UKernelArguments::ParseProperty(FString VariableName, FProperty* Property, void* ValuePtr, UKernelArguments* KernelArgs)
{
    if (!KernelArgs || !Property || !ValuePtr) return;

    FKernelArgument Argument;
    Argument.bIsArray = false;

    if (FNumericProperty* NumericProperty = CastField<FNumericProperty>(Property))
    {
        if (NumericProperty->IsFloatingPoint())
        {
            if (NumericProperty->GetSize() == sizeof(double))
            {
                Argument.SingleValue.SetDouble(NumericProperty->GetFloatingPointPropertyValue(ValuePtr));
                Argument.Alignment = 8;
            }
            else
            {
                Argument.SingleValue.SetFloat(static_cast<float>(NumericProperty->GetFloatingPointPropertyValue(ValuePtr)));
                Argument.Alignment = 4;
            }
        }
        else if (NumericProperty->IsInteger())
        {
            if (NumericProperty->GetSize() == sizeof(int64))
            {
                Argument.SingleValue.SetInt64(NumericProperty->GetSignedIntPropertyValue(ValuePtr));
                Argument.Alignment = 8;
            }
            else
            {
                Argument.SingleValue.SetInt32(static_cast<int32>(NumericProperty->GetSignedIntPropertyValue(ValuePtr)));
                Argument.Alignment = 4;
            }
        }

        Argument.Size = NumericProperty->GetSize();
        KernelArgs->Arguments.Add(VariableName, Argument);
    }
}

void UKernelArguments::AddPointerArgument(const FString& Name, void* Pointer)
{
    FKernelArgument Argument;
    Argument.bIsArray = false;
    Argument.SingleValue.SetPointer(Pointer);
    Argument.Alignment = sizeof(void*);
    Argument.Size = sizeof(void*);
    Arguments.Add(Name, Argument);
}

void UKernelArguments::AddIntArgument(const FString& Name, int32 Value)
{
    FKernelArgument Argument;
    Argument.bIsArray = false;
    Argument.SingleValue.SetInt32(Value);
    Argument.Alignment = 4;
    Argument.Size = sizeof(int32);
    Arguments.Add(Name, Argument);
}

uint32 UKernelArguments::GetTotalArgumentSize() const
{
    uint32 totalSize = 0;
    for (const auto& Pair : Arguments)
    {
        const FKernelArgument& Arg = Pair.Value;
        uint32 alignedOffset = AlignOffset(totalSize, Arg.Alignment);
        totalSize = alignedOffset + Arg.Size;
    }
    return totalSize;
}

void UKernelArguments::GetArgumentPointers(TArray<void*>& OutPointers) const
{
    OutPointers.Reset();
    for (const auto& Pair : Arguments)
    {
        const FKernelArgument& Arg = Pair.Value;
        OutPointers.Add(const_cast<void*>(static_cast<const void*>(ArgumentBuffer.GetData() + Arg.Offset)));
    }
}

void UKernelArguments::PrepareForLaunch()
{
    uint32 totalSize = GetTotalArgumentSize();
    ArgumentBuffer.SetNumZeroed(totalSize);
    uint32 currentOffset = 0;

    for (auto& Pair : Arguments)
    {
        FKernelArgument& Arg = Pair.Value;
        uint32 alignedOffset = AlignOffset(currentOffset, Arg.Alignment);
        Arg.Offset = alignedOffset;

        if (!Arg.bIsArray)
        {
            switch (Arg.SingleValue.Type)
            {
            case FKernelArgumentValue::EValueType::Double:
                *reinterpret_cast<double*>(ArgumentBuffer.GetData() + alignedOffset) = Arg.SingleValue.AsDouble;
                currentOffset = alignedOffset + sizeof(double);
                break;
            case FKernelArgumentValue::EValueType::Float:
                *reinterpret_cast<float*>(ArgumentBuffer.GetData() + alignedOffset) = Arg.SingleValue.AsFloat;
                currentOffset = alignedOffset + sizeof(float);
                break;
            case FKernelArgumentValue::EValueType::Int64:
                *reinterpret_cast<int64*>(ArgumentBuffer.GetData() + alignedOffset) = Arg.SingleValue.AsInt64;
                currentOffset = alignedOffset + sizeof(int64);
                break;
            case FKernelArgumentValue::EValueType::Int32:
                *reinterpret_cast<int32*>(ArgumentBuffer.GetData() + alignedOffset) = Arg.SingleValue.AsInt32;
                currentOffset = alignedOffset + sizeof(int32);
                break;
            case FKernelArgumentValue::EValueType::UInt32:
                *reinterpret_cast<uint32*>(ArgumentBuffer.GetData() + alignedOffset) = Arg.SingleValue.AsUInt32;
                currentOffset = alignedOffset + sizeof(uint32);
                break;
            case FKernelArgumentValue::EValueType::Pointer:
                *reinterpret_cast<void**>(ArgumentBuffer.GetData() + alignedOffset) = Arg.SingleValue.AsPointer;
                currentOffset = alignedOffset + sizeof(void*);
                break;
            }
        }
    }
}