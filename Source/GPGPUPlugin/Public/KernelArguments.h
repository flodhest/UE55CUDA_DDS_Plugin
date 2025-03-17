// Plugins/CUDAPlugin/Source/GPGPUPlugin/Public/KernelArguments.h
#pragma once

#include "CoreMinimal.h"
#include "Containers/Array.h"
#include "Containers/Map.h"
#include "UObject/UnrealType.h"
#include "KernelArguments.generated.h"

class FProperty;
class FNumericProperty;

struct FKernelArgumentValue
{
    union
    {
        double AsDouble;
        float AsFloat;
        int64 AsInt64;
        int32 AsInt32;
        uint32 AsUInt32;
        void* AsPointer; // Added for device pointers
    };

    enum class EValueType : uint8
    {
        Double,
        Float,
        Int64,
        Int32,
        UInt32,
        Pointer // Added for CUDA device pointers
    };

    EValueType Type;

    FKernelArgumentValue() : AsInt64(0), Type(EValueType::Int64) {}

    void SetDouble(double Value) { AsDouble = Value; Type = EValueType::Double; }
    void SetFloat(float Value) { AsFloat = Value; Type = EValueType::Float; }
    void SetInt64(int64 Value) { AsInt64 = Value; Type = EValueType::Int64; }
    void SetInt32(int32 Value) { AsInt32 = Value; Type = EValueType::Int32; }
    void SetUInt32(uint32 Value) { AsUInt32 = Value; Type = EValueType::UInt32; }
    void SetPointer(void* Value) { AsPointer = Value; Type = EValueType::Pointer; } // Added for pointers

    uint32 GetSize() const
    {
        switch (Type)
        {
        case EValueType::Double: return sizeof(double);
        case EValueType::Float: return sizeof(float);
        case EValueType::Int64: return sizeof(int64);
        case EValueType::Int32: return sizeof(int32);
        case EValueType::UInt32: return sizeof(uint32);
        case EValueType::Pointer: return sizeof(void*); // Size of a pointer
        default: return 0;
        }
    }
};

struct FKernelArgument
{
    bool bIsArray;
    FKernelArgumentValue SingleValue;
    TArray<FKernelArgumentValue> ArrayValues;
    uint32 Size;
    uint32 Alignment;
    uint32 Offset;

    FKernelArgument()
        : bIsArray(false)
        , Size(0)
        , Alignment(8)
        , Offset(0)
    {
    }
};

UCLASS(Blueprintable)
class UKernelArguments : public UObject
{
    GENERATED_BODY()

public:
    static FKernelArgumentValue SubParseProperty(FProperty* Property, void* ValuePtr);
    static void ParseProperty(FString VariableName, FProperty* Property, void* ValuePtr, UKernelArguments* KernelArgs);
    static void IterateThroughStructProperty(FStructProperty* StructProperty, void* StructPtr, UKernelArguments* KernelArgs);

    UFUNCTION(BlueprintCallable, Category = "Kernel Argument Functions", CustomThunk, meta = (CustomStructureParam = "KernelArgumentStructure"))
    static UKernelArguments* ParseKernelArgumentsFromStructure(UObject* KernelArgumentStructureIn);

    DECLARE_FUNCTION(execParseKernelArgumentsFromStructure)
    {
        UKernelArguments* KernelArguments = NewObject<UKernelArguments>();
        Stack.Step(Stack.Object, NULL);
        FStructProperty* StructProperty = CastField<FStructProperty>(Stack.MostRecentProperty);
        void* StructPtr = Stack.MostRecentPropertyAddress;
        P_FINISH;
        P_NATIVE_BEGIN;
        KernelArguments->InternalStructureProperty = StructProperty;
        KernelArguments->InternalStructPtr = StructPtr;
        IterateThroughStructProperty(StructProperty, StructPtr, KernelArguments);
        *(UKernelArguments**)Z_Param__Result = KernelArguments;
        P_NATIVE_END;
    }

    void AddPointerArgument(const FString& Name, void* Pointer); // Added for device pointers
    void AddIntArgument(const FString& Name, int32 Value); // Added for width/height
    void PrepareForLaunch();
    uint32 GetTotalArgumentSize() const;
    void GetArgumentPointers(TArray<void*>& OutPointers) const;

    FStructProperty* InternalStructureProperty;
    void* InternalStructPtr;
    TMap<FString, FKernelArgument> Arguments;

protected:
    static uint32 AlignOffset(uint32 Offset, uint32 Alignment)
    {
        return (Offset + Alignment - 1) & ~(Alignment - 1);
    }

private:
    TArray<uint8> ArgumentBuffer;
};