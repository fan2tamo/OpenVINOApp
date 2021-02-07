#pragma once

#include <vector>

#ifdef DLL_EXPORT
#define DLL __declspec(dllexport)
#else
#define DLL __declspec(dllimport)
#endif

enum Layout
{
    ANY, // Meaning of uninitialized.
    NCHW,
    // NHWC, Not support !
    NC
};

enum Precision
{
    UNSPECIFIED = 0, // Meaning of uninitialized.
    FP16,
    FP32,
    U8,
};

enum Device
{
    CPU,
    GPU,
    MYRIAD,
    FPGA,
    GNA,
};

class CallbackHandlerBase
{
public:

    virtual void  InferCallBack(int inferID, bool isSuccessed, std::vector<float> results) = 0;
   // virtual  ~CallbackHandlerBase() = 0;
};

class NetworkInfo
{
public:
    std::wstring modelName;
    Layout inputLayout = Layout::ANY;
    Precision inputPrecision = Precision::UNSPECIFIED;
    Layout outputLayout = Layout::ANY;
    Precision outputPrecision = Precision::UNSPECIFIED;
    std::vector<Device> devices;
    bool isMultiDevices = false;
    unsigned long threadNum = 1;
    unsigned long inferRequestNum = 0;
};

class IMyOpenVINO
{
public:
    virtual bool Initialize(const NetworkInfo &networkInfo) = 0;
    virtual std::vector<Device> GetAvailableDevices() = 0;
 
    virtual int InferASync(const std::wstring &imageName)  = 0;
    virtual std::vector<float>InferSync(const std::wstring& imageName) = 0;
    virtual void SetInferCallBack(CallbackHandlerBase& callbackHandler) = 0;
    virtual void WaitForEndOfInfer() = 0;
};

typedef IMyOpenVINO* (*GetInstanceFuncPointer)(void);
extern "C" DLL IMyOpenVINO * GetInstance(void);
