#pragma once
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include "MyOpenVINO.h"

class InferInfo
{
public:
    InferenceEngine::InferRequest inferRequest;
    std::wstring inputImage;

    InferInfo()
    {

    }

    InferInfo(InferenceEngine::InferRequest& iRequest, const std::wstring& iImange)
    {
        inferRequest = iRequest;
        inputImage = iImange;
    }
};

class MyOpenVINOImpl : public IMyOpenVINO
{
public:
	MyOpenVINOImpl();
	~MyOpenVINOImpl();
	bool Initialize(const NetworkInfo &networkInfo);
	std::vector<Device> GetAvailableDevices();
	
    std::vector<float> InferSync(const std::wstring& imageName);
    int InferASync(const std::wstring &imageName);
    void SetInferCallBack(CallbackHandlerBase& callbackHandler);
    void WaitForEndOfInfer();

private:
    int inferCounter;
    CallbackHandlerBase* pCallbackHandler;
	InferenceEngine::Core core;
	InferenceEngine::CNNNetwork network;
	InferenceEngine::ExecutableNetwork executableNetwork;
	NetworkInfo networkInfo;
	std::string inputLayerName;
	std::string outputLayerName;
    Layout inputLayout = Layout::NCHW;
    Precision inputPresicion = Precision::FP32;
    std::map<int, InferInfo> inferMap;
    std::vector<std::thread *> threadVector;

    // 推論同時実行数
    const int INFER_NUM = 2;

    // Windows依存コード
    HANDLE semhd;

	bool ReadNetwork(const std::wstring &modelName);
	bool SetNetworkConfiguration(const Layout& iLayout, const Precision& iPrecision, const Layout& oLayout, const Precision& oPrecision);
	bool SetDeviceSetting(const std::vector<Device>& devices, const unsigned long& threadNum, const bool& isMulti);
	bool LoadNetwork(const std::vector<Device>& devices, const bool& isMulti);
	bool SetOptimalNumberOfInferRequests(unsigned long &inferRequestNum);
    InferenceEngine::InferRequest CreateInferRequest(const unsigned long inferRequestNum);
	bool SetInputData(InferenceEngine::InferRequest& inferRequest, const std::wstring& imageName);
    bool GetOutput(InferenceEngine::InferRequest& iRequest);
    void  InferASyncLocal(int inferID);
    static InferenceEngine::Layout ConvertInferenceEngineLayout(Layout layout);
	static InferenceEngine::Precision ConvertInferenceEnginePrecision(Precision precision);
	static Device ConvertString2Device(std::string device);
	static std::string ConvertDevice2String(Device device);
	static std::string ConvertDevices2String(std::vector<Device> devices, bool isMulti);
    static std::string ConvertWideChar2MultiChar(const std::wstring& wideCharString, const unsigned long code);
};

/**
* @brief Sets image data stored in cv::Mat object to a given Blob object.
* @param orig_image - given cv::Mat object with an image data.
* @param blob - Blob object which to be filled by an image data.
* @param batchIndex - batch index of an image inside of the blob.
*/
template <typename T>
void matU8ToBlob(const cv::Mat& input_image, InferenceEngine::Blob::Ptr& blob,  int batchIndex = 0) {
    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();

    const size_t height = blobSize[2];
    const size_t width = blobSize[3];
    const size_t channels = blobSize[1];

    cv::Mat resized_image(input_image);

    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->wmap();
    T* blob_data = blobMapped.as<T*>();

    if (static_cast<int>(width) != input_image.size().width ||
        static_cast<int>(height) != input_image.size().height) {
        cv::resize(input_image, resized_image, cv::Size(width, height));
    }

    int batchOffset = batchIndex * width * height * channels;


    if (static_cast<size_t>(input_image.channels()) == channels)
    {
        // 入力画像のチャネルとモデルが求めるチャネルの数が同じ場合
        if (channels == 1)
        {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    blob_data[batchOffset + h * width + w] = resized_image.at<uchar>(h, w);
                }
            }
        }
        else if (channels == 3)
        {
            for (size_t c = 0; c < channels; c++) {
                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        blob_data[batchOffset + c * width * height + h * width + w] = resized_image.at<cv::Vec3b>(h, w)[c];
                    }
                }
            }
        }
        else
        {
            THROW_IE_EXCEPTION << "Unsupported number of channels";
        }
    }
    else if (channels == 3 && static_cast<size_t>(input_image.channels()) == 1)
    {
        // 入力画像はグレースケールで、モデルが求めるものはカラーの場合、
        // とりあえず、グレースケールをRGBにコピーする。
        for (size_t c = 0; c < channels; c++)
        {
            for (size_t h = 0; h < height; h++)
            {
                for (size_t w = 0; w < width; w++)
                {
                    blob_data[batchOffset + c * width * height + h * width + w] = resized_image.at<uchar>(h, w);
                }
            }
        }
    }
    else
    {
        THROW_IE_EXCEPTION << "Unsupported  input layout";
    }
}