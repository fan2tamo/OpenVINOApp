#define DLL_EXPORT

#include <Windows.h>
#include "MyOpenVINOImpl.h"
#include "ie_common.h"
#include "ie_precision.hpp"
#include"ie_extension.h"
#include <exception>
#include <stdio.h>

#pragma region Public API

MyOpenVINOImpl::MyOpenVINOImpl()
{
	this->pCallbackHandler = NULL;
	inferCounter = 0;
	semhd = CreateSemaphore(NULL, 1, 1, L"SemName");
}

MyOpenVINOImpl::~MyOpenVINOImpl()
{	
	CloseHandle(semhd);
}

/// <summary>
/// 初期化
/// </summary>
/// <param name="networkInfo"></param>
/// <returns></returns>
bool MyOpenVINOImpl::Initialize(const NetworkInfo &networkInfo)
{
	bool ret = false;

	ret = ReadNetwork(networkInfo.modelName);
	if (ret == false)
	{
		return ret;
	}

	ret = SetNetworkConfiguration(networkInfo.inputLayout, networkInfo.inputPrecision, networkInfo.outputLayout, networkInfo.outputPrecision);
	if (ret == false)
	{
		return ret;
	}
	
	
	ret = SetDeviceSetting(networkInfo.devices, networkInfo.threadNum, networkInfo.isMultiDevices);
	if (ret == false)
	{
		return ret;
	}
	
	ret = LoadNetwork(networkInfo.devices, networkInfo.isMultiDevices);
	if (ret == false)
	{
		return ret;
	}

	ret = CreateInferRequest(networkInfo.inferRequestNum);

	return ret;
}

/// <summary>
/// 推論(同期)
/// </summary>
/// <param name="imageName"></param>
/// <returns></returns>
std::vector<float>  MyOpenVINOImpl::InferSync(const std::wstring& imageName)
{
	OutputDebugStringW(L"Test");


	SetInputData(imageName);
	inferRequest.Infer();
	std::vector<float> outputVect;
	InferenceEngine::SizeVector dims = inferRequest.GetBlob(outputLayerName)->getTensorDesc().getDims();
	const float* oneHotVector = (inferRequest.GetBlob(outputLayerName))->buffer().as<float*>();
	int dim = dims[1];
	
	for (int i = 0; i < dim; i++)
	{
		outputVect.push_back(oneHotVector[i]);
	}
	
	return outputVect;
}

/// <summary>
/// 非同期で推論を実施する際に結果を返すコールバックを登録
/// </summary>
/// <param name="callbackHandler"></param>
void MyOpenVINOImpl::SetInferCallBack(CallbackHandlerBase& callbackHandler)
{
	this->pCallbackHandler = &callbackHandler;
}

/// <summary>
/// 推論(非同期)
/// </summary>
/// <param name="imageName"></param>
/// <returns></returns>
int MyOpenVINOImpl::InferASync(const std::wstring& imageName)
{
	int inferID = inferCounter;
	inferCounter += 1;
	inferMap[inferID] = imageName;
	std::thread* th = NULL;

	// 永久に待つ
	DWORD ret = WaitForSingleObject(semhd, INFINITE);

	th =new std::thread(&MyOpenVINOImpl::InferASyncLocal, this, inferID);
	//threadVector.push_back(th);
	threadVector.push_back(std::move(th));
	return inferID;
}

/// <summary>
/// 非同期で推論を実施する際、結果を待ち合わせる
/// </summary>
void MyOpenVINOImpl::WaitForEndOfInfer()
{
	for (int i = 0; i < threadVector.size(); i++)
	{
		threadVector[i]->join();
	}
}

#pragma endregion

#pragma region Private API

std::vector<Device> MyOpenVINOImpl::GetAvailableDevices()
{
	std::vector<std::string> availableDevices = core.GetAvailableDevices();
	std::vector<Device> devices;

	for (int i = 0; i < availableDevices.size(); i++)
	{
		Device d = ConvertString2Device(availableDevices[i]);
		devices.push_back(d);
		printf("availableDevice[%d] = %s\r\n", i, availableDevices[i].c_str());
	}

	return devices;

}

bool MyOpenVINOImpl::ReadNetwork(const std::wstring &modelName)
{
	bool ret = true;

	try
	{
		if (modelName.find(L".onnx") == std::wstring::npos)
		{
			// IR
			std::wstring binName = modelName.substr(0, modelName.length() - 4) + L".bin";
			network = core.ReadNetwork(modelName, binName);
		}
		else
		{
			// onnx
			network = core.ReadNetwork(modelName, NULL);
		}
	}
	catch (InferenceEngine::details::InferenceEngineException e)
	{
		printf("Error ReadNetwork() : %s\r\n", e.what());
		ret = false;
	}

	/*
	catch (...)
	{
		printf("Error ReadNetwork() \r\n");
		ret = false;
	}
	*/
	return ret;
}

bool MyOpenVINOImpl::SetNetworkConfiguration(const Layout&  iLayout, const Precision& iPrecision, const Layout& oLayout, const Precision& oPrecision)
{
	bool ret = true;
	try
	{
		auto inputLayerData = network.getInputsInfo().begin()->second;
		InferenceEngine::Layout il = ConvertInferenceEngineLayout(iLayout);
		InferenceEngine::Precision ip = ConvertInferenceEnginePrecision(iPrecision);
		inputLayerData->setLayout(il);
		inputLayerData->setPrecision(ip);
		inputLayerName = std::string( network.getInputsInfo().begin()->first.c_str() );
		inputLayout = iLayout;
		inputPresicion = iPrecision;

		auto outputLayerData = network.getOutputsInfo().begin()->second;
		InferenceEngine::Layout ol = ConvertInferenceEngineLayout(oLayout);
		InferenceEngine::Precision op = ConvertInferenceEnginePrecision(oPrecision);
		outputLayerData->setLayout(ol);
		outputLayerData->setPrecision(op);
		outputLayerName = std::string ( network.getOutputsInfo().begin()->first.c_str() );
	}
	catch (...)
	{
		printf("Error SetNetworkConfiguration()\r\n");
		ret = false;
	}

	return ret;
}

bool MyOpenVINOImpl::SetDeviceSetting(const std::vector<Device>& devices,  const unsigned long& threadNum , const bool &isMulti )
{
	for (auto& device : devices)
	{
		std::string deviceStr = ConvertDevice2String(device);
		switch (device)
		{
		case Device::CPU:
			// CPU supports a few special performance-oriented keys
			// limit threading for CPU portion of inference
			if (threadNum != 0)
			{
				core.SetConfig({ { CONFIG_KEY(CPU_THREADS_NUM), std::to_string(threadNum) } }, deviceStr);
			}

			if (isMulti == true && std::find(devices.begin(), devices.end(), Device::GPU) != devices.end())
			{
				core.SetConfig({ { CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(NO) } }, deviceStr);
			}
			else
			{
				// pin threads for CPU portion of inference
				core.SetConfig({ { CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(YES) } }, deviceStr);
			}

			// TODO : 解読しないといけないけど、よくわかんない・・・
			/*
				// for CPU execution, more throughput-oriented execution via streams
				ie.SetConfig({ { CONFIG_KEY(CPU_THROUGHPUT_STREAMS),
								(deviceNstreams.count(device) > 0 ? std::to_string(deviceNstreams.at(device))
																  : CONFIG_VALUE(CPU_THROUGHPUT_AUTO)) } }, device);
				deviceNstreams[device] = std::stoi(
					ie.GetConfig(device, CONFIG_KEY(CPU_THROUGHPUT_STREAMS)).as<std::string>());
			*/

			break;

		case Device::GPU:
			// TODO : 解読しないといけないけど、よくわかんない・・・
			core.SetConfig({{ CONFIG_KEY(GPU_THROUGHPUT_STREAMS),CONFIG_VALUE(GPU_THROUGHPUT_AUTO) } }, "GPU");

			/*
			ie.SetConfig({ { CONFIG_KEY(GPU_THROUGHPUT_STREAMS),
							(deviceNstreams.count(device) > 0 ? std::to_string(deviceNstreams.at(device))
															  : CONFIG_VALUE(GPU_THROUGHPUT_AUTO)) } }, device);
			deviceNstreams[device] = std::stoi(
				ie.GetConfig(device, CONFIG_KEY(GPU_THROUGHPUT_STREAMS)).as<std::string>());

			if (FLAGS_d.find("MULTI") != std::string::npos && devices.find("CPU") != devices.end())
			{
				// multi-device execution with the CPU + GPU performs best with GPU throttling hint,
				// which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
				ie.SetConfig({ { CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), "1" } }, "GPU");
			}
			*/
			break;
		}
	}

	return true;
}

bool MyOpenVINOImpl::LoadNetwork(const std::vector<Device>& devices, const bool &isMulti )
{
	bool ret = true;

	try
	{
		std::string deviceStr = ConvertDevices2String(devices, isMulti);
		executableNetwork = core.LoadNetwork(network, deviceStr);
	}
	catch (...)
	{
		ret = false;
		printf("Error : LoadNetwork()\r\n");
	}

	return ret;
}

bool MyOpenVINOImpl::SetOptimalNumberOfInferRequests(unsigned long &inferRequestNum)
{
	bool ret = true;

	if (inferRequestNum == 0) 
	{
		std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
		try {
			inferRequestNum = executableNetwork.GetMetric(key).as<unsigned int>();
		}
		catch (...) 
		{
			ret = false;
		}
	}

	return ret;
}

bool MyOpenVINOImpl::CreateInferRequest(const unsigned long inferRequestsNum)
{
	bool ret = true;

	try
	{
		inferRequest = executableNetwork.CreateInferRequest();
		/*
		if (inputShape.empty())
		{
			inputShape = inferRequest.GetBlob(inputLayerName)->getTensorDesc().getDims();
		}
		*/
	}
	catch (...)
	{
		ret = false;
	}

	return ret;
}

bool MyOpenVINOImpl::SetInputData(const std::wstring &imageName)
{
	bool ret = true;

	std::string multiCharStr = std::string(ConvertWideChar2MultiChar(imageName, CP_UTF8));
	cv::Mat imageData = cv::imread(multiCharStr);

	InferenceEngine::Blob::Ptr inputBlob = inferRequest.GetBlob(inputLayerName);
	matU8ToBlob<uint8_t>(imageData, inputBlob);
	
	return ret;
}

bool  MyOpenVINOImpl::GetOutput(InferenceEngine::InferRequest &ir)
{
	bool ret = true;
	const float* oneHotVector = (ir.GetBlob(outputLayerName))->buffer().as<float*>();

	return ret;

}

void  MyOpenVINOImpl::InferASyncLocal(int inferID)
{
	std::wstring imageName = inferMap[inferID];
	SetInputData(imageName);
	inferRequest.StartAsync();
	std::vector<float> outputVect;
	inferRequest.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
	InferenceEngine::SizeVector dims = inferRequest.GetBlob(outputLayerName)->getTensorDesc().getDims();
	const float* oneHotVector = (inferRequest.GetBlob(outputLayerName))->buffer().as<float*>();
	int dim = dims[1];

	for (int i = 0; i < dim; i++)
	{
		outputVect.push_back(oneHotVector[i]);
	}

	ReleaseSemaphore(semhd, 1, NULL);

	if (pCallbackHandler != NULL)
	{
		pCallbackHandler->InferCallBack(inferID, true,  outputVect);
	}

	return ;
}

#pragma endregion

#pragma region static API

InferenceEngine::Layout MyOpenVINOImpl::ConvertInferenceEngineLayout(Layout layout)
{
	InferenceEngine::Layout ret = InferenceEngine::Layout::ANY;

	switch (layout)
	{
	case Layout::NCHW:
		ret = InferenceEngine::Layout::NCHW;
		break;

		/*
		* not support layout
	case Layout::NHWC:
		ret = InferenceEngine::Layout::NHWC;
		break;
		*/
	case Layout::NC:
		ret = InferenceEngine::Layout::NC;
		break;

	default:
		throw std::exception("Unknown Layout\n");
		break;
	}

	return ret;
}

InferenceEngine::Precision MyOpenVINOImpl::ConvertInferenceEnginePrecision(Precision precision)
{
	InferenceEngine::Precision ret = InferenceEngine::Precision::UNSPECIFIED;

	switch (precision)
	{
	case Precision::FP16:
		ret = InferenceEngine::Precision::FP16;
		break;

	case Precision::FP32:
		ret = InferenceEngine::Precision::FP32;
		break;

	case Precision::U8:
		ret = InferenceEngine::Precision::U8;
		break;

	default:
		throw std::exception("Unknown Precision\n");
		break;
	}

	return ret;
}

Device MyOpenVINOImpl::ConvertString2Device(std::string device)
{
	Device ret = Device::CPU;

	if (device == "CPU")
	{
		ret = Device::CPU;
	}
	else if (device == "VPU" || device == "MYRIAD")
	{
		ret = Device::MYRIAD;
	}
	else if (device == "GPU")
	{
		ret =Device::GPU;
	}
	else if (device == "FPGA")
	{
		ret = Device::FPGA;
	}
	else if (device == "GNA")
	{
		ret = Device::GNA;
	}
	else
	{
		throw std::exception("Unknown Device Name\n");
	}

	return ret;
}

std::string MyOpenVINOImpl::ConvertDevice2String(Device device)
{
	std::string deviceStr = "CPU";

	switch (device)
	{
	case CPU:
		deviceStr = "CPU";
		break;
	case GPU:
		deviceStr = "GPU";
		break;
	case MYRIAD:
		deviceStr = "MYRIAD";
		break;
	case FPGA:
		deviceStr = "FPGA";
		break;
	default:
		throw std::exception("Unknown Device Name\n");
		break;
	}

	return deviceStr;
}

std::string MyOpenVINOImpl::ConvertDevices2String(std::vector<Device> devices, bool isMulti)
{
	std::string deviceStr;

	if (isMulti == true)
	{
		deviceStr = "Multi:";
	}

	for (auto& device : devices)
	{
		switch (device)
		{
		case CPU:
			deviceStr += "CPU,";
			break;
		case GPU:
			deviceStr += "GPU,";
			break;
		case MYRIAD:
			deviceStr += "MYRIAD,";
			break;
		case FPGA:
			deviceStr += "FPGA,";
			break;
		default:
			deviceStr += "CPU,";
			break;
		}
	}
	
	deviceStr.pop_back();

	return deviceStr;
}

std::string MyOpenVINOImpl::ConvertWideChar2MultiChar(const std::wstring& wideCharString, const unsigned long code)
{
	char buf[1024];
	std::string multiCharStr;
	WideCharToMultiByte(code, 0, wideCharString.c_str(), -1, buf, 1024, NULL, NULL);
	multiCharStr = buf;
	return multiCharStr;

}

#pragma endregion


// クラスの実態を取得するAPI
DLL IMyOpenVINO* GetInstance(void)
{
	return new MyOpenVINOImpl;
}