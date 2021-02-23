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
}

MyOpenVINOImpl::~MyOpenVINOImpl()
{	
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
		
	ret = SetDeviceConfig(networkInfo.devices, networkInfo.threadNum, networkInfo.isMultiDevices);
	if (ret == false)
	{
		return ret;
	}
	
	ret = LoadNetwork(networkInfo.devices, networkInfo.isMultiDevices);

	return ret;
}

/// <summary>
/// 推論(同期)
/// </summary>
/// <param name="imageName"></param>
/// <returns></returns>
std::vector<float>  MyOpenVINOImpl::InferSync(const std::string& imageName)
{
	InferenceEngine::InferRequest inferRequest = CreateInferRequest();

	SetInputData(inferRequest, imageName);
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
int MyOpenVINOImpl::InferASync(const std::string& imageName)
{
	int inferID = inferCounter;
	inferCounter += 1;

	InferenceEngine::InferRequest inferRequest = CreateInferRequest();

	InferInfo inferInfo(inferRequest, imageName);
	inferMap[inferID] = inferInfo;
	std::thread* th = NULL;

	// 現状はInferASync()が実行されるたびにスレッドを作ってその中でStartAsync()をしている。
	// スレッドを作るコストは高いので、本当はthreadNumだけWorkerスレッド作って、ということをするべき。
	th =new std::thread(&MyOpenVINOImpl::InferASyncLocal, this, inferID);
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
		delete threadVector[i];
	}
	threadVector.clear();
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

bool MyOpenVINOImpl::ReadNetwork(const std::string &modelName)
{
	bool ret = true;

	try
	{
		if (modelName.find(".onnx") == std::string::npos)
		{
			// IR
			std::string binName = modelName.substr(0, modelName.length() - 4) + ".bin";
			network = core.ReadNetwork(modelName, binName);
		}
		else
		{
			// onnx
			network = core.ReadNetwork(modelName);
		}
	}
	catch (InferenceEngine::details::InferenceEngineException e)
	{
		printf("Error ReadNetwork() : %s\r\n", e.what());
		ret = false;
	}
	catch (...)
	{
		printf("Error ReadNetwork() \r\n");
		ret = false;
	}

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

/// <summary>
/// https://docs.openvinotoolkit.org/latest/openvino_docs_optimization_guide_dldt_optimization_guide.html
/// </summary>
/// <param name="devices"></param>
/// <param name="threadNum"></param>
/// <param name="isMulti"></param>
/// <returns></returns>
bool MyOpenVINOImpl::SetDeviceConfig(const std::vector<Device>& devices,  const unsigned long& threadNum , const bool &isMulti )
{
	for (auto& device : devices)
	{
		std::string deviceStr = ConvertDevice2String(device);
		switch (device)
		{
		case Device::CPU:
			// Check following link.
			// https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_CPU.html


			// CPU supports a few special performance-oriented keys
			// limit threading for CPU portion of inference
			if (threadNum != 0)
			{
				core.SetConfig({ { CONFIG_KEY(CPU_THREADS_NUM), std::to_string(threadNum) } }, "CPU");
			}

			core.SetConfig({ { CONFIG_KEY(CPU_THROUGHPUT_STREAMS),CONFIG_VALUE(CPU_THROUGHPUT_AUTO) } }, "CPU");


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
			// Check following link.
			// https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_CL_DNN.html


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

unsigned long MyOpenVINOImpl::GetOptimalNumberOfInferRequests()
{
	unsigned long  ret = 1;

	std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
	try
	{
		ret = executableNetwork.GetMetric(key).as<unsigned int>();
	}
	catch (...)
	{
		ret = 0;
	}

	return ret;
}

InferenceEngine::InferRequest MyOpenVINOImpl::CreateInferRequest()
{
	bool ret = true;
	InferenceEngine::InferRequest inferRequest;
	try
	{
		inferRequest = executableNetwork.CreateInferRequest();
	}
	catch (...)
	{
		;
	}

	return inferRequest;
}

bool MyOpenVINOImpl::SetInputData(InferenceEngine::InferRequest  &inferRequest,const std::string &imageName)
{
	bool ret = true;
	cv::Mat imageData = cv::imread(imageName);

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
	bool successed = true;
	std::vector<float> outputVect;
	std::string inputImage;
	try
	{
		InferInfo inferInfo = inferMap[inferID];
		inputImage = inferInfo.inputImage;
		SetInputData(inferInfo.inferRequest, inputImage);
		inferInfo.inferRequest.StartAsync();		
		inferInfo.inferRequest.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
		InferenceEngine::SizeVector dims = inferInfo.inferRequest.GetBlob(outputLayerName)->getTensorDesc().getDims();
		const float* oneHotVector = (inferInfo.inferRequest.GetBlob(outputLayerName))->buffer().as<float*>();
		int dim = dims[1];

		for (int i = 0; i < dim; i++)
		{
			outputVect.push_back(oneHotVector[i]);
		}
	}
	catch (...)
	{
		successed = false;
	}

	if (pCallbackHandler != NULL)
	{
		pCallbackHandler->InferCallBack(inferID, inputImage  , successed,  outputVect);
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

#pragma endregion


// クラスの実態を取得するAPI
DLL IMyOpenVINO* GetInstance(void)
{
	return new MyOpenVINOImpl;
}