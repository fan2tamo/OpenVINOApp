// OpenVINOAppCpp.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <iostream>
#include <Windows.h>
#include "../MyOpenVINO/MyOpenVINO.h"

class CallbackHandler: public CallbackHandlerBase
{
public:
    void InferCallBack(int inferID, const std::string& str, bool isSuccessed, const std::vector<float> &results) override
    {
        for (int i = 0; i < results.size(); i++)
        {
            printf("inferID = %d : [%2d] = %lf\n", inferID,  i, results[i]);
        }
    }
};

int main()
{
    std::cout << "Hello World!\n";
    
    HMODULE hHandle = LoadLibrary(L"MyOpenVINO.dll");
    if (hHandle == INVALID_HANDLE_VALUE || hHandle == NULL)
    {
        printf("Error : LoadLibrary()\r\n");
        printf("GetLastError() = 0x%X\r\n", GetLastError());
        exit(0);
    }

    GetInstanceFuncPointer func = (GetInstanceFuncPointer)GetProcAddress(hHandle, "GetInstance");
    if (func == NULL)
    {
        printf("Error : GetProcAddress()\r\n");
        exit(0);
    }
    
    std::vector<std::string> inputImageFiles;
    inputImageFiles.push_back("Image/img_0.png");
    inputImageFiles.push_back("Image/img_1.png");
    inputImageFiles.push_back("Image/img_2.png");
    inputImageFiles.push_back("Image/img_3.png");
    inputImageFiles.push_back("Image/img_4.png");
    inputImageFiles.push_back("Image/img_5.png");
    inputImageFiles.push_back("Image/img_6.png");
    inputImageFiles.push_back("Image/img_7.png");
    inputImageFiles.push_back("Image/img_8.png");
    inputImageFiles.push_back("Image/img_9.png");

    NetworkInfo networkInfo;
    networkInfo.modelName = "Model\\mnist.xml";
    networkInfo.inputLayout = Layout::NCHW;
    networkInfo.inputPrecision = Precision::U8;
    networkInfo.outputLayout = Layout::NC;
    networkInfo.outputPrecision = Precision::FP32;
    networkInfo.threadNum = 4;
    networkInfo.isMultiDevices = false;
    networkInfo.devices.push_back(Device::CPU);

    IMyOpenVINO* instance = func();
    instance->GetAvailableDevices();
    instance->Initialize(networkInfo);
    CallbackHandler inf;;
    instance->SetInferCallBack(inf);

    for (std::string inputImageFile : inputImageFiles)
    {
        instance->InferASync(inputImageFile);
        /*
        std::vector<float> outputVec = instance->InferSync(inputImageFile);
        printf("inputImageFile : %ls\r\n", inputImageFile.c_str());
        for (int i = 0; i < outputVec.size(); i++)
        {
            printf("[%d] : %lf\r\n", i, outputVec[i]);
        }
        */
    }
   instance->WaitForEndOfInfer();
}

// プログラムの実行: Ctrl + F5 または [デバッグ] > [デバッグなしで開始] メニュー
// プログラムのデバッグ: F5 または [デバッグ] > [デバッグの開始] メニュー

// 作業を開始するためのヒント: 
//    1. ソリューション エクスプローラー ウィンドウを使用してファイルを追加/管理します 
//   2. チーム エクスプローラー ウィンドウを使用してソース管理に接続します
//   3. 出力ウィンドウを使用して、ビルド出力とその他のメッセージを表示します
//   4. エラー一覧ウィンドウを使用してエラーを表示します
//   5. [プロジェクト] > [新しい項目の追加] と移動して新しいコード ファイルを作成するか、[プロジェクト] > [既存の項目の追加] と移動して既存のコード ファイルをプロジェクトに追加します
//   6. 後ほどこのプロジェクトを再び開く場合、[ファイル] > [開く] > [プロジェクト] と移動して .sln ファイルを選択します

