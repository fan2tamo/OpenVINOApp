echo off
set buf=%~dp0
set CurrentPath=%buf:~0,-1%
set OpenVINOPath="C:\Program Files (x86)\Intel\openvino_2021\"
set EnvName=""

rem Frameworkは onnxかtfかtf2を設定すること
set Framework=onnx
set DataType=FP16
set ModelName=mnist

rem Pythonを仮想環境上で実行したい場合は仮想環境名をEnvNameにセットすること
if not %EnvName%=="" (
    call conda activate %EnvName%
) else (
    echo **** not use vertual env. ***
)

rem OpenVINOの初期設定を実施
cd /d %OpenVINOPath%\bin
call setupvars.bat

rem model optimizerを実行するために必要なライブラリをインストール
cd %OpenVINOPath%\deployment_tools\model_optimizer\install_prerequisites\
if %framework%==onnx (
    call install_prerequisites_onnx.bat
) else if %framework%==tf (
    call install_prerequisites_tf.bat
) else if %framework%==tf2 (
    call install_prerequisites_tf2.bat
) else (
    echo Frameworkを設定すること！
    pause
    goto end
)
cd ..


rem model optimizerで学習済みモデルをIR形式に変換
if %framework%==onnx (
    echo python mo.py --framework %framework% --input_model "%CurrentPath%\%ModelName%.onnx" --output_dir "%CurrentPath%\OpenVINOApp\OpenVINOAppCpp\Data\Model" --b 1 --data_type %DataType% --scale_values [255,255,255]
    python mo.py --framework %framework% --input_model "%CurrentPath%\%ModelName%.onnx" --output_dir "%CurrentPath%\OpenVINOApp\OpenVINOAppCpp\Data\Model" --b 1 --data_type %DataType% --scale_values [255,255,255]
) else (
    echo python mo.py --framework tf --input_model "%CurrentPath%\%ModelName%.pb" --output_dir "%CurrentPath%\OpenVINOApp\OpenVINOAppCpp\Data\Model" --b 1 --data_type %DataType% --scale_values [255,255,255]
    python mo.py --framework tf --input_model "%CurrentPath%\%ModelName%.pb" --output_dir "%CurrentPath%\OpenVINOApp\OpenVINOAppCpp\Data\Model" --b 1 --data_type %DataType% --scale_values [255,255,255]
) 

:end
cd /d "%CurrentPath%"
if not %EnvName%=="" (
call conda deactivate
)

exit /b
