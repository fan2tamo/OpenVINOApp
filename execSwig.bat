echo off
set buf=%~dp0
set CurrentPath=%buf:~0,-1%
set SwigPath=C:\Program Files\swigwin-4.0.2

cd /d %CurrentPath%
"%SwigPath%\swig.exe" -c++ -csharp -outdir OpenVINOApp\OpenVINOAppNet\Swig\ -cppext cpp OpenVINOApp\MyOpenVINO\Swig.i
