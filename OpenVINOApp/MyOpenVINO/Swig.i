/* File : Swig.i */

%include <windows.i> 
%include <std_string.i>
%include <std_vector.i>
%include <arrays_csharp.i>

%module (directors="1") MyOpenVINO
 
%{
#include "MyOpenVINO.h"
%}
 
%feature("director") CallbackHandlerBase;

/* Let's just grab the original header file here */
%include "MyOpenVINO.h"

%template(DeviceVector) std::vector<Device>;
%template(floatVector) std::vector<float>;