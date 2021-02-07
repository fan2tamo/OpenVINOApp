/* File : Swig.i */

%include <windows.i> 
%include <wchar.i>
%include <std_wstring.i>
%include <std_vector.i>
%include <arrays_csharp.i>
/*
%define %cs_callback(TYPE, CSTYPE)
%typemap(ctype) TYPE, TYPE& "void*"
%typemap(in) TYPE %{ $1 = (TYPE)$input; %}
%typemap(in) TYPE& %{ $1 = (TYPE*)&$input; %}
%typemap(imtype, out="IntPtr") TYPE, TYPE& "CSTYPE"
%typemap(cstype, out="IntPtr") TYPE, TYPE& "CSTYPE"
%typemap(csin) TYPE, TYPE& "$csinput"
%enddef
%define %cs_callback2(TYPE, CTYPE, CSTYPE)
%typemap(ctype) TYPE "CTYPE"
%typemap(in) TYPE %{ $1 = (TYPE)$input; %}
%typemap(imtype, out="IntPtr") TYPE "CSTYPE"
%typemap(cstype, out="IntPtr") TYPE "CSTYPE"
%typemap(csin) TYPE "$csinput"
%enddef

%cs_callback(InferCallBack, InferCallBackCpp)
*/

%module (directors="1") MyOpenVINO

 
%{
#include "MyOpenVINO.h"
%}
 
//%include "arrays_csharp.i"
//%apply float INPUT[] {float* results}
%feature("director") CallbackHandlerBase;

/* Let's just grab the original header file here */
%include "MyOpenVINO.h"

%template(DeviceVector) std::vector<Device>;
%template(floatVector) std::vector<float>;