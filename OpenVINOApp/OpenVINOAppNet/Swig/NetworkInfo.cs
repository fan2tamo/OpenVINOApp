//------------------------------------------------------------------------------
// <auto-generated />
//
// This file was automatically generated by SWIG (http://www.swig.org).
// Version 4.0.2
//
// Do not make changes to this file unless you know what you are doing--modify
// the SWIG interface file instead.
//------------------------------------------------------------------------------


public class NetworkInfo : global::System.IDisposable {
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;
  protected bool swigCMemOwn;

  internal NetworkInfo(global::System.IntPtr cPtr, bool cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  internal static global::System.Runtime.InteropServices.HandleRef getCPtr(NetworkInfo obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }

  ~NetworkInfo() {
    Dispose(false);
  }

  public void Dispose() {
    Dispose(true);
    global::System.GC.SuppressFinalize(this);
  }

  protected virtual void Dispose(bool disposing) {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwn) {
          swigCMemOwn = false;
          MyOpenVINOPINVOKE.delete_NetworkInfo(swigCPtr);
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
    }
  }

  public string modelName {
    set {
      MyOpenVINOPINVOKE.NetworkInfo_modelName_set(swigCPtr, value);
      if (MyOpenVINOPINVOKE.SWIGPendingException.Pending) throw MyOpenVINOPINVOKE.SWIGPendingException.Retrieve();
    } 
    get {
      string ret = MyOpenVINOPINVOKE.NetworkInfo_modelName_get(swigCPtr);
      if (MyOpenVINOPINVOKE.SWIGPendingException.Pending) throw MyOpenVINOPINVOKE.SWIGPendingException.Retrieve();
      return ret;
    } 
  }

  public Layout inputLayout {
    set {
      MyOpenVINOPINVOKE.NetworkInfo_inputLayout_set(swigCPtr, (int)value);
    } 
    get {
      Layout ret = (Layout)MyOpenVINOPINVOKE.NetworkInfo_inputLayout_get(swigCPtr);
      return ret;
    } 
  }

  public Precision inputPrecision {
    set {
      MyOpenVINOPINVOKE.NetworkInfo_inputPrecision_set(swigCPtr, (int)value);
    } 
    get {
      Precision ret = (Precision)MyOpenVINOPINVOKE.NetworkInfo_inputPrecision_get(swigCPtr);
      return ret;
    } 
  }

  public Layout outputLayout {
    set {
      MyOpenVINOPINVOKE.NetworkInfo_outputLayout_set(swigCPtr, (int)value);
    } 
    get {
      Layout ret = (Layout)MyOpenVINOPINVOKE.NetworkInfo_outputLayout_get(swigCPtr);
      return ret;
    } 
  }

  public Precision outputPrecision {
    set {
      MyOpenVINOPINVOKE.NetworkInfo_outputPrecision_set(swigCPtr, (int)value);
    } 
    get {
      Precision ret = (Precision)MyOpenVINOPINVOKE.NetworkInfo_outputPrecision_get(swigCPtr);
      return ret;
    } 
  }

  public DeviceVector devices {
    set {
      MyOpenVINOPINVOKE.NetworkInfo_devices_set(swigCPtr, DeviceVector.getCPtr(value));
    } 
    get {
      global::System.IntPtr cPtr = MyOpenVINOPINVOKE.NetworkInfo_devices_get(swigCPtr);
      DeviceVector ret = (cPtr == global::System.IntPtr.Zero) ? null : new DeviceVector(cPtr, false);
      return ret;
    } 
  }

  public bool isMultiDevices {
    set {
      MyOpenVINOPINVOKE.NetworkInfo_isMultiDevices_set(swigCPtr, value);
    } 
    get {
      bool ret = MyOpenVINOPINVOKE.NetworkInfo_isMultiDevices_get(swigCPtr);
      return ret;
    } 
  }

  public uint threadNum {
    set {
      MyOpenVINOPINVOKE.NetworkInfo_threadNum_set(swigCPtr, value);
    } 
    get {
      uint ret = MyOpenVINOPINVOKE.NetworkInfo_threadNum_get(swigCPtr);
      return ret;
    } 
  }

  public uint inferRequestNum {
    set {
      MyOpenVINOPINVOKE.NetworkInfo_inferRequestNum_set(swigCPtr, value);
    } 
    get {
      uint ret = MyOpenVINOPINVOKE.NetworkInfo_inferRequestNum_get(swigCPtr);
      return ret;
    } 
  }

  public NetworkInfo() : this(MyOpenVINOPINVOKE.new_NetworkInfo(), true) {
  }

}