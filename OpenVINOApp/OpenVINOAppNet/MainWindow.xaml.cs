using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace OpenVINOAppNet
{
    
   public  class InferCallBackHandler : CallbackHandlerBase
    {
        private TextBox textBox;

        public InferCallBackHandler(TextBox tbx)
        {
            textBox = tbx;
        }

        // dll側から呼ばれる
        public override void InferCallBack(int inferID, bool isSuccessed, floatVector results)
        {
            textBox.Dispatcher.Invoke((Action)(() =>
            {
                int i = 0;
                foreach (float result in results)
                {
                    textBox.Text += $"inferID[{inferID}] : [{i}] = {result:E4}" + System.Environment.NewLine;
                    i += 1;
                }
                textBox.Text += $"---------------------" + System.Environment.NewLine;
            }));
        }
    }
    

    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        IMyOpenVINO instance = null;

        public MainWindow()
        {
            InitializeComponent();
            inputImageDir.Text = System.AppDomain.CurrentDomain.BaseDirectory + "Image";
            instance = MyOpenVINO.GetInstance();
            var devices = instance.GetAvailableDevices();

            if(devices.Contains(Device.CPU))
            {
                rbtn_CPU.IsEnabled = true;
                rbtn_CPU.IsChecked = true;
            }
            if (devices.Contains(Device.GPU))
            {
                rbtn_GPU.IsEnabled = true;
                rbtn_GPU.IsChecked = true;
            }
            if (devices.Contains(Device.MYRIAD))
            {
                rbtn_VPU.IsEnabled = true;
                rbtn_VPU.IsChecked = true;
            }
        }

        private void Click_InferSync(object sender, RoutedEventArgs e)
        {
            string deviceStr = GetCheckedRadioButtonString();
            NetworkInfo networkInfo = new NetworkInfo();
            networkInfo.modelName = @"Model\mnist.xml";
            networkInfo.inputLayout = Layout.NCHW;
            networkInfo.inputPrecision = Precision.U8;
            networkInfo.outputLayout = Layout.NC;
            networkInfo.outputPrecision = Precision.FP32;
            networkInfo.threadNum = 1;
            networkInfo.isMultiDevices = false;
            networkInfo.devices.Add(ConvertString2Device(deviceStr));

            string[] images = System.IO.Directory.GetFiles(inputImageDir.Text);
            List<string> inputImageFiles = new List<string>();
            inputImageFiles.AddRange(images);

            
            // instance.GetAvailableDevices();
            instance.Initialize(networkInfo);
          
            foreach (string inputImage in inputImageFiles)
            {             
                instance.InferSync(inputImage);

                
                floatVector outputVec = instance.InferSync(inputImage);
                textBox.Text += $"inputImageFile : {System.IO.Path.GetFileName(inputImage)}" + System.Environment.NewLine;
                int i = 0;
                foreach(float output in outputVec)
                {
                    //textBox.Text += string.Format("{0} : {1:f4}", i, outputVec[i]) + System.Environment.NewLine;
                    textBox.Text +=  $"[{i}] :  {output:E4}" + System.Environment.NewLine;
                    i++;
                }
            }
        }

        private void Click_InferASync(object sender, RoutedEventArgs e)
        {
            InferCallBackHandler obj = new InferCallBackHandler(textBox);
            string deviceStr = GetCheckedRadioButtonString();
            NetworkInfo networkInfo = new NetworkInfo();
            networkInfo.modelName = @"Model\mnist.xml";
            networkInfo.inputLayout = Layout.NCHW;
            networkInfo.inputPrecision = Precision.U8;
            networkInfo.outputLayout = Layout.NC;
            networkInfo.outputPrecision = Precision.FP32;
            networkInfo.threadNum = 1;
            networkInfo.isMultiDevices = false;
            networkInfo.devices.Add(ConvertString2Device(deviceStr));

            string[] images = System.IO.Directory.GetFiles(inputImageDir.Text);
            List<string> inputImageFiles = new List<string>();
            inputImageFiles.AddRange(images);


            // instance.GetAvailableDevices();
            instance.Initialize(networkInfo);
            instance.SetInferCallBack(obj);

            foreach (string inputImage in inputImageFiles)
            {
                int inferID = instance.InferASync(inputImage);
            }
        }

        private void Click_GetAvailableDevices(object sender, RoutedEventArgs e)
        {
            var devices = instance.GetAvailableDevices();
            textBox.Text += "***AvailableDevices***" + System.Environment.NewLine;
            foreach (Device d in devices)
            {
                textBox.Text += $"{d}" + System.Environment.NewLine;
            }

            if (devices.Contains(Device.CPU))
            {
                rbtn_CPU.IsEnabled = true;
            }
            if (devices.Contains(Device.GPU))
            {
                rbtn_GPU.IsEnabled = true;
            }
            if (devices.Contains(Device.MYRIAD))
            {
                rbtn_VPU.IsEnabled = true;
            }
        }

        private Device ConvertString2Device(string deviceStr)
        {
            Device device;

            switch(deviceStr)
            {
                case "CPU":
                    device = Device.CPU;
                    break;

                case "GPU":
                    device = Device.GPU;
                    break;

                case "MYRIAD":
                    device = Device.MYRIAD;
                    break;

                default:
                    throw new Exception("not supported device");
            }

            return device;
        }

        private string GetCheckedRadioButtonString()
        {
            if(rbtn_CPU.IsChecked == true)
            {
                return (string)rbtn_CPU.Content;
            }

            if (rbtn_GPU.IsChecked == true)
            {
                return (string)rbtn_GPU.Content;
            }

            if (rbtn_VPU.IsChecked == true)
            {
                return (string)rbtn_VPU.Content;
            }

            throw new Exception();

        }


    }
}
