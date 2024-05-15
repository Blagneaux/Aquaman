using System.Diagnostics;
using System.Net.Sockets;
using NModbus;
using Task = System.Threading.Tasks.Task;
using NationalInstruments;
using NationalInstruments.DAQmx;
using System.Threading.Tasks;


namespace motion2
{
    public partial class Form1 : Form
    {

        // Motion
        private IModbusMaster master;
        private TcpClient tcpClient;
        private Process runningProcess;

        // NI parameters
        private AnalogMultiChannelReader analogInReader;
        private NationalInstruments.DAQmx.Task myTask;
        private NationalInstruments.DAQmx.Task runningTask;
        private AsyncCallback analogCallback;
        private AnalogWaveform<double>[] data;

        public Form1()
        {
            InitializeComponent();
            InitializeModbusTcp();

        }
        private void InitializeModbusTcp()
        {
            tcpClient = new TcpClient();
            try
            {
                tcpClient.Connect("192.168.1.3", 502);
                var factory = new ModbusFactory();
                master = factory.CreateMaster(tcpClient);
                MessageBox.Show("success!");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error connecting to Modbus server: " + ex.Message);
                MessageBox.Show("Unable to connect to Modbus server: " + ex.Message, "Connection Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }


        private async void button1_Click(object sender, EventArgs e)
        {
            int loopCount = 1;
            PerformActions(loopCount);
        }

        private async Task PerformActions(int loopCount)
        {
            await Task.Run(() => InitializeLocation());     //������� setting your initial location
            await Task.Run(() => BackHome());   // �����  back initial location
            LoadFile();  //����csv�켣�ļ�    upload your trajectory data
             //ȷ�����������ص����
            await Task.Run(() => RunPythonScriptWithPyCharmInterpreter(@"D:\PycharmProjects\pythonProject\checkmove.py"));  // check positon before moving 
            Task linkageMotionTask = Task.Run(() => LinkageMotion());   //�������˶�ͬ������
            Task createNIChannelsTask = Task.Run(() => createNIChannels(loopCount));    //�������ɼ�����
            Task readSensorsTask = Task.Run(() => RunPythonScriptWithPyCharmInterpreter("python/readSensors.py"));    // Read the sensors

            await Task.WhenAll(linkageMotionTask, createNIChannelsTask, readSensorsTask);    //����˶������ݲɼ�ͬʱ��ʼ
             //ȷ����������ᵽ���յ�
        
            await Task.Run(() => RunPythonScriptWithPyCharmInterpreter(@"D:\PycharmProjects\pythonProject\checkback.py"));  // check position before back home 

            Task killNIChannelsTask = Task.Run(() => stopNIread());    //ֹͣ�ɼ�����������
            Task killreadSensorsTask = Task.Run(() => killReadSensors());
            await Task.WhenAll(killNIChannelsTask, killreadSensorsTask)
            await Task.Run(() => BackHome());
            Task delayTask = Task.Delay(30000);  // waiting for 30s

        }

        private async Task PerformActions_harmonic(int loopCount)
        {

            if (loopCount == 1)
            {
                await Task.Run(() => InitializeLocation());
            }
            await Task.Run(() => RunPythonScriptWithPyCharmInterpreter(@"D:\PycharmProjects\pythonProject\checkmove_hm.py"));
            Task Ymotion = Task.Run(() => Y_motion());
            Task X2motion = Task.Run(() => X2_motion());
            Task Cmotion = Task.Run(() => C_motion());
            await Task.WhenAll(Ymotion, X2motion, Cmotion);
            await Task.Run(() => RunPythonScriptWithPyCharmInterpreter(@"D:\PycharmProjects\pythonProject\checkback_hm.py"));

        }


        private void InitializeLocation()
        {
            string autoDirFilePath = @"E:\00_DoNotMove\UploadProg\AutoDir.txt";
            string filePath = File.ReadLines(autoDirFilePath).First();

            string[] firstLine = File.ReadLines(filePath).First().Split(',');
            float[] initialLocations = { float.Parse(firstLine[0]), float.Parse(firstLine[2]), float.Parse(firstLine[4]), float.Parse(firstLine[6]) };
            ushort[] addresses = { 25, 31, 37, 43 };

            // ʹ�ö�ȡ�ĳ�ʼλ�����ݸ��¼Ĵ���
            for (int i = 0; i < addresses.Length; i++)
            {
                ModifyRegister(addresses[i], initialLocations[i]);
            }
        }


        private void ModifyRegister(ushort address, float newValue)
        {
            var registers = master.ReadHoldingRegisters(1, address, 2);
            var currentValue = BitConverter.ToSingle(BitConverter.GetBytes(registers[0] + (registers[1] << 16)), 0);
            Console.WriteLine($"Original initial location at address {address}: {currentValue}");

            byte[] floatBytes = BitConverter.GetBytes(newValue);
            ushort[] payload = { BitConverter.ToUInt16(floatBytes, 0), BitConverter.ToUInt16(floatBytes, 2) };
            master.WriteMultipleRegisters(1, address, payload);

            registers = master.ReadHoldingRegisters(1, address, 2);
            var verifyValue = BitConverter.ToSingle(BitConverter.GetBytes(registers[0] + (registers[1] << 16)), 0);
            Console.WriteLine($"Successfully modified initial location to {verifyValue} at address {address}");
        }

        private void BackHome()
        {
            // Reset X axis
            ToggleRegisterBit(0, 10);
            // Reset Y axis
            ToggleRegisterBit(0, 15);
            // Reset x2 axis
            ToggleRegisterBit(2, 4);
            // Reset C axis
            ToggleRegisterBit(2, 9);
        }

        private void ToggleRegisterBit(ushort address, int bitPosition)
        {
            ushort currentValue = master.ReadHoldingRegisters(1, address, 1)[0];
            ushort newValue = SetBit(currentValue, bitPosition, true);
            master.WriteSingleRegister(1, address, newValue);
            Thread.Sleep(100);
            newValue = SetBit(newValue, bitPosition, false);
            master.WriteSingleRegister(1, address, newValue);
        }

        private ushort SetBit(ushort value, int bitPosition, bool set)
        {
            if (set)
                return (ushort)(value | (1 << bitPosition));
            else
                return (ushort)(value & ~(1 << bitPosition));
        }

        private void LoadFile()
        {
            string exePath = @"C:\Users\MSI\Desktop\���ݵ���3.exe";
            try
            {
                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = exePath,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using (Process process = Process.Start(startInfo))
                {
                    string output = process.StandardOutput.ReadToEnd();
                    string errors = process.StandardError.ReadToEnd();
                    process.WaitForExit();

                    if (!string.IsNullOrEmpty(errors))
                    {
                        MessageBox.Show(errors, "������Ϣ", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }

                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"�޷���������: {ex.Message}", "����", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }


        private void RunPythonScriptWithPyCharmInterpreter(string scriptPath)
        {
            // Assuming the path to PyCharm executable is known and constant
            string pyCharmPath = @"D:\ana\python.exe";

            try
            {
                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = pyCharmPath,
                    Arguments = $"\"{scriptPath}\"",
                    UseShellExecute = false, // ��ҪΪ false ���ض������
                    CreateNoWindow = true, // ����������
                };

                Process process = new Process
                {
                    StartInfo = startInfo
                };

                process.Start();
                process.WaitForExit();  // Optionally wait for the script to complete
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to start PyCharm: {ex.Message}");
            }
        }

        private void record_traj()
        {
            string exePath = @"C:\Users\MSI\Desktop\���ݼ�¼.exe";
            try
            {
                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = exePath,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using (Process process = Process.Start(startInfo))
                {
                    string output = process.StandardOutput.ReadToEnd();
                    string errors = process.StandardError.ReadToEnd();
                    process.WaitForExit();


                    if (!string.IsNullOrEmpty(errors))
                    {
                        MessageBox.Show(errors, "������Ϣ", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }

                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"�޷���������: {ex.Message}", "����", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
        private void Stoprecord_traj(Process process)
        {
            if (process != null && !process.HasExited)
            {
                try
                {
                    process.Kill();
                    process.WaitForExit(); // �ȴ�������ֹ
                    MessageBox.Show("�����ѳɹ���ֹ��", "��ֹ�ɹ�", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"��ֹ����ʱ����: {ex.Message}", "����", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    process.Dispose();
                }
            }
            else
            {
                MessageBox.Show("�����Ѿ���ֹ�򲻴��ڡ�", "������Ч", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }


        private void LinkageMotion()    // if use make sure all the linkage switches on, or it would be only activate X axis
        {
            ushort address = 0;
            int bitPosition = 9; // ��9λ������

            // ��ȡ��ǰ�Ĵ�����ֵ
            ushort[] registers = master.ReadHoldingRegisters(1, address, 1);
            if (registers != null && registers.Length > 0)
            {
                ushort currentValue = registers[0];
                Console.WriteLine($"Initial value: {Convert.ToString(currentValue, 2)}");

                // ����λΪ1
                ushort newValue = SetBit(currentValue, bitPosition, true);
                master.WriteSingleRegister(1, address, newValue);

                // �ٴζ�ȡ����֤����λΪ1
                registers = master.ReadHoldingRegisters(1, address, 1);
                if (registers.Length > 0)
                {
                    Console.WriteLine($"Value after setting bit to 1: {Convert.ToString(registers[0], 2)}");
                }
                else
                {
                    Console.WriteLine("Failed to read the register after setting bit to 1");
                }

                Thread.Sleep(100); // time gap to reset

                // ����λΪ0
                newValue = SetBit(newValue, bitPosition, false);
                master.WriteSingleRegister(1, address, newValue);

                // �ٴζ�ȡ����֤����λΪ0
                registers = master.ReadHoldingRegisters(1, address, 1);
                if (registers.Length > 0)
                {
                    Console.WriteLine($"Value after setting bit to 0: {Convert.ToString(registers[0], 2)}");
                }
                else
                {
                    Console.WriteLine("Failed to read the register after setting bit to 0");
                }
            }
            else
            {
                Console.WriteLine("Failed to read the register");
            }
        }

        private void X_motion()    
        {
            ushort address = 0;
            int bitPosition = 9; // ��9λ������

            // ��ȡ��ǰ�Ĵ�����ֵ
            ushort[] registers = master.ReadHoldingRegisters(1, address, 1);
            if (registers != null && registers.Length > 0)
            {
                ushort currentValue = registers[0];
                Console.WriteLine($"Initial value: {Convert.ToString(currentValue, 2)}");

                // ����λΪ1
                ushort newValue = SetBit(currentValue, bitPosition, true);
                master.WriteSingleRegister(1, address, newValue);

                // �ٴζ�ȡ����֤����λΪ1
                registers = master.ReadHoldingRegisters(1, address, 1);
                if (registers.Length > 0)
                {
                    Console.WriteLine($"Value after setting bit to 1: {Convert.ToString(registers[0], 2)}");
                }
                else
                {
                    Console.WriteLine("Failed to read the register after setting bit to 1");
                }

                Thread.Sleep(100); // time gap to reset

                // ����λΪ0
                newValue = SetBit(newValue, bitPosition, false);
                master.WriteSingleRegister(1, address, newValue);

                // �ٴζ�ȡ����֤����λΪ0
                registers = master.ReadHoldingRegisters(1, address, 1);
                if (registers.Length > 0)
                {
                    Console.WriteLine($"Value after setting bit to 0: {Convert.ToString(registers[0], 2)}");
                }
                else
                {
                    Console.WriteLine("Failed to read the register after setting bit to 0");
                }
            }
            else
            {
                Console.WriteLine("Failed to read the register");
            }
        }

        private void Y_motion()
        {
            ushort address = 0;
            int bitPosition = 14;

            // ��ȡ��ǰ�Ĵ�����ֵ
            ushort[] registers = master.ReadHoldingRegisters(1, address, 1);
            if (registers != null && registers.Length > 0)
            {
                ushort currentValue = registers[0];
                Console.WriteLine($"Initial value: {Convert.ToString(currentValue, 2)}");

                // ����λΪ1
                ushort newValue = SetBit(currentValue, bitPosition, true);
                master.WriteSingleRegister(1, address, newValue);

                // �ٴζ�ȡ����֤����λΪ1
                registers = master.ReadHoldingRegisters(1, address, 1);
                if (registers.Length > 0)
                {
                    Console.WriteLine($"Value after setting bit to 1: {Convert.ToString(registers[0], 2)}");
                }
                else
                {
                    Console.WriteLine("Failed to read the register after setting bit to 1");
                }

                Thread.Sleep(100); // time gap to reset

                // ����λΪ0
                newValue = SetBit(newValue, bitPosition, false);
                master.WriteSingleRegister(1, address, newValue);

                // �ٴζ�ȡ����֤����λΪ0
                registers = master.ReadHoldingRegisters(1, address, 1);
                if (registers.Length > 0)
                {
                    Console.WriteLine($"Value after setting bit to 0: {Convert.ToString(registers[0], 2)}");
                }
                else
                {
                    Console.WriteLine("Failed to read the register after setting bit to 0");
                }
            }
            else
            {
                Console.WriteLine("Failed to read the register");
            }
        }

        private void X2_motion()
        {
            ushort address = 2;
            int bitPosition = 3; 

            // ��ȡ��ǰ�Ĵ�����ֵ
            ushort[] registers = master.ReadHoldingRegisters(1, address, 1);
            if (registers != null && registers.Length > 0)
            {
                ushort currentValue = registers[0];
                Console.WriteLine($"Initial value: {Convert.ToString(currentValue, 2)}");

                // ����λΪ1
                ushort newValue = SetBit(currentValue, bitPosition, true);
                master.WriteSingleRegister(1, address, newValue);

                // �ٴζ�ȡ����֤����λΪ1
                registers = master.ReadHoldingRegisters(1, address, 1);
                if (registers.Length > 0)
                {
                    Console.WriteLine($"Value after setting bit to 1: {Convert.ToString(registers[0], 2)}");
                }
                else
                {
                    Console.WriteLine("Failed to read the register after setting bit to 1");
                }

                Thread.Sleep(100); // time gap to reset

                // ����λΪ0
                newValue = SetBit(newValue, bitPosition, false);
                master.WriteSingleRegister(1, address, newValue);

                // �ٴζ�ȡ����֤����λΪ0
                registers = master.ReadHoldingRegisters(1, address, 1);
                if (registers.Length > 0)
                {
                    Console.WriteLine($"Value after setting bit to 0: {Convert.ToString(registers[0], 2)}");
                }
                else
                {
                    Console.WriteLine("Failed to read the register after setting bit to 0");
                }
            }
            else
            {
                Console.WriteLine("Failed to read the register");
            }
        }

        private void C_motion()
        {
            ushort address = 2;
            int bitPosition = 8; 

            // ��ȡ��ǰ�Ĵ�����ֵ
            ushort[] registers = master.ReadHoldingRegisters(1, address, 1);
            if (registers != null && registers.Length > 0)
            {
                ushort currentValue = registers[0];
                Console.WriteLine($"Initial value: {Convert.ToString(currentValue, 2)}");

                // ����λΪ1
                ushort newValue = SetBit(currentValue, bitPosition, true);
                master.WriteSingleRegister(1, address, newValue);

                // �ٴζ�ȡ����֤����λΪ1
                registers = master.ReadHoldingRegisters(1, address, 1);
                if (registers.Length > 0)
                {
                    Console.WriteLine($"Value after setting bit to 1: {Convert.ToString(registers[0], 2)}");
                }
                else
                {
                    Console.WriteLine("Failed to read the register after setting bit to 1");
                }

                Thread.Sleep(100); // time gap to reset

                // ����λΪ0
                newValue = SetBit(newValue, bitPosition, false);
                master.WriteSingleRegister(1, address, newValue);

                // �ٴζ�ȡ����֤����λΪ0
                registers = master.ReadHoldingRegisters(1, address, 1);
                if (registers.Length > 0)
                {
                    Console.WriteLine($"Value after setting bit to 0: {Convert.ToString(registers[0], 2)}");
                }
                else
                {
                    Console.WriteLine("Failed to read the register after setting bit to 0");
                }
            }
            else
            {
                Console.WriteLine("Failed to read the register");
            }
        }

        private void createNIChannels(int recordNum)
        {
            if (runningTask == null)
            {
                try
                {
                    // Create a new task
                    myTask = new NationalInstruments.DAQmx.Task();

                    // Create VI channels 

                    myTask.AIChannels.CreateVoltageChannel("Dev3/ai1", "Fx",
                         AITerminalConfiguration.Rse, -10, 10, AIVoltageUnits.Volts);
                    myTask.AIChannels.CreateVoltageChannel("Dev3/ai2", "Fy",
                        AITerminalConfiguration.Rse, -10, 10, AIVoltageUnits.Volts);
                    myTask.AIChannels.CreateVoltageChannel("Dev3/ai3", "Fz",
                        AITerminalConfiguration.Rse, -10, 10, AIVoltageUnits.Volts);
                    myTask.AIChannels.CreateVoltageChannel("Dev3/ai4", "Mx",
                        AITerminalConfiguration.Rse, -10, 10, AIVoltageUnits.Volts);
                    myTask.AIChannels.CreateVoltageChannel("Dev3/ai5", "My",
                        AITerminalConfiguration.Rse, -10, 10, AIVoltageUnits.Volts);
                    myTask.AIChannels.CreateVoltageChannel("Dev3/ai6", "Mz",
                        AITerminalConfiguration.Rse, -10, 10, AIVoltageUnits.Volts);
                    myTask.AIChannels.CreateVoltageChannel("Dev3/ai7", "Sync",
                        AITerminalConfiguration.Rse, -10, 10, AIVoltageUnits.Volts);

                    // Configure the timing parameters
                    myTask.Timing.ConfigureSampleClock("", 1000, SampleClockActiveEdge.Rising,
                        SampleQuantityMode.ContinuousSamples, 100);

                    // Configure TDMS Logging
                    // if (this.txtPath.Text.Trim().Length > 0)
                    //{
                    // String filePath = this.txtPath.Text + "\\Data\\Force\\" + recordNum.ToString("D3") + "_force.tdms";
                    String filePath = "D:\\dehan\\Proj\\NI_data\\" + recordNum.ToString("D3") + "_force.tdms";
                    myTask.ConfigureLogging(filePath, TdmsLoggingOperation.CreateOrReplace, LoggingMode.LogAndRead, "Group Name");
                    // }

                    // Verify the Task
                    myTask.Control(TaskAction.Verify);

                    runningTask = myTask;
                    analogInReader = new AnalogMultiChannelReader(myTask.Stream);
                    analogCallback = new AsyncCallback(AnalogInCallback);

                    // Use SynchronizeCallbacks to specify that the object 
                    // marshals callbacks across threads appropriately.
                    analogInReader.SynchronizeCallbacks = true;
                    analogInReader.BeginReadWaveform(100, analogCallback, myTask);

                }
                catch (DaqException exception)
                {
                    MessageBox.Show(exception.Message);
                    runningTask = null;
                    myTask.Dispose();
                }
            }
        }

        private void AnalogInCallback(IAsyncResult ar)
        {
            try
            {
                if (runningTask != null && runningTask == ar.AsyncState)
                {
                    // Read the available data from the channels
                    data = analogInReader.EndReadWaveform(ar);

                    // Plot data
                    /*                    this.waveformGraph.PlotWaveformsAppend(data);*/
                    //this.globalwaveformGraph.PlotWaveformAppend(data[2]);


                    analogInReader.BeginMemoryOptimizedReadWaveform(100, analogCallback, myTask, data);
                }
            }
            catch (DaqException exception)
            {
                // Display Errors
                MessageBox.Show(exception.Message);
                runningTask = null;
                myTask.Dispose();
            }
        }

        private void stopNIread()
        {
            if (runningTask != null)
            {
                // Dispose of the task
                runningTask = null;
                myTask.Dispose();
            }
        }

        private void killReadSensors(Task pythonTask)
        {
            pythonTask.Dispose();
        }

        private async void button2_Click(object sender, EventArgs e)
        {
            int numberOfLoops = 2;
            for (int i = 1; i <= numberOfLoops; i++)
            {
                if (i % 2 != 0)
                {
                    /*                    GenerateAndSaveTrajectory(0.1+0.02*i, 2, 0.3, 300, 30+0.1*i, 0.1+0.01*i, 0.1+0.005*i);
                                        await Task.Delay(10000);*/
                    await PerformActions(i);
                }
                else
                {
                    await PerformActions(i);
                }


            }
        }

        private void GenerateAndSaveTrajectory(double f, int NT, double vx, int N, double thetam, double hm, double sm)
        {
            double T = 1.0 / f;  // Total time for one period
            double[] t = Enumerable.Range(0, N).Select(i => i * T / (N - 1)).ToArray();
            double[] time_step = new double[t.Length];
            for (int i = 1; i < t.Length; i++)
            {
                time_step[i] = T / N * 1000;
            }

            double[] theta = t.Select(time => thetam * Math.Sin(2 * Math.PI * f * time)).ToArray();
            double[] y = t.Select(time => hm * Math.Sin(2 * Math.PI * f * time) * 1000).ToArray();
            double[] x = t.Select(time => vx * time * 1000).ToArray();
            double[] x2 = t.Select(time => sm * Math.Sin(2 * Math.PI * f * time) / 2 * 1000 + 80).ToArray();

            if (NT > 1)
            {
                for (int i = 1; i < NT; i++)
                {
                    time_step = time_step.Concat(time_step.Skip(1)).ToArray();
                    y = y.Concat(y.Skip(1)).ToArray();
                    x2 = x2.Concat(x2.Skip(1)).ToArray();
                    theta = theta.Concat(theta.Skip(1)).ToArray();
                }
            }

            double[] ttotal = Enumerable.Range(0, theta.Length).Select(i => i * T / (theta.Length - 1)).ToArray();
            x = ttotal.Select(time => vx * time * 1000).ToArray();

            var path = x.Zip(time_step, (xi, ti) => new { X = xi, Time = ti })
                        .Zip(y, (xt, yi) => $"{xt.X},{xt.Time},{yi},{xt.Time}")
                        .Zip(x2, (xy, x2i) => $"{xy},{x2i},{xy.Split(',')[1]}")
                        .Zip(theta, (x2theta, thetai) => $"{x2theta},{thetai},{x2theta.Split(',')[3]}");

            string filePath = @"D:\dehan\traj_test\path.csv";
            using (StreamWriter file = new StreamWriter(filePath))
            {
                foreach (var line in path)
                {
                    file.WriteLine(line);
                }
            }
        }

        private async void button3_Click(object sender, EventArgs e)
        {
            await Task.Run(() => InitializeLocation());
            await Task.Run(() => BackHome());   
            LoadFile();
            
            int Count = 1;
            Task.Run(() => createNIChannels(Count));    // start recording sensor data

            int Loops = 3;
            for (int i = 1; i <= Loops; i++)
            {
                if (i == 1) 
                {
                    Task hm = Task.Run(() => PerformActions_harmonic(i));
                    Task Xmove = Task.Run(() => X_motion());
                    await Task.WhenAll(hm, Xmove);
                }

                if (i > 1) 
                {
                    await PerformActions_harmonic(i);
                }


            }
            Task delayTask = Task.Delay(5000);  // waiting for 5s
            Task stopNItask = Task.Run(() => stopNIread());
            await Task.WhenAll(delayTask, stopNItask);
            await Task.Run(() => BackHome());
        }
    }       
}

