// ----------------------------------------------------------------------------------------------------------------------------------------------
// Sandbox for INA
// ----------------------------------------------------------------------------------------------------------------------------------------------
BDIM flow;
Body circle;
FloodPlot flood;

Table parameters;

int _n=(int)pow(2, 7);
float _L=_n/20;
float Re = 10000;     // Reynolds number                                                            Read from coordinates file generates by python GPR code
PrintWriter outputFullMap;
float dt;
float t = 0;
int line = 0;
ArrayList<float[]> pressureDataList = new ArrayList<>();
int numTimeStep = 0;

void setup() {
  size(1400, 700);                             // display window size
  int n=_n;                                   // number of grid points      n = 1m
  float L = _L;                            // length-scale in grid units    L = 5cm, so L = n/20
  Window view = new Window(2*n, n);

  circle = new CircleBody(4*n/7, L, L, view);     // define geom
  flow = new BDIM(2*n, n, 0, circle, (float)L/Re, true, 1);             // solve for flow using BDIM
  flood = new FloodPlot(view);               // initialize a flood plot...
  flood.setLegend("vorticity", -.5, .5);       //    and its legend
  
  outputFullMap = createWriter("E:/benchmark_SINDy/FullPressureMapRe10000_h6_extended.csv");
}
void draw() {
  if (flow.QUICK) {
    dt = flow.checkCFL();
    flow.dt = dt;//modify
  }
  t += dt;
  println(t);
  circle.follow();
  flow.update(circle);
  flow.update2();
  flood.display(flow.u.curl());              // compute and display vorticity
  circle.display();                            // display the body


  // Store pressure data for every point in the window
  float[] pressureData = new float[2*_n * _n];
  int index = 0;
  for (int i = 0; i < 2*_n; i++) {
    for (int j = 0; j < _n; j++) {
      pressureData[index] = flow.p.extract(i, j);
      index++;
    }
  }
  // Add the pressure data array to the list for this time step
  pressureDataList.add(pressureData);
  numTimeStep++;
  
  if (t > 10000) {  // close and save everything when t>Time
    dataAdd();
    exit();
  }
  saveFrame("saved/frame-####.png");
}

void dataAdd() {
  // Write the pressure data to the CSV file as a single column
  for (int i = 0; i < _n * 2*_n; i++) {
    for (int tStep = 0; tStep < numTimeStep; tStep++) {
      println(_n*2*_n, numTimeStep, i, tStep);
      float[] pressure = pressureDataList.get(tStep);
      outputFullMap.print(pressure[i]);
      if (tStep < numTimeStep - 1) {
        outputFullMap.print(","); // Separate values with newlines
      }
    }
    outputFullMap.println(); // Move to the next row
  }
  outputFullMap.close();
}

// ----------------------------------------------------------------------------------------------------------------------------------------------
// Digital twin of the experiments held in China to generate a dataset for future data fusion between physical sensors and the result of HAACHAMA
// ----------------------------------------------------------------------------------------------------------------------------------------------
//BDIM flow;
//Body circle;
//FloodPlot flood;

//Table parameters;

//int _n=(int)pow(2, 7);
//float _L=_n/20;
//float speed = 1;    //grid per time step
//float Re; // = 10000;     // Reynolds number                                                            Read from coordinates file generates by python GPR code
//float origin; // = 1*_L;      //mean position                                                            Read from coordinates file generates by python GPR code
//float Ay = _L;      //spatial oscillations amplitude                                               Read from coordinates file generates by python GPR code
//float Fy = 3*speed/_n;      //spatial oscillations frequency                                       Read from coordinates file generates by python GPR code
//float Av = speed/2;        //speed oscillations amplitude                                          Read from coordinates file generates by python GPR code
//float Fv = 2*speed/_n;        //speed oscillations frequency                                       Read from coordinates file generates by python GPR code
//PrintWriter output;
//PrintWriter output2;
//PrintWriter outputFullMap;
//float dt;
//float t = 0;
//float posX = 0;
//SaveData dat;
//SaveData dat2;
//int line = 0;
//ArrayList<float[]> pressureDataList = new ArrayList<>();
//int numTimeStep = 0;
//String name;

//void setup() {
//  parameters = loadTable("E:/simuChina/metric_test_next_param.csv", "header");
//  Re = parameters.getFloat(0,0);
//  origin = parameters.getFloat(0,1);
//  name = "Re"+(int)Re+"_h"+origin;
  
//  size(1400, 700);                             // display window size
//  int n=_n;                                   // number of grid points      n = 1m
//  float L = _L;                            // length-scale in grid units    L = 5cm, so L = n/20
//  Window view = new Window(2*n, n);

//  circle = new CircleBody(3*n/2, origin, L, view);     // define geom
//  flow = new BDIM(2*n, n, 0, circle, (float)L/Re, true);             // solve for flow using BDIM
//  flood = new FloodPlot(view);               // initialize a flood plot...
//  flood.setLegend("vorticity", -.5, .5);       //    and its legend
//  output = createWriter("E:/simuChina/"+name+"/Motion.csv");        // open output file
//  output2 = createWriter("E:/simuChina/"+name+"/AfterMotion.csv");        // open output file
//  //outputFullMap = createWriter("E:/simuChina/"+name+"/FullMap.csv");
//  dat = new SaveData("E:/simuChina/"+name+"/pressureMotion.txt", circle.coords, (int)L, 2*n, n, 1);    // initialize the output data file with header information
//  dat2 = new SaveData("E:/simuChina/"+name+"/pressureAfterMotion.txt", circle.coords, (int)L, 2*n, n, 1);    // initialize the output data file with header information
//}
//void draw() {
//  if (flow.QUICK) {
//    dt = flow.checkCFL();
//    flow.dt = dt;//modify
//  }
//  t += dt;
//  if (posX < _n) {
//    circle.follow();
//    circle.translate(-(speed + line*Av*sin(2*PI*Fv*t))*dt, line*2*PI*Fy*Ay*cos(2*PI*Fy*t)*dt);                      // translate the body in a straight line
//    flow.update(circle);
//    flow.update2();
//    flood.display(flow.u.curl());              // compute and display vorticity
//    circle.display();                            // display the body

//    PVector forces = circle.pressForce(flow.p);  // pressure force
//    float thrust = 2.*forces.x/_L;                     // thrust coefficient                             Cd = 1.24 for a fix cylinder in a steady flow, after the transitional regime (in theory, it's around 1.17)      [values are given for a relative speed of 1]
//    //                                                                                                   Cd = 0.78 for the transitional regime, corresponding to the experiment held in China
//    //                                                                                                   Cd = -0.14 for the transitional regime corresponding to the stopping motion, which is what happens just after the experiment
//    float lift = 2.*forces.y/_L;
//    output.println(""+t+","+thrust+","+lift);           // print to file
//    dat.addData(t, flow.p);    // add the pressure arounf the foil to the data file
//    for (int i=0; i<21; i++) {
//      dat.output.print(flow.p.extract(_n+i*_L, 1)+" ");
//    }
//    dat.output.println("");

//    //// Store pressure data for every point in the window
//    //float[] pressureData = new float[2*_n * _n];
//    //int index = 0;
//    //for (int i = 0; i < 2*_n; i++) {
//    //  for (int j = 0; j < _n; j++) {
//    //    pressureData[index] = flow.p.extract(i, j);
//    //    index++;
//    //  }
//    //}
//    //// Add the pressure data array to the list for this time step
//    //pressureDataList.add(pressureData);
//    numTimeStep++;
//  } else if (posX < 2*_n) {
//    circle.follow();
//    circle.translate(0, 0);
//    flow.update(circle);
//    flow.update2();
//    flood.display(flow.u.curl());              // compute and display vorticity
//    circle.display();                            // display the body

//    PVector forces = circle.pressForce(flow.p);  // pressure force
//    float thrust = 2.*forces.x/_L;                     // thrust coefficient
//    float lift = 2.*forces.y/_L;
//    dat2.addData(t, flow.p);    // add the pressure arounf the foil to the data file
//    for (int i=0; i<21; i++) {
//      dat2.output.print(flow.p.extract(_n+i*_L, 1)+" ");
//    }
//    dat2.output.println("");
//    output2.println(""+t+","+thrust+","+lift);           // print to file
//  } else {  // close and save everything when t>Time
//    output.close();
//    output2.close();
//    dat.finish();
//    dat2.finish();
//    //dataAdd();
//    exit();
//  }
//  fill(0);
//  text("posX = " + min(posX,128), width/2, height-30);
//  text("posY = " + Ay*sin(2*PI*Fy*t), width/2, height-60);
//  posX += (speed + line*Av*sin(2*PI*Fv*t))*dt;
//  saveFrame("saved/frame-####.png");
//}

//void dataAdd() {
//  // Write the pressure data to the CSV file as a single column
//  for (int i = 0; i < _n * 2*_n; i++) {
//    for (int tStep = 0; tStep < numTimeStep; tStep++) {
//      println(_n*2*_n, numTimeStep, i, tStep);
//      float[] pressure = pressureDataList.get(tStep);
//      outputFullMap.print(pressure[i]);
//      if (tStep < numTimeStep - 1) {
//        outputFullMap.print(","); // Separate values with newlines
//      }
//    }
//    outputFullMap.println(); // Move to the next row
//  }
//  outputFullMap.close();
//}

//// -------------------------------------------------------------------------------------------------------------
//// HAACHAMA pipeline for automatic generation of digital twin from csv files resulting from YOLOv8 segmentation
//// -------------------------------------------------------------------------------------------------------------
//BDIM flow;
//CSV2DigitalTwin body;
//TestLine upWall;
//TestLine bottomWall;
//BodyUnion twin;
//BodyUnion wall;
//FloodPlot flood;
//SaveData dat;
//PrintWriter output;
//float time=0;
//float[] a={0,.2,-.1};

//Table xTable; // Variable to store x-coordinate data
//Table yTable; // Variable to store y-coordinate data

//ArrayList<float[]> pressureDataList = new ArrayList<>();
//int numTimeStep = 0;
//int numRows = (int)pow(2,7);                                   // Number of rows
//int numCols = (int)pow(2,8);                                 // Number of columns

//void setup(){
//  int n=(int)pow(2,8);
//  int m=(int)pow(2,7);
//  size(1000,600);
//  Window view = new Window(n,m);

//  // Load x-coordinate data from CSV file
//  xTable = loadTable("C:/Users/blagn771/Documents/Aquaman/Aquaman/x.csv", "header");
//  // Load y-coordinate data from CSV file
//  yTable = loadTable("C:/Users/blagn771/Documents/Aquaman/Aquaman/y.csv", "header");

//  upWall = new TestLine(0, 36.5, 256, view);                 // 37 because 39 grid is the gap between the top and the begining of the tank, but it is 2 grid thick
//  body = new CSV2DigitalTwin(xTable.getFloat(0,0), yTable.getFloat(0,0), xTable.getRowCount(), "C:/Users/blagn771/Documents/Aquaman/Aquaman/x.csv","C:/Users/blagn771/Documents/Aquaman/Aquaman/y.csv","C:/Users/blagn771/Documents/Aquaman/Aquaman/y_dot.csv",view);
//  bottomWall = new TestLine(0, 127-38.5, 256, view);
//  wall = new BodyUnion(upWall, bottomWall);
//  twin = new BodyUnion(body, wall);
//  flow = new BDIM(n,m,0.69,twin,0.00095,true, 0);
//  //
//  //Don't forget to adapt line 171 in CSV2DigitalTwin accordingly
//  //
//  flood = new FloodPlot(view);
//  flood.range = new Scale(-.5,.5);
//  flood.setLegend("vorticity");
//  flood.setColorMode(1);


//  dat = new SaveData("saved/pressure.txt", body.coords, 0,n,n,1);
//  output = createWriter("testDataSave/pressure_map_test.csv"); // open output file
//}

//void draw(){
//  if ((int)(flow.t / 0.69 -1) < 236){
//    time += flow.dt;
//    body.update(time);
//    flow.update(twin); flow.update2();         // 2-step fluid update
//    flood.display(flow.u.curl());              // compute and display vorticity
//    body.display();                            // display the body
//    upWall.display();
//    bottomWall.display();

//    // Save the x and y values for every points of the body at each time step
//    // This is used to create the labels for YOLO quick training
//    int size = body.coords.size();
//    //dat.output.print(0 + " ");
//    for(int i=0; i<size; i++){
//      dat.output.print("Point numero " + i + " : ");
//      dat.output.print("x: ");
//      dat.output.print(body.coords.get(i).x +" ");
//      dat.output.print("y: ");
//      dat.output.print(body.coords.get(i).y +" ");
//      dat.output.print("p: ");
//      dat.output.print(flow.p.extract(body.coords.get(i).x, body.coords.get(i).y)+" ");
//    }
//    dat.output.println("");

//    // Store pressure data for every point in the window
//    float[] pressureData = new float[numCols * numRows];
//    int index = 0;
//    for (int i = 0; i < numCols; i++) {
//      for (int j = 0; j < numRows; j++) {
//        pressureData[index] = flow.p.extract(i, j);
//        index++;
//      }
//    }
//    // Add the pressure data array to the list for this time step
//    pressureDataList.add(pressureData);
//    numTimeStep++;

//    saveFrame("saved/frame-####.png");

//  } else if ((int)(flow.t / 0.69 -1) < 6000) {
//    time += flow.dt;
//    body.translate(-0.1,0);
//    flow.update(twin); flow.update2();         // 2-step fluid update
//    flood.display(flow.u.curl());              // compute and display vorticity
//    body.display();                            // display the body
//    upWall.display();
//    bottomWall.display();
//    saveFrame("saved/frame-####.png");
//  } else {
//    //dat.finish();
//    dataAdd();
//    exit();
//  }
//}

//void dataAdd() {
//  // Write the pressure data to the CSV file as a single column
//  for (int i = 0; i < numRows * numCols; i++) {
//    for (int tStep = 0; tStep < numTimeStep; tStep++) {
//      float[] pressure = pressureDataList.get(tStep);
//      output.print(pressure[i]);
//      if (tStep < numTimeStep - 1) {
//        output.print(","); // Separate values with newlines
//      }
//    }
//    output.println(); // Move to the next row
//  }
//  output.close();
//}

////// Plot the average field, which need to be computed beforehand in python
////Table mean_field;

////void setup(){
////  size(640,640);
////  int n=(int)pow(2,6);
////  int index = 0;
////  mean_field = loadTable("C:/Users/blagn771/Documents/Aquaman/Aquaman/mean_customTest2_field.csv", "header");
////  FloodPlot plot = new FloodPlot(new Window(n,n)); // standard window
////  Field p = new Field(n,n);
////  for( int i=0; i<n; i++){
////  for( int j=0; j<n; j++){
////    if (index < n*n-1){
////    p.a[i][j] = mean_field.getFloat(index,0);
////    index += 1;
////    }
////  }}
////  plot.range = new Scale(-0.5,0.5);
////  plot.setLegend("Mean field");
////  plot.display(p);
////  saveFrame("saved/mean_customTestMid_field.png");
////}
