//// //----------------------------------------------------------------------------------------------------------------------------------------------
//// //Sandbox for INA: base flow estimation
//// //----------------------------------------------------------------------------------------------------------------------------------------------

////BDIM flow;
////Body circle;
////FloodPlot flood;

////Table parameters;

////int _n=(int)pow(2, 7);
////float _L=_n/20;
////float Re = 1000;     // Reynolds number                                                            Read from coordinates file generates by python GPR code
////PrintWriter outputFullMap;
////float dt;
////float t = 0;
////int line = 0;
////ArrayList<float[]> pressureDataList = new ArrayList<>();
////int numTimeStep = 0;
////boolean steadyStateAchieved = false;  // Flag to check if steady state is reached

////void setup() {
////  size(1400, 700);                             // display window size
////  int n=_n;                                   // number of grid points      n = 1m
////  float L = _L;                            // length-scale in grid units    L = 5cm, so L = n/20
////  Window view = new Window(2*n, n);

////  circle = new CircleBody(4*n/7, L, L, view);     // define geom
////  flow = new BDIM(2*n, n, 0, circle, (float)L/Re, true, 1);             // solve for flow using BDIM
////  flood = new FloodPlot(view);               // initialize a flood plot...
////  flood.setLegend("vorticity", -.5, .5);       //    and its legend
  
////  outputFullMap = createWriter("E:/benchmark_SINDy/FullVorticityMapRe1000_h6_baseFlow.csv");

////  // Call to the new steady state solver, tolerance and max iterations need to be tuned based on the case
////  flow.solveSteadyState(0.0001, 1000);           // Tolerance for steady state and max iterations
////  steadyStateAchieved = true;
////}

////void draw(){
////  if (!steadyStateAchieved) {
////    println("Steady state not achieved, check console for details.");
////    noLoop();  // Stop the drawing loop if steady state is not achieved
////  } else {
////    circle.follow();                           // Update the body
////    flood.display(flow.u.curl());            // Compute and display vorticity
////    circle.display();                          // Display the body
////    saveFrame("saved/baseFlowRe_"+Re+"h_"+6+".png");
////    // Store pressure data for every point in the window
////    float[] pressureData = new float[2*_n * _n];
////    int index = 0;
////    for (int i = 0; i < 2*_n; i++) {
////      for (int j = 0; j < _n; j++) {
////        pressureData[index] = flow.u.curl().extract(i, j);
////        index++;
////      }
////    }
////    // Add the pressure data array to the list for this time step
////    pressureDataList.add(pressureData);
////    numTimeStep++;
////    dataAdd();
////  }
////}

////void dataAdd() {
////  // Write the pressure data to the CSV file as a single column
////  for (int i = 0; i < _n * 2*_n; i++) {
////    for (int tStep = 0; tStep < numTimeStep; tStep++) {
////      float[] pressure = pressureDataList.get(tStep);
////      outputFullMap.print(pressure[i]);
////      if (tStep < numTimeStep - 1) {
////        outputFullMap.print(","); // Separate values with newlines
////      }
////    }
////    outputFullMap.println(); // Move to the next row
////  }
////  outputFullMap.close();
////}


////// ----------------------------------------------------------------------------------------------------------------------------------------------
////// Sandbox for INA
////// ----------------------------------------------------------------------------------------------------------------------------------------------
////BDIM flow;
////Body circle;
////FloodPlot flood;
////SaveData dat;

////Table parameters;

////int _n=(int)pow(2, 7);
////float _L=_n/20;
////float Re = 9000;     // Reynolds number                                                            Read from coordinates file generates by python GPR code
////float origin = 2.33*_L;
////PrintWriter outputFullMap;
////float dt;
////float t = 0;
////int line = 0;
////ArrayList<float[]> pressureDataList = new ArrayList<>();
////int numTimeStep = 0;
////String name;

////void setup() {
////  size(1400, 700);                             // display window size
////  int n=_n;                                   // number of grid points      n = 1m
////  float L = _L;                            // length-scale in grid units    L = 5cm, so L = n/20
////  Window view = new Window(2*n, n);
////  name = "Re"+(int)Re+"_h"+origin;

////  circle = new CircleBody(4*n/7, origin, L, view);     // define geom
////  flow = new BDIM(2*n, n, 0, circle, (float)L/Re, true, 1);             // solve for flow using BDIM
////  flood = new FloodPlot(view);               // initialize a flood plot...
////  flood.setLegend("vorticity", -.5, .5);       //    and its legend
  
////  outputFullMap = createWriter("E:/simuChina/"+name+"/FullMap.csv");
////  dat = new SaveData("E:/simuChina/"+name+"/Time.txt", circle.coords, (int)L, 2*n, n, 1);    // initialize the output data file with header information
////}
////void draw() {
////  if (flow.QUICK) {
////    dt = flow.checkCFL();
////    flow.dt = dt;//modify
////  }
////  t += dt;
////  dat.addData(t, flow.p);    // add the pressure arounf the foil to the data file
////  circle.follow();
////  flow.update(circle);
////  flow.update2();
////  flood.display(flow.u.curl());              // compute and display vorticity
////  circle.display();                            // display the body


////  // Store pressure data for every point in the window
////  float[] pressureData = new float[2*_n * _n];
////  int index = 0;
////  for (int i = 0; i < 2*_n; i++) {
////    for (int j = 0; j < _n; j++) {
////      pressureData[index] = flow.u.curl().extract(i, j);
////      index++;
////    }
////  }
////  // Add the pressure data array to the list for this time step
////  pressureDataList.add(pressureData);
////  numTimeStep++;
  
////  if (t > 500) {  // close and save everything when t>Time
////    dataAdd();
////    dat.finish();
////    println("Finsihed");
////    exit();
////  }
////}

////void dataAdd() {
////  // Write the pressure data to the CSV file as a single column
////  for (int i = 0; i < _n * 2*_n; i++) {
////    for (int tStep = 0; tStep < numTimeStep; tStep++) {
////      float[] pressure = pressureDataList.get(tStep);
////      outputFullMap.print(pressure[i]);
////      if (tStep < numTimeStep - 1) {
////        outputFullMap.print(","); // Separate values with newlines
////      }
////    }
////    outputFullMap.println(); // Move to the next row
////  }
////  outputFullMap.close();
////}

//// //----------------------------------------------------------------------------------------------------------------------------------------------
//// //Digital twin of the experiments held in China to generate a dataset for future data fusion between physical sensors and the result of HAACHAMA
//// //----------------------------------------------------------------------------------------------------------------------------------------------
////BDIM flow;
////Body circle;
////FloodPlot flood;

////Table parameters;

////int _n=(int)pow(2, 7);
////float _L=_n/20;
////float speed = 1;    //grid per time step
////float Re;// = 100;     // Reynolds number                                                            Read from coordinates file generates by python GPR code
////float origin;// = _n/2;      //mean position                                                            Read from coordinates file generates by python GPR code
////float Ay = _L;      //spatial oscillations amplitude                                               Read from coordinates file generates by python GPR code
////float Fy = 3*speed/_n;      //spatial oscillations frequency                                       Read from coordinates file generates by python GPR code
////float Av = speed/2;        //speed oscillations amplitude                                          Read from coordinates file generates by python GPR code
////float Fv = 2*speed/_n;        //speed oscillations frequency                                       Read from coordinates file generates by python GPR code
////PrintWriter output;
////PrintWriter output2;
////PrintWriter outputFullMap;
////float dt;
////float t = 0;
////float posX = 0;
////SaveData dat;
////SaveData dat2;
////int line = 0;
////ArrayList<float[]> pressureDataList = new ArrayList<>();
////int numTimeStep = 0;
////String name;

////void setup() {
////  parameters = loadTable("E:/simuChina/metric_test_next_param.csv", "header");
////  Re = parameters.getFloat(0,0);
////  origin = parameters.getFloat(0,1);
////  name = "Re"+(int)Re+"_h"+origin;
  
////  size(1400, 700);                             // display window size
////  int n=_n;                                   // number of grid points      n = 1m
////  float L = _L;                            // length-scale in grid units    L = 5cm, so L = n/20
////  Window view = new Window(2*n, n);

////  circle = new CircleBody(3*n/2, origin, L, view);     // define geom
////  flow = new BDIM(2*n, n, 0, circle, (float)L/Re, true);             // solve for flow using BDIM
////  flood = new FloodPlot(view);               // initialize a flood plot...
////  flood.setLegend("vorticity", -.5, .5);       //    and its legend
////  output = createWriter("E:/simuChina/"+name+"/Motion.csv");        // open output file
////  output2 = createWriter("E:/simuChina/"+name+"/AfterMotion.csv");        // open output file
////  outputFullMap = createWriter("E:/simuChina/"+name+"/FullMap.csv");
////  dat = new SaveData("E:/simuChina/"+name+"/pressureMotion.txt", circle.coords, (int)L, 2*n, n, 1);    // initialize the output data file with header information
////  dat2 = new SaveData("E:/simuChina/"+name+"/pressureAfterMotion.txt", circle.coords, (int)L, 2*n, n, 1);    // initialize the output data file with header information
////  println("Attention, ici la vorticity est extraite et non la pression");
////}
////void draw() {
////  if (flow.QUICK) {
////    dt = flow.checkCFL();
////    flow.dt = dt;//modify
////  }
////  t += dt;
////  if (posX < _n) {
////    circle.follow();
////    circle.translate(-(speed + line*Av*sin(2*PI*Fv*t))*dt, line*2*PI*Fy*Ay*cos(2*PI*Fy*t)*dt);                      // translate the body in a straight line
////    flow.update(circle);
////    flow.update2();
////    flood.display(flow.u.curl());              // compute and display vorticity
////    circle.display();                            // display the body

////    PVector forces = circle.pressForce(flow.p);  // pressure force
////    float thrust = 2.*forces.x/_L;                     // thrust coefficient                             Cd = 1.24 for a fix cylinder in a steady flow, after the transitional regime (in theory, it's around 1.17)      [values are given for a relative speed of 1]
////    //                                                                                                   Cd = 0.78 for the transitional regime, corresponding to the experiment held in China
////    //                                                                                                   Cd = -0.14 for the transitional regime corresponding to the stopping motion, which is what happens just after the experiment
////    float lift = 2.*forces.y/_L;
////    output.println(""+t+","+thrust+","+lift);           // print to file
////    dat.addData(t, flow.p);    // add the pressure arounf the foil to the data file
////    for (int i=0; i<21; i++) {
////      dat.output.print(flow.p.extract(_n+i*_L, 1)+" ");
////    }
////    dat.output.println("");

////    // Store pressure data for every point in the window
////    float[] pressureData = new float[2*_n * _n];
////    int index = 0;
////    for (int i = 0; i < 2*_n; i++) {
////      for (int j = 0; j < _n; j++) {
////        pressureData[index] = flow.u.curl().extract(i, j);
////        index++;
////      }
////    }
////    // Add the pressure data array to the list for this time step
////    pressureDataList.add(pressureData);
////    numTimeStep++;
////  } else if (posX < 2*_n) {
////    circle.follow();
////    circle.translate(0, 0);
////    flow.update(circle);
////    flow.update2();
////    flood.display(flow.u.curl());              // compute and display vorticity
////    circle.display();                            // display the body

////    PVector forces = circle.pressForce(flow.p);  // pressure force
////    float thrust = 2.*forces.x/_L;                     // thrust coefficient
////    float lift = 2.*forces.y/_L;
////    dat2.addData(t, flow.p);    // add the pressure arounf the foil to the data file
////    for (int i=0; i<21; i++) {
////      dat2.output.print(flow.p.extract(_n+i*_L, 1)+" ");
////    }
////    dat2.output.println("");
////    output2.println(""+t+","+thrust+","+lift);           // print to file
////  } else {  // close and save everything when t>Time
////    output.close();
////    output2.close();
////    dat.finish();
////    dat2.finish();
////    dataAdd();
////    println("Finished saving");
////    exit();
////  }
////  fill(0);
////  text("posX = " + min(posX,128), width/2, height-30);
////  text("posY = " + Ay*sin(2*PI*Fy*t), width/2, height-60);
////  posX += (speed + line*Av*sin(2*PI*Fv*t))*dt;
////}

////void dataAdd() {
////  // Write the pressure data to the CSV file as a single column
////  for (int i = 0; i < _n * 2*_n; i++) {
////    for (int tStep = 0; tStep < numTimeStep; tStep++) {
////      float[] pressure = pressureDataList.get(tStep);
////      outputFullMap.print(pressure[i]);
////      if (tStep < numTimeStep - 1) {
////        outputFullMap.print(","); // Separate values with newlines
////      }
////    }
////    outputFullMap.println(); // Move to the next row
////  }
////  outputFullMap.close();
////}

// -------------------------------------------------------------------------------------------------------------
// HAACHAMA pipeline for automatic generation of digital twin from csv files resulting from YOLOv8 segmentation
// -------------------------------------------------------------------------------------------------------------
BDIM flow;
TestLine upWall;
TestLine bottomWall;
BodyUnion twin;
BodyUnion wall;
FloodPlot flood;
SaveData dat;
PrintWriter output;
float time=0;
float[] a={0,.2,-.1};
float pas = 0.686778;
float mu = 0.00095;

int startIndex;
int Uturn;
int haato;
int haachama;

Boolean fish;

Boolean pressure = false;
Boolean automated = true;

CSV2DigitalTwin body;
CSV2CircleTwin circle_body;

String xFile;
String yFile;
String yDotFile;

String spine;

//String xFile = "C:/Users/blagn771/Documents/Aquaman/Aquaman/x.csv";
//String yFile = "C:/Users/blagn771/Documents/Aquaman/Aquaman/y.csv";
//String yDotFile = "C:/Users/blagn771/Documents/Aquaman/Aquaman/y_dot.csv";

Table xTable; // Variable to store x-coordinate data
Table yTable; // Variable to store y-coordinate data
Table spineTable; // Variable to store xy1 xy2 xy3 ... for the spine coordinates

ArrayList<float[]> pressureDataList = new ArrayList<>();
int numTimeStep = 0;
int numRows = (int)pow(2,7);                                   // Number of rows
int numCols = (int)pow(2,8);                                 // Number of columns


void setup(){
  
  // Debug: print all args
  if (args != null) {
    println("Args received:");
    for (int i = 0; i < args.length; i++) {
      println("args[" + i + "] = " + args[i]);
    }
  }
  // Important: skip args[0] == "--args" if you're using processing-java
  int offset = (args != null && args.length >= 6 && args[0].equals("--args")) ? 1 : 0;
  
  if (automated) {
    try {
      if (args != null && args.length >= offset + 5) {
        Uturn = int(args[offset]);
        startIndex = int(args[offset + 1]);
        haato = int(args[offset + 2]);
        haachama = int(args[offset + 3]);
        fish = args[offset + 4].equalsIgnoreCase("true");
      }
    } catch (Exception e) {
      println("Error parsing command-line arguments: " + e.getMessage());
    }
  }
  
  xFile = "D:/crop_nadia/"+str(haato)+"/"+str(haachama)+"/rawYolo"+str(haachama)+"_x.csv";
  yFile = "D:/crop_nadia/"+str(haato)+"/"+str(haachama)+"/rawYolo"+str(haachama)+"_y.csv";
  yDotFile = "D:/crop_nadia/"+str(haato)+"/"+str(haachama)+"/rawYolo"+str(haachama)+"_y_dot.csv";
  
  spine = "D:/thomas_files/"+str(haato)+"/"+str(haachama)+"/final/spines_interpolated.csv";

  int n=(int)pow(2,8);
  int m=(int)pow(2,7);
  size(1000,600);
  Window view = new Window(n,m);

  // Load x-coordinate data from CSV file
  xTable = loadTable(xFile, "header");
  // Load y-coordinate data from CSV file
  yTable = loadTable(yFile, "header");
  // Load x and y coordinates data from CSV file
  spineTable = loadTable(spine, "header");

  upWall = new TestLine(0, 36.5, 256, view);                 // 37 because 39 grid is the gap between the top and the begining of the tank, but it is 2 grid thick
  if (fish) {
    body = new CSV2DigitalTwin(xTable.getFloat(0,0), yTable.getFloat(0,0), xTable.getRowCount(), xFile, yFile, yDotFile,view, startIndex);
  } else {
    circle_body = new CSV2CircleTwin(3.5, spineTable, view, startIndex);
  };
  bottomWall = new TestLine(0, 37+43.5, 256, view);
  wall = new BodyUnion(upWall, bottomWall);
  if (fish) {
    twin = new BodyUnion(body, wall);
  } else {
    twin = new BodyUnion(circle_body, wall);
  };
  flow = new BDIM(n,m,pas,twin,mu,true, 0);
  //
  //Don't forget to adapt line 171 in CSV2DigitalTwin accordingly
  //
  flood = new FloodPlot(view);
  flood.range = new Scale(-.5,.5);
  flood.setLegend("vorticity");
  flood.setColorMode(1);


  if (fish) {
    dat = new SaveData("D:/crop_nadia/test_vortex/circle_"+str(haato)+"_"+str(haachama)+"_bodyPressure.txt", body.coords, 0,n,n,1);
  };
  if (fish) {
    if (pressure) {
      output = createWriter("D:/crop_nadia/"+str(haato)+"/"+str(haachama)+"/pressure_map.csv"); // open output file
    }
    else {
      output = createWriter("D:/crop_nadia/"+str(haato)+"/"+str(haachama)+"/vorticity_map.csv"); // open output file
    }
  }
  else {
    if (pressure) {
      output = createWriter("D:/crop_nadia/"+str(haato)+"/"+str(haachama)+"/circle_pressure_map.csv"); // open output file
    }
    else {
      output = createWriter("D:/crop_nadia/"+str(haato)+"/"+str(haachama)+"/circle_vorticity_map.csv"); // open output file
    }
  }
}

void draw(){
  if ((int)(flow.t / pas -1) < loadTable(xFile, "header").getColumnCount() - 2 - startIndex){
    time += flow.dt;
    // Check if the nose changes direction (U-turn)
    //uTurn = 0 means there is a uTurn, uTurn = 1 means there is not
    if (fish && body.isUTurn() && Uturn == 0) {
        println("Nose direction changed! Replacing fish at index ", (int)(flow.t / pas -1));

        // Replace fish with a new instance
        CSV2DigitalTwin newBody = new CSV2DigitalTwin(
            body.coords.get(0).x, 
            body.coords.get(0).y, 
            body.coords.size(), 
            xFile, 
            yFile, 
            yDotFile, 
            new Window(numCols,numRows),
            (int)(flow.t / pas -1) + startIndex
        );

        // Replace the old fish instance
        body = newBody;
        
        // Rebuild the BodyUnion with the new fish
        twin = new BodyUnion(body, wall);
        time = flow.dt;
    }

    // Regular update of the fish and fluid
    if (fish) {
      body.update(time);
    } else {
      circle_body.update(time);
    };
    flow.update(twin); 
    flow.update2(); // 2-step fluid update

    // Display vorticity and objects
    flood.display(flow.u.curl());
    if (fish) {
      body.display();
    } else {
      circle_body.display();
    };
    upWall.display();
    bottomWall.display();    

    // Save the x and y values for every points of the body at each time step
    // This is used to create the labels for YOLO quick training
    if (fish) {
      int size = body.coords.size();
      //dat.output.print(0 + " ");
      for(int i=0; i<size; i++){
        dat.output.print("Point numero " + i + " : ");
        dat.output.print("x: ");
        dat.output.print(body.coords.get(i).x +" ");
        dat.output.print("y: ");
        dat.output.print(body.coords.get(i).y +" ");
        dat.output.print("p: ");
        dat.output.print(flow.p.extract(body.coords.get(i).x, body.coords.get(i).y)+" ");
      }
      dat.output.println("");
    };

    // Store pressure data for every point in the window
    float[] pressureData = new float[numCols * numRows];
    int index = 0;
    for (int i = 0; i < numCols; i++) {
      for (int j = 0; j < numRows; j++) {
        if (pressure) {
          pressureData[index] = flow.p.extract(i, j);
        }
        else {
          pressureData[index] = flow.u.curl().extract(i,j);
        }
        index++;
      }
    }
    // Add the pressure data array to the list for this time step
    pressureDataList.add(pressureData);
    numTimeStep++;

    if (automated == false) {
      saveFrame("saved/frame-####.png");
    }

  //  saveFrame("saved/frame-####.png");
  } else {
    //dat.finish();
    dataAdd();
    exit();
  }
}

void dataAdd() {
  // Write the pressure data to the CSV file as a single column
  for (int i = 0; i < numRows * numCols; i++) {
    for (int tStep = 0; tStep < numTimeStep; tStep++) {
      float[] pressure = pressureDataList.get(tStep);
      output.print(pressure[i]);
      if (tStep < numTimeStep - 1) {
        output.print(","); // Separate values with newlines
      }
    }
    output.println(); // Move to the next row
  }
  output.close();
}

//// Plot the average field, which need to be computed beforehand in python
//Table mean_field;

//void setup(){
//  size(640,640);
//  int n=(int)pow(2,6);
//  int index = 0;
//  mean_field = loadTable("C:/Users/blagn771/Documents/Aquaman/Aquaman/mean_customTest2_field.csv", "header");
//  FloodPlot plot = new FloodPlot(new Window(n,n)); // standard window
//  Field p = new Field(n,n);
//  for( int i=0; i<n; i++){
//  for( int j=0; j<n; j++){
//    if (index < n*n-1){
//    p.a[i][j] = mean_field.getFloat(index,0);
//    index += 1;
//    }
//  }}
//  plot.range = new Scale(-0.5,0.5);
//  plot.setLegend("Mean field");
//  plot.display(p);
//  saveFrame("saved/mean_customTestMid_field.png");
//}

////*********************************************************/
//// Circle that can be dragged by the mouse
//BDIM flow;
//DigitalTwin body;
//CircleBody body_c;
//CircleBody body_s;
//FloodPlot flood;
//SaveData dat;
//ArrayList<PVector> extraction;
//float pos = 0;

//void setup(){
//  size(1200,400);
//  int chord = 16;
//  int n = 12*chord;
//  int m = 4*chord;
//  int Re = 6500;
  
//  Window view = new Window(n,m);
//  body = new DigitalTwin(chord,view);
//  body_c = new CircleBody(9*chord, 1.1*chord, chord,view);
//  body_s = new CircleBody(94,0.1*chord,2,view);
//  flow = new BDIM(n,m,0.416,body_c,chord/Re,true);
//  flood = new FloodPlot(view);
//  flood.setLegend("vorticity",-0.5,0.5);
  
//  extraction = new ArrayList<PVector>(1);
//  extraction.add(new PVector(94,0.1*chord));
//  dat = new SaveData("saved/pressure.txt", extraction, chord,n,m,1);
//}
//void draw(){
//  //if (flow.t < 180){
//  //  body.update(flow.t, flow.dt);
//  //  flow.update(body); flow.update2();         // 2-step fluid update
//  //  flood.display(flow.u.curl());              // compute and display vorticity
//  //  body.display();                            // display the body
    
//  //  dat.addData(flow.t, flow.p);
//  //  saveFrame("saved/frame-####.png");
    
//  //  text("sin(omega) = " + 25*sin(2*PI*0.61*flow.t/2/0.466/body.chord),width/4,height-30);
//  //  text("t : " + flow.t, width/4,height-60);
//  //} else {
//  //  dat.finish();
//  //  exit(); 
//  //}
//  if (pos < 0.3*16/0.05) {
//    flow.update(body_c); flow.update2();
//    flood.display(flow.p);
//    body_c.display();
//    body_s.display();
//    body_c.translate(-0.416,0);
//    pos += 0.416;
//    dat.addData(flow.t, flow.p);
//    saveFrame("saved/frame-####.png");
//  }
//  else if (pos < 2*0.3*16/0.05) {
//    flow.update(body_c); flow.update2();
//    flood.display(flow.u.curl());
//    body_c.display();
//    pos += 0.416;
//    dat.addData(flow.t, flow.p);
//  }
//  else {
//    dat.finish();
//    exit(); 
//  }
//}
