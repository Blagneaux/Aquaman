BDIM flow;
//FlexNACA body;
CSV2DigitalTwin body;
FloodPlot flood;
SaveData dat;
PrintWriter output;
float time=0;
float[] a={0,.2,-.1};

Table xTable; // Variable to store x-coordinate data
Table yTable; // Variable to store y-coordinate data

ArrayList<float[]> pressureDataList = new ArrayList<>();
int numTimeStep = 0;
int numRows = (int)pow(2,6);                                   // Number of rows
int numCols = (int)pow(2,6);                                   // Number of columns

void setup(){
  int n=(int)pow(2,6);
  size(640,640);      
  Window view = new Window(n,n);

  // Load x-coordinate data from CSV file
  xTable = loadTable("C:/Users/blagn771/Documents/Aquaman/Aquaman/x.csv", "header");
  // Load y-coordinate data from CSV file
  yTable = loadTable("C:/Users/blagn771/Documents/Aquaman/Aquaman/y.csv", "header");
  
  //body = new FlexNACA(n/4,n/2,n/3,0.20,0.25,1.2,1.,a,view);
  body = new CSV2DigitalTwin(xTable.getFloat(0,0), yTable.getFloat(0,0), xTable.getRowCount(), "C:/Users/blagn771/Documents/Aquaman/Aquaman/x.csv","C:/Users/blagn771/Documents/Aquaman/Aquaman/y.csv","C:/Users/blagn771/Documents/Aquaman/Aquaman/y_dot.csv",view);
  flow = new BDIM(n,n,0.5,body,0.001,true);
  flood = new FloodPlot(view);
  flood.range = new Scale(-.5,.5);
  flood.setLegend("vorticity");
  flood.setColorMode(1); 
  
  
  dat = new SaveData("saved/pressure.txt", body.coords, 0,n,n,1);
  output = createWriter("testDataSave/pressure_map_customFull.csv"); // open output file
}

void draw(){
  if (flow.t < 250){
    time += flow.dt;
    body.update(time);
    flow.update(body); flow.update2();         // 2-step fluid update
    flood.display(flow.p);              // compute and display vorticity
    body.display();                            // display the body
    
    // Save the x and y values for every points of the body at each time step
    // This is used to create the labels for YOLO quick training
    int size = body.coords.size();
    dat.output.print(0 + " ");
    for(int i=0; i<size; i++){
      dat.output.print(body.coords.get(i).x +" ");
      dat.output.print(body.coords.get(i).y +" ");
    }
    dat.output.println("");
    
    // Store pressure data for every point in the window
    float[] pressureData = new float[numCols * numRows];
    int index = 0;
    for (int i = 0; i < numCols; i++) {
      for (int j = 0; j < numRows; j++) {
        pressureData[index] = flow.p.extract(i, j);
        index++;
      }
    }
    // Add the pressure data array to the list for this time step
    pressureDataList.add(pressureData);
    numTimeStep++;

    saveFrame("saved/frame-####.png");
    
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

///*********************************************************
//                  Main Window!

//Click the "Run" button to Run the simulation.

//Change the geometry, flow conditions, numerical parameters
//visualizations and measurements from this window.

//This screen has an example. Other examples are found at 
//the top of each tab. Copy/paste them here to run, but you 
//can only have one setup & run at a time.

//*********************************************************/
//// Circle that can be dragged by the mouse
//BDIM flow;
//CSV2DigitalTwin body;
//FloodPlot flood;
//SaveData dat;

//void setup(){
//  size(640,640);
//  int chord = 32;
//  int n = 16*chord;
//  int m = 4*chord;
//  int Re = 250;
//  VectorField uinit = new VectorField(n+2,m+2,0,0);
//  Window view = new Window(n,m);
  
//  body = new CSV2DigitalTwin(0, 0, "C:/Users/blagn771/Desktop/x-pos2.csv","C:/Users/blagn771/Desktop/y-pos.csv",view);
//  flow = new BDIM(n,m,0.25,body,uinit,chord/Re,true);
//  flood = new FloodPlot(view);
//  flood.setLegend("vorticity",-3,3);
  
//  dat = new SaveData("saved/pressure.txt", body.coords, chord,n,m,1);
//  int size = body.coords.size();
//  dat.output.print(0 + " ");
//  for(int i=0; i<size; i++){
//    dat.output.print(body.coords.get(i).x +" ");
//    dat.output.print(body.coords.get(i).y +" ");
//  }
//  dat.output.println("");
  
//}
//void draw(){
//  if (flow.t < 250){
//    time += flow.dt;
//    //body.update();
//    body.update(time);
//    flow.update(body); flow.update2();         // 2-step fluid update
//    flood.display(flow.u.curl());              // compute and display vorticity
//    body.display();                            // display the body
    
//    int size = body.coords.size();
//    dat.output.print(0 + " ");
//    for(int i=0; i<size; i++){
//      dat.output.print(body.coords.get(i).x +" ");
//      dat.output.print(body.coords.get(i).y +" ");
//    }
//    dat.output.println("");
//    saveFrame("saved/frame-####.png");
    
//    //text("sin(omega) = " + 25*sin(2*PI*0.61*flow.t/2/0.466/body.chord),width/4,height-30);
//    //text("t : " + flow.t, width/4,height-60);
//  } else {
//    dat.finish();
//    exit(); 
//  }
//}


//void mousePressed(){body.mousePressed();}    // user mouse...
//void mouseReleased(){body.mouseReleased();}  // interaction methods
//void mouseWheel(MouseEvent event){body.mouseWheel(event);}

/***************************************************************************************************************************
                  Main Window!

Click the "Run" button to Run the simulation.

Change the geometry, flow conditions, numerical parameters
visualizations and measurements from this window.

This screen has an example. Other examples are found at 
the top of each tab. Copy/paste them here to run, but you 
can only have one setup & run at a time.

*********************************************************/
//// Circle that can be dragged by the mouse
//BDIM flow;
//DigitalTwin body;
//FloodPlot flood;
//PrintWriter output;

//ArrayList<float[]> pressureDataList = new ArrayList<>();
//int numTimeStep = 0;
//int chord = 32;
//int numRows = 4*chord;                                   // Number of rows
//int numCols = 16*chord;                                   // Number of columns

//void setup(){
//  size(1600,400);
//  int n = 16*chord;
//  int m = 4*chord;
//  int Re = 6604;
//  VectorField uinit = new VectorField(n+2,m+2,0,0);
  
//  Window view = new Window(n,m);
//  body = new DigitalTwin(chord,view);
//  flow = new BDIM(n,m,0.25,body,uinit,chord/Re,true);
//  flood = new FloodPlot(view);
//  flood.setLegend("pressure",-2,2);
//  output = createWriter("testDataSave/pressure_data.csv"); // open output file
  
//  //flow.p.linear((15.5-242/74.2)*chord,(2-(100-56)/74.2)*chord)
//  //ArrayList<PVector> extractCoord = new ArrayList<>();
//  //extractCoord.add(new PVector( (15.5-242/74.2)*chord,(2-(100-56)/74.2)*chord ));
  
//  //dat = new SaveData("saved/pressure.txt", extractCoord, chord,n,m,1);
//}
//void draw(){
//  if (flow.t < 180*chord/16){
//    body.update(flow.t, flow.dt);
//    flow.update(body); flow.update2();         // 2-step fluid update
//    flood.display(flow.p);              // compute and display vorticity
//    body.display();                            // display the body
    
//    //dat.addData(flow.t, flow.p);
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
    
//    text("sin(omega) = " + 25*sin(2*PI*0.61*flow.t/2/0.466/body.chord),width/4,height-30);
//    text("t : " + flow.t, width/4,height-60);
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

//void mousePressed(){body.mousePressed();}    // user mouse...
//void mouseReleased(){body.mouseReleased();}  // interaction methods
//void mouseWheel(MouseEvent event){body.mouseWheel(event);}

/***************************************************************************************************************************
                  Main Window!

Click the "Run" button to Run the simulation.

Change the geometry, flow conditions, numerical parameters
visualizations and measurements from this window.

This screen has an example. Other examples are found at 
the top of each tab. Copy/paste them here to run, but you 
can only have one setup & run at a time.

*********************************************************/
//// Circle that can be dragged by the mouse
//BDIM flow;
////Body body;
//CSV2DigitalTwin body;
//FloodPlot flood;
//PrintWriter output;
//SaveData dat;

//int n=(int)pow(2,7);                       // number of grid points
//ArrayList<float[]> pressureDataList = new ArrayList<>();
//int numTimeStep = 0;
//int numRows = n;                                   // Number of rows
//int numCols = n;                                   // Number of columns
//float L = n/8.;                            // length-scale in grid units
//float t = 0;                                // time


//void setup(){
//  size(700,700);                             // display window size
//  Window view = new Window(n,n);
//  //body = new CircleBody(n/3,n/2,L,view);     // define geom
//  //body = new CSV2DigitalTwin(0, 0, "C:/Users/blagn771/Desktop/x-pos2.csv","C:/Users/blagn771/Desktop/y-pos.csv",view);
//  body = new CSV2DigitalTwin(0, 0, "C:/Users/blagn771/Documents/Aquaman/Aquaman/x.csv","C:/Users/blagn771/Documents/Aquaman/Aquaman/y.csv",view);
//  flow = new BDIM(n,n,1.5,body);             // solve for flow using BDIM
//  flood = new FloodPlot(view);               // initialize a flood plot...
//  flood.setLegend("pressure",-1,1);       //    and its legend
//  dat = new SaveData("saved/pressure.txt", body.coords, 16,n,n,1);
//  output = createWriter("testDataSave/pressure_data.csv"); // open output file
//}
//void draw(){
//  t += flow.dt;                              // update the time
//  print("flow.dt", flow.dt);
//  //body.translate(0.5,0);                             // update the body
//  body.update();
//  flow.update(body); flow.update2();         // 2-step fluid update
//  flood.display(flow.p);              // compute and display vorticity
//  body.display();                            // display the body
  
//  // Store pressure data for every point in the window
//  float[] pressureData = new float[numCols * numRows];
//  int index = 0;
//  for (int i = 0; i < numCols; i++) {
//    for (int j = 0; j < numRows; j++) {
//      pressureData[index] = flow.p.extract(i, j);
//      index++;
//    }
//  }
  
//  // Add the pressure data array to the list for this time step
//  pressureDataList.add(pressureData);
//  numTimeStep++;
  
//  saveFrame("saved/frame-####.png");
  
//  if (t >= 200) {     // finish after 4 cycles
//    dataAdd();
//    dat.finish();
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

//void mousePressed(){body.mousePressed();}    // user mouse...
//void mouseReleased(){body.mouseReleased();}  // interaction methods
//void mouseWheel(MouseEvent event){body.mouseWheel(event);}
