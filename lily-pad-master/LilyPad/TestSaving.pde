// /*********************************************************
//                   Main Window!

// Click the "Run" button to Run the simulation.

// Change the geometry, flow conditions, numerical parameters
// visualizations and measurements from this window.

// This screen has an example. Other examples are found at 
// the top of each tab. Copy/paste them here to run, but you 
// can only have one setup & run at a time.

// *********************************************************/
// // Circle that can be dragged by the mouse
// BDIM flow;
// Body body;
// FloodPlot flood;
// PrintWriter output;

// int n=(int)pow(2,7);                       // number of grid points
// ArrayList<float[]> pressureDataList = new ArrayList<>();
// int numTimeStep = 0;
// int numRows = n;                                   // Number of rows
// int numCols = n;                                   // Number of columns
// float L = n/8.;                            // length-scale in grid units
// float t = 0;                                // time


// void setup(){
//   size(700,700);                             // display window size
//   Window view = new Window(n,n);
//   body = new CircleBody(n/3,n/2,L,view);     // define geom
//   flow = new BDIM(n,n,1.5,body);             // solve for flow using BDIM
//   flood = new FloodPlot(view);               // initialize a flood plot...
//   flood.setLegend("vorticity",-.5,.5);       //    and its legend
//   output = createWriter("testDataSave/pressure_data.csv"); // open output file
// }
// void draw(){
//   t += flow.dt;                              // update the time
//   print("flow.dt", flow.dt);
//   body.follow();                             // update the body
//   flow.update(body); flow.update2();         // 2-step fluid update
//   flood.display(flow.u.curl());              // compute and display vorticity
//   body.display();                            // display the body
  
//   // Store pressure data for every point in the window
//   float[] pressureData = new float[numCols * numRows];
//   int index = 0;
//   for (int i = 0; i < numCols; i++) {
//     for (int j = 0; j < numRows; j++) {
//       pressureData[index] = flow.p.linear(i, j);
//       index++;
//     }
//   }
  
//   // Add the pressure data array to the list for this time step
//   pressureDataList.add(pressureData);
//   numTimeStep++;
  
//   if (t >= 100) {     // finish after 4 cycles
//     dataAdd();
//     exit();
//   }
// }

// void dataAdd() {
//   // Write the pressure data to the CSV file as a single column
//   print("pressure", pressureDataList);
//   for (int i = 0; i < numRows * numCols; i++) {
//     for (int tStep = 0; tStep < numTimeStep; tStep++) {
//       float[] pressure = pressureDataList.get(tStep);
//       output.print(pressure[i]);
//       if (tStep < numTimeStep - 1) {
//         output.print(","); // Separate values with newlines
//       }
//     }
//     output.println(); // Move to the next row
//   }
//   output.close();
// }