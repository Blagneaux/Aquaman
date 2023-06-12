/*********************************************************
                  Main Window!

Click the "Run" button to Run the simulation.

Change the geometry, flow conditions, numerical parameters
visualizations and measurements from this window.

This screen has an example. Other examples are found at 
the top of each tab. Copy/paste them here to run, but you 
can only have one setup & run at a time.

*********************************************************/
// Circle that can be dragged by the mouse
BDIM flow;
DigitalTwin body;
FloodPlot flood;
SaveData dat;

void setup(){
  size(1600,400);
  int chord = 16;
  int n = 16*chord;
  int m = 4*chord;
  int Re = 6604;
  
  Window view = new Window(n,m);
  body = new DigitalTwin(chord,view);
  flow = new BDIM(n,m,0.25,body,chord/Re,true);
  flood = new FloodPlot(view);
  flood.setLegend("vorticity",-3,3);
  
  dat = new SaveData("saved/pressure.txt", body.bodyList.get(3).coords, chord,n,m,1);
}
void draw(){
  if (flow.t < 180){
    body.update(flow.t, flow.dt);
    flow.update(body); flow.update2();         // 2-step fluid update
    flood.display(flow.u.curl());              // compute and display vorticity
    body.display();                            // display the body
    
    dat.addData(flow.t, flow.p);
    saveFrame("saved/frame-####.png");
    
    text("sin(omega) = " + 25*sin(2*PI*0.61*flow.t/2/0.466/body.chord),width/4,height-30);
    text("t : " + flow.t, width/4,height-60);
  } else {
    dat.finish();
    exit(); 
  }
}


void mousePressed(){body.mousePressed();}    // user mouse...
void mouseReleased(){body.mouseReleased();}  // interaction methods
void mouseWheel(MouseEvent event){body.mouseWheel(event);}
