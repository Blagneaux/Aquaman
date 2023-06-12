class Rectangle extends Body{
  int m = 100;

  Rectangle(float x, float y, float l, float w, Window window){
    super(x, y, window);
    float dx = 0.5*w, dy = 0.5*w;

    for ( int i=0; i<m; i++ ) {
      float theta = -TWO_PI*i/((float)m)/2;
      add(xc.x+dx*cos(theta-PI/2), xc.y+dy*sin(theta-PI/2));
    }

    add(xc.x+l, xc.y+w/2);
    for ( int i=m; i>0; i-- ) {
      float theta = -TWO_PI*i/((float)m)/4.5;
      add(xc.x+1.5*dx*cos(theta-PI+PI/4.5)+2*dx, xc.y+1.5*dy*sin(theta-PI+PI/4.5));
    }
    add(xc.x+l, xc.y-w/2);
    

    end();
    
    // ma = new PVector(PI*sq(dy),PI*sq(dx),0.125*PI*sq(sq(dx)-sq(dy)));
  }
}