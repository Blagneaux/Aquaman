class Triangle extends Body {
    int m = 100;

    Triangle( float x, float y, float b, float h, float chord, Window window) {
        super(x, y, window);
        float dx = 0.5*b, dy = 0.5*b;

        // for ( int i=0; i<m; i++ ) {
        //     float theta = -TWO_PI*i/((float)m);
        //     add(xc.x+dx/4*cos(theta), xc.y+dy/4*sin(theta));
        // }

        for ( int i=0; i<m; i++ ) {
            float theta = -TWO_PI*i/((float)m)/4.5;
            add(xc.x+1.5*dx*cos(theta-PI+PI/4.5)+2*dx-7.7*chord/74.2, xc.y+1.5*dy*sin(theta-PI+PI/4.5));
        }

        // add(xc.x+h-1-7.7*chord/74.2, xc.y+0.2);
        add(xc.x+h-7.7*chord/74.2, xc.y);
        // add(xc.x+h-1-7.7*chord/74.2, xc.y-0.2);

        end();
    
        // ma = new PVector(PI*sq(dy),PI*sq(dx),0.125*PI*sq(sq(dx)-sq(dy)));
    }
}