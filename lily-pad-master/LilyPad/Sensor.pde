class Sensor extends Body {
    float chord;

    Sensor(float x, float y, int chord, Window window) {
        super(x,y,window);
        float res = chord/16;
        float len = 74.2;

        add(xc.x+chord-241*res/len,xc.y);

        // //sensor 5 first side
        // add(xc.x+chord-508*res/len,xc.y);

        // //sensor 4 first side
        // add(xc.x+chord-413*res/len,xc.y);

        // //sensor 3 first side
        // add(xc.x+chord-314*res/len,xc.y);

        // //sensor 2 first side
        // add(xc.x+chord-205*res/len,xc.y);

        //sensor 1 first side
        add(xc.x+chord-242*res/len,xc.y);

        // add(xc.x+chord,xc.y);
        // add(xc.x+chord,xc.y+2);

        //sensor 1 scd side
        add(xc.x+chord-242*res/len,xc.y+56*res/len);

        // //sensor 2 scd side
        // add(xc.x+chord-205*res/len,xc.y+2);

        // //sensor 3 scd side
        // add(xc.x+chord-314*res/len,xc.y+2);

        // //sensor 4 scd side
        // add(xc.x+chord-413*res/len,xc.y+2);

        // //sensor 5 scd side
        // add(xc.x+chord-508*res/len,xc.y+2);
        
        add(xc.x+chord-241*res/len,xc.y+56*res/len);
        end();
    }
}