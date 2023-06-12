class VWall extends Body {
    float chord;

    VWall(float x, float y, int chord, Window window) {
        super(x,y,window);

        add(xc.x,xc.y);
        add(xc.x+2,xc.y);
        add(xc.x+2,xc.y+chord);
        add(xc.x,xc.y+chord);
        end();
    }
}