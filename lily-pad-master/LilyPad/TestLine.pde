class TestLine extends Body {
    float chord;

    TestLine(float x, float y, int chord, Window window) {
        super(x,y,window);

        add(xc.x,xc.y);
        add(xc.x+chord,xc.y);
        add(xc.x+chord,xc.y+2);
        add(xc.x,xc.y+2);
        end();
    }
}