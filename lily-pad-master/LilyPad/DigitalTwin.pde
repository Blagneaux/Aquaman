class DigitalTwin extends BodyUnion{
    // variable definition
    float chord;
    float len;
    float f;

    DigitalTwin(int chord, Window window) {
        super(5*chord,4*chord,window);
        float len = 74.2;

        // head of the foil
        add(new Rectangle((15.5-1-51/len)*chord,2*chord,chord*5/len,chord*13/len,window));

        // tail of the foil
        add(new Triangle((15.5-1-51/len+12.7/len)*chord,2*chord,chord*12.71/len,chord*62.7/len,chord,window));

        // walls
        add(new TestLine(0,(2+100/len)*chord-2,16*chord,window));
        add(new Sensor(0,(2-100/len+34/len)*chord,16*chord,window));
        add(new TestLine(0,(2-100/len)*chord,16*chord,window));
        add(new VWall(15.5*chord,(2+100/len)*chord,chord,window));
        add(new VWall((15.5-605/len)*chord,(2+100/len)*chord,chord,window));
        add(new VWall((15.5-505/len)*chord,(2+100/len)*chord,chord,window));

        // tests
        // add(new TestLine(14*chord+6.5*chord/len,4.4*chord, chord, window)); //length
        // add(new TestLine(0,(1.5+tan(25*PI/180))*chord-2,16*chord, window));
        // add(new TestLine(0,(1.5-tan(25*PI/180))*chord,16*chord, window));

        this.len = len;
        this.chord = chord;
        this.f = 0.61/1.375/0.466/chord;
    }

    // Rotate the tail
    void update(float t, float dt) {
        if (t < 4/f) {
            float omega = (25*PI/180)*sin(2*PI*f*t);
            float omega_0 = (25*PI/180)*sin(2*PI*f*(t-dt));
            if (t == 0.) {
                this.bodyList.get(1).rotate(omega);
            }
            else {
            this.bodyList.get(1).rotate(omega-omega_0); 
            }
            this.bodyList.get(0).translate(-89*dt/len,0);
            this.bodyList.get(1).translate(-89*dt/len,0);
        }else{

        }
    }
}