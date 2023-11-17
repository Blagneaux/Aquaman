class CSV2DigitalTwin extends NACA {
    ArrayList<PVector[]> positionsList; // List to store positions for each time step
    ArrayList<float[]> spaceDerivativesList; // List to store the spatial derivatives
    ArrayList<float[]> timeDerivativesList; // List to store the time derivates
    int numColumns;                     // Number of columns in the tables
    int numRows;                        // Number of rows in the tables
    Table xTable, yTable, yFilteredTable;   // CSV tables for x and y coordinates
    float startTime = 0;                // Start time
    float endTime;                      // End time
    float currentTime = 0;              // Current time for interpolation
    NACA orig;                          // Base on which to build the Digital Twin
    int index;
    float x0_dep = 0;
    float y0_dep = 0;

    // For test purpose only
    float[] a={0,.2,-.1};
    float k, omega, T, xc, time=0;
    float x_init = (int)pow(2,6)/4;
    float c = (int)pow(2,6)/3;

    CSV2DigitalTwin(float x0, float y0, int m, String xFilePath, String yFilePath, String yFilteredFilePath, Window window) {
        // Just as in flex NACA, set a regular NACA coords and save as orig
        super(x0+0.25*(int)pow(2,6)/3, y0, (int)pow(2,6)/3, 0.2, m / 2, window);
        orig = new NACA(x0+0.25*(int)pow(2,6)/3,y0,(int)pow(2,6)/3,0.2,m/2,window);

        // For test purpose only ----------------------------------------------------------------------------------
        xc = x_init - 0.25*c;
        k = TWO_PI/c;
        omega = 1.2*k;
        T = TWO_PI/omega;
        float s = 0;
        for(float ai: a) s+=ai;
        if(s==0) {s=1; a[0]=1;}
        for(int i=0; i<a.length; i++) a[i] *= 0.25*T/s;
        // --------------------------------------------------------------------------------------------------------

        // Load the coordinates
        xTable = loadTable(xFilePath);
        yTable = loadTable(yFilePath);
        yFilteredTable = loadTable(yFilteredFilePath);

        // Get the number of columns in the tables
        numColumns = xTable.getColumnCount();
        endTime = numColumns;

        // Get the number of rows in the tables
        if (xTable.getRowCount() != yTable.getRowCount()) {
            println("x and y tables are not the same size");
        }
        numRows = m;
        println("Rows: ", m, "; Columns: ", endTime);

        // Initialize the positionsList based on the number of columns
        positionsList = new ArrayList<PVector[]>(numColumns);

        // Extract x and y coordinates from the tables and create points
        for (int i = 0; i < numColumns; i++) {
            PVector[] positions = new PVector[numRows]; // Create an array for positions
            for (int j = 0; j < numRows; j++) {
                float x = xTable.getFloat(j, i); // Get x-coordinate from xTable
                float y = yTable.getFloat(j, i); // Get y-coordinate from yTable
                positions[j] = new PVector(x, y); // Create PVector and store it in positions array
            }
            positionsList.add(positions); // Add the positions array to the list
        }

        // Pre compute the spatial derivatives for every time step
        spaceDerivativesList = new ArrayList<float[]>(numColumns);
        for (int i = 0; i < numColumns; i++) {
            float[] spaceDerivatives = new float[numRows];
            for (int j = 0; j < numRows; j++) {
                if (j==0) { // Left border computation
                    spaceDerivatives[j] = spaceDerivative(j+1, numRows-1, i);
                }
                else if (j==numRows-1) { // Right border computation
                    spaceDerivatives[j] = spaceDerivative(0, j-1, i);
                }
                else {
                    spaceDerivatives[j] = spaceDerivative(j+1, j-1, i);
                }
            }
            spaceDerivativesList.add(spaceDerivatives); // Add the derivatives array to the list
        }

        // Pre compute the time derivatives for every point of the shape
        timeDerivativesList = new ArrayList<float[]>(numColumns);
        for (int i = 0; i < numColumns; i++) {
            float[] timeDerivatives = new float[numRows];
            for (int j = 0; j < numRows; j++) {
                timeDerivatives[j] = yFilteredTable.getFloat(j, i);
            }
            timeDerivativesList.add(timeDerivatives); // Add the derivatives array to the list
        }

        // Draw the first state by replacing the coordinates
        for (int k = 0; k < numRows; k++) {
            coords.get(k).x = positionsList.get(0)[k].x - positionsList.get(0)[0].x + x0_dep;
            coords.get(k).y = positionsList.get(0)[k].y - positionsList.get(0)[0].y + y0_dep;
        }
        end(true, true);
    }

    // // TODO peut etre que le probleme vient d'ici ------------------------------------------------------------------
    // float distance( float x, float y) {
    //     // Find the closest point to (x,y) to get the corresponding local time derivative
    //     PVector[] currentPosDist = positionsList.get(this.index);
    //     int pt_index = 0;
    //     float min_dis = 1e10;
    //     for (int i = 0; i < currentPosDist.length; i++){
    //         if (dist(x,y,currentPosDist[i].x,currentPosDist[i].y) < min_dis) {
    //             pt_index = i;
    //         }
    //     }
    //     println(orig.distance(x,y-1));
    //     return orig.distance(x,y)+dist(x,y,currentPosDist[pt_index].x, currentPosDist[pt_index].y);
    // }

    float distance(float x, float y) {
        // PVector[] currentPosTimeD = positionsList.get(index);
        // int pt_index = 0;
        // float min_dis = 1e10;
        // for (int i = 0; i < currentPosTimeD.length; i++){
        //     if (dist(x,y,currentPosTimeD[i].x,currentPosTimeD[i].y) < min_dis) {
        //         pt_index = i;
        //         min_dis = dist(x,y,currentPosTimeD[i].x,currentPosTimeD[i].y);
        //     }
        // }
        // return min_dis;
        return orig.distance( x, y-h(x));
    }

    PVector WallNormal(float x, float y) {// adjust orig normal
        PVector n = orig.WallNormal(x, y);
        n.x -= dhdx(x,y)*n.y;
        float m = n.mag();
        if(m>0) return PVector.div(n,m);
        else return n; 
    }

    float velocity(int d, float dt, float x, float y){ // use 'wave' velocity
        float v = super.velocity(d,dt,x,y);

        PVector[] currentTest = positionsList.get(index);
        int pt_index = 0;
        float min_dis = 1e10;
        for (int i = 0; i < currentTest.length; i++){
            if (abs(x - currentTest[i].x) < min_dis) {
            // if (dist(x, y, currentPosTimeD[i].x, currentPosTimeD[i].y) <  min_dis) {
                pt_index = i;
                min_dis = abs(x - currentTest[i].x);
                // min_dis = dist(x, y, currentPosTimeD[i].x, currentPosTimeD[i].y);
            }
        }

        if ((x > 10) & (x < 31) & (y > 20) & (y < 40)) {
            println(x, y, currentTest[pt_index].x, currentTest[pt_index].y, hdot(x,y), hdot1(x,y), index, time);
        }
        if(d==1) return v;
        else return v+hdot(x,y);
    }

    void translate(float dx, float dy) {
        super.translate(dx,dy);
        orig.translate(dx,dy);
        x0_dep += dx;
        y0_dep += dy;
    }

    void rotate(float dphi) {} // no rotation

    void update( float time) { // update 'time' and coords

        // Calculate the index based on currentTime
        // int index = int(map(currentTime, startTime, endTime, 0, numColumns - 1));
        // index = constrain(index, 0, numColumns - 1);
        int index = (int)(2 * time - 1);
        this.index = index;
        this.time = time;

        if (index < numColumns) {
            for (int i=0; i<coords.size(); i++) coords.set(i,orig.coords.get(i).copy());
            for (int k=0; k<numRows; k++) {
                coords.get(k).x = x0_dep + positionsList.get(index)[k].x;
                coords.get(k).y = y0_dep + positionsList.get(index)[k].y;
            }
            // Update currentTime
            currentTime += 1;
            if (currentTime > endTime) {
                currentTime = startTime;
            }
        }
        getOrth();
    }

    boolean unsteady() {return true;}

    // With this geometry, we don't know the global deformation
    // We thus have to approximate the derivative numerically: dy/dx = y(n+1)-y(n-1)/2DeltaX
    // We keep the names used in FlexNACA

    float spaceDerivative(int a, int b, int c) {
        if (xTable.getFloat(a,c) == xTable.getFloat(b,c)){
            return 1e10;
        }
        else {
            return (yTable.getFloat(a,c)-yTable.getFloat(b,c))/(xTable.getFloat(a,c)-xTable.getFloat(b,c));
        }
    }

    float hdot(float x, float y) {
        // Find the closest point to (x,y) to get the corresponding local time derivative
        PVector[] currentPosTimeD = positionsList.get(index);
        int pt_index = 0;
        float min_dis = 1e10;
        for (int i = 0; i < currentPosTimeD.length; i++){
            if (abs(x - currentPosTimeD[i].x) < min_dis) {
            // if (dist(x, y, currentPosTimeD[i].x, currentPosTimeD[i].y) <  min_dis) {
                pt_index = i;
                min_dis = abs(x - currentPosTimeD[i].x);
                // min_dis = dist(x, y, currentPosTimeD[i].x, currentPosTimeD[i].y);
            }
        }
        float coef_time = timeDerivativesList.get(index)[pt_index];
        return coef_time;
    }

    float dhdx(float x, float y) {
        // Find the closest point to (x,y) to get the corresponding local derivative
        PVector[] currentPosSpaceD = positionsList.get(index);
        int pt_index = 0;
        float min_dis = 1e10;
        for (int i = 0; i < currentPosSpaceD.length; i++){
            if (dist(x,y,currentPosSpaceD[i].x,currentPosSpaceD[i].y) < min_dis) {
                pt_index = i;
                min_dis = dist(x,y,currentPosSpaceD[i].x,currentPosSpaceD[i].y);
            }
        }
        float coef_derivative = spaceDerivativesList.get(index)[pt_index];
        return coef_derivative;
    }

    float Ax( float x){
        float amp = a[0];
        for (int i=1; i<a.length; i++) amp += a[i]*pow((x-xc)/c,i);
        return amp;
    }
    float arg( float x) { return k*(x - xc)-omega*time;}
    float h( float x) { return Ax(x)*sin(arg(x));}
    float hdot1( float x, float y) {return -Ax(x)*omega*cos(arg(x));}
    // float dhdx1( float x, float y) {
    //     float amp = a[1]/c;
    //     for (int i=2; i<a.length; i++) amp += a[i]*(float)i/c*pow((x - xc)/c, i-1);
    //     return Ax(x)*k*cos(arg(x))+amp*sin(arg(x));
    // }
}