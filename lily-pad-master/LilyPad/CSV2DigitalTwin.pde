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

    float prevNoseX; // Previous x-coordinate of the nose
    float prevDx;    // Previous horizontal velocity

    // For test purpose only
    float[] a={0,.2,-.1};
    float k, omega, T, xc, time=0;
    float x_init = (int)pow(2,6)/4;
    float c = (int)pow(2,6)/3;

    CSV2DigitalTwin(float x0, float y0, int m, String xFilePath, String yFilePath, String yFilteredFilePath, Window window, int startIndex) {
        // Just as in flex NACA, set a regular NACA coords and save as orig
        super(x0+0.25*(int)pow(2,6)/3, y0, (int)pow(2,6)/3, 0.2, m / 2, window);
        orig = new NACA(x0+0.25*(int)pow(2,6)/3,y0,(int)pow(2,6)/3,0.2,m/2,window);

        // Load the coordinates
        xTable = loadTable(xFilePath);
        yTable = loadTable(yFilePath);
        yFilteredTable = loadTable(yFilteredFilePath);
    
        xTable = cropTableColumns(xTable, startIndex);
        yTable = cropTableColumns(yTable, startIndex);
        yFilteredTable = cropTableColumns(yFilteredTable, startIndex);

        this.prevNoseX = coords.get(0).x; // Set the initial nose position
        this.prevDx = 0;                  // Initial velocity is zero


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
            coords.get(k).x = positionsList.get(0)[k].x;
            coords.get(k).y = positionsList.get(0)[k].y;
        }
        
        end(true, true);

        PVector mn = positionsList.get(0)[0].copy(), mx = positionsList.get(0)[0].copy();
        for (int k=0; k<numColumns; k++) {
            for (int i=0; i<numRows; i++) {
                mn.x = min(mn.x, positionsList.get(k)[i].x);
                mn.y = min(mn.y, positionsList.get(k)[i].y);
                mx.x = max(mx.x, positionsList.get(k)[i].x);
                mx.y = max(mx.y, positionsList.get(k)[i].y);
            }
        }
    }

    CSV2DigitalTwin(float x0, float y0, int m, String xFilePath, String yFilePath, String yFilteredFilePath, Window window) {
        this(x0, y0, m, xFilePath, yFilePath, yFilteredFilePath, window, 0);
    }

    Table cropTableColumns(Table table, int startIndex) {
        Table croppedTable = new Table(); // Create a new table to store cropped data
        
        // Add columns from startIndex to the end of the original table
        for (int col = startIndex; col < table.getColumnCount(); col++) {
            String columnTitle = table.getColumnTitle(col);
            croppedTable.addColumn(columnTitle); // Add the column with the same title
        }
        
        // Copy rows into the cropped table
        for (int row = 0; row < table.getRowCount(); row++) {
            TableRow originalRow = table.getRow(row);
            TableRow newRow = croppedTable.addRow(); // Add a new row to the cropped table
            
            for (int col = startIndex; col < table.getColumnCount(); col++) {
                // Copy data from the original table to the cropped table
                newRow.setFloat(col - startIndex, originalRow.getFloat(col));
            }
        }
        
        return croppedTable; // Return the cropped table
    }

    boolean isUTurn() {
        float currentNoseX = coords.get(0).x; // Current x-coordinate of the nose
        float dx = currentNoseX - prevNoseX; // Compute the change in x-direction (velocity)

        // Detect if the direction of the nose changes
        boolean isUTurn = (dx < 0 && prevDx > 0) || (dx > 0 && prevDx < 0);

        // Update previous values for the next check
        prevNoseX = currentNoseX;
        prevDx = dx;

        return isUTurn;
    }

    void updateBoxFromCoordsXY(float padX, float padY) {
        if (box == null || box.coords.size() < 4) return;

        float mnx =  1e10, mxx = -1e10;
        float mny =  1e10, mxy = -1e10;

        for (PVector p : coords) {
            mnx = min(mnx, p.x);
            mxx = max(mxx, p.x);
            mny = min(mny, p.y);
            mxy = max(mxy, p.y);
        }

        float x0 = mnx - padX;
        float x1 = mxx + padX;
        float y0 = mny - padY;
        float y1 = mxy + padY;

        // Rectangle CCW
        box.coords.get(0).set(x0, y0);
        box.coords.get(1).set(x0, y1-3);
        box.coords.get(2).set(x1-3, y1-3);
        box.coords.get(3).set(x1-3, y0);

        box.getOrth();
        box.convex = true;
        }


    // float distance(float x, float y) {
    //     return orig.distance(x, y-h(x));
    // }

    float distance( float x, float y ) { // in cells
    float dis;
    if (n>4) { // distance to bounding box
      dis = box.distance(x, y);
      if (dis>=3) return dis;
    }
    
    if(convex){ // distance to convex body
      // check distance to each line, choose max
      dis = -1e10;
      for ( OrthoNormal o : orth ) dis = max(dis, o.distance(x, y));
      return dis;
    } else {   // distance to non-convex body
      // check distance to each line segment, choose min
      dis = 1e10;
      for( OrthoNormal o: orth ) dis = min(dis,o.distance(x,y,false));
      return (wn(x,y)==0)?dis:-dis; // use winding to set inside/outside
    }
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

    void update(float time) {
        int index = (int)(time / 0.686778 - 1);
        index = max(0, min(index, numColumns - 1));
        this.index = index;
        this.time = time;

        // Restore base geometry then overwrite with CSV pose
        for (int i = 0; i < numRows; i++) coords.set(i, orig.coords.get(i).copy());

        for (int k = 0; k < numRows; k++) {
            coords.get(k).x = x0_dep + positionsList.get(index)[k].x;
            coords.get(k).y = y0_dep + positionsList.get(index)[k].y;
        }

        // Adaptive bbox in BOTH X and Y (tune pads)
        updateBoxFromCoordsXY(0, 1);

        getOrth();
        }


    boolean unsteady() {return true;}

    // With this geometry, we don't know the global deformation
    // We thus have to approximate the derivative numerically: dy/dx = y(n+1)-y(n-1)/2DeltaX

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
            if (dist(x, y, currentPosTimeD[i].x, currentPosTimeD[i].y) <  min_dis) {
                pt_index = i;
                min_dis = dist(x, y, currentPosTimeD[i].x, currentPosTimeD[i].y);
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

    // float Ax( float x){
    //     float amp = a[0];
    //     for (int i=1; i<a.length; i++) amp += a[i]*pow((x-xc)/c,i);
    //     return amp;
    // }
    // float arg( float x) { return k*(x - xc)-omega*time;}
    // float h( float x) { return Ax(x)*sin(arg(x));}
    // float hdot1( float x, float y) {return -Ax(x)*omega*cos(arg(x));}
    // float dhdx1( float x, float y) {
    //     float amp = a[1]/c;
    //     for (int i=2; i<a.length; i++) amp += a[i]*(float)i/c*pow((x - xc)/c, i-1);
    //     return Ax(x)*k*cos(arg(x))+amp*sin(arg(x));
    // }
}
