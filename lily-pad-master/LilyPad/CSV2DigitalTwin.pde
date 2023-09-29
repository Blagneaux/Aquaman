class CSV2DigitalTwin extends Body {
    ArrayList<PVector[]> positionsList; // List to store positions for each time step
    int numColumns;                     // Number of columns in the tables
    int numRows;                        // Number of rows in the tables
    Table xTable, yTable;               // CSV tables for x and y coordinates
    float startTime = 0;                // Start time
    float endTime;                      // End time
    float currentTime = 0;              // Current time for interpolation

    CSV2DigitalTwin(float x0, float y0, String xFilePath, String yFilePath, Window window) {
        super(x0, y0, window);
        xTable = loadTable(xFilePath);
        yTable = loadTable(yFilePath);

        // Get the number of columns in the tables
        numColumns = xTable.getColumnCount();
        endTime = numColumns;

        // Get the number of rows in the tables
        numRows = min(xTable.getRowCount(), yTable.getRowCount());

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

        // Draw the first state
        for (int k = 0; k < numRows; k++) {
            add(x0+positionsList.get(0)[k].x, y0+positionsList.get(0)[k].y);
        }
        end();
    }

    void update() {
    
        // Calculate the index based on currentTime
        int index = int(map(currentTime, startTime, endTime, 0, numColumns - 1));
        index = constrain(index, 0, numColumns - 1);
        
        // Calculate the interpolation factor based on currentTime
        float t = map(currentTime, startTime, endTime, 0, 1);
        
        // Interpolate between positions in the current and next columns
        if (index + 1 < numColumns) {
            PVector[] currentPositions = positionsList.get(index);
            PVector[] nextPositions = positionsList.get((index + 1)); // Wrap around at the end
            PVector[] interpolatedPositions = new PVector[currentPositions.length];
            for (int i = 0; i < currentPositions.length; i++) {
                float dx = nextPositions[i].x - currentPositions[i].x;
                float dy = nextPositions[i].y - currentPositions[i].y;
                interpolatedPositions[i] = new PVector(dx, dy);
            }
        
        
            // Update the shape using the interpolated positions
            body.translate(interpolatedPositions);
            
            // Update currentTime
            currentTime += 1;
            if (currentTime > endTime) {
                currentTime = startTime;
            }
        }
        else {
            body.translate(0,0);
        }
  }
}