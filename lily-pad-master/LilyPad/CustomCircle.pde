class CSV2CircleTwin extends CircleBody {
  Table spineTable;
  int numRows;
  float time = 0;
  int index = 0;

  CSV2CircleTwin(float d, Table spineFilePath, Window window, int st) {
    super(0, 0, d, window); // Initialize at origin for now
    this.spineTable = cropTableRows(spineFilePath, startIndex);
    this.numRows = this.spineTable.getRowCount();

    // Immediately position at time zero
    update(0);
  }

  void update(float time) {
    // Calculate the index based on currentTime
    int index = (int)(time / 0.686778 - 1);
    this.index = index;
    this.time = time;

    // Clamp index to valid range
    index = max(0, min(index, numRows - 1));

    // Get the CSV values from columns 69 and 70
    float val69 = spineTable.getFloat(index, 69);
    float val70 = spineTable.getFloat(index, 70);

    // Compute new center position
    float cx = 256 * val69 / 2000;
    float cy = 128 * val70 / 1200;

    // Move the body to this position
    float dx = cx - xc.x;
    float dy = cy - xc.y;
    translate(dx, dy);
  }

  boolean unsteady() { return true; }

  void rotate(float dphi) {
    // No-op, no rotation needed
  }

  Table cropTableRows(Table table, int startRowIndex) {
    Table croppedTable = new Table(); // Create a new table to store cropped data

    // Copy column structure (preserve all columns)
    for (int col = 0; col < table.getColumnCount(); col++) {
        String columnTitle = table.getColumnTitle(col);
        croppedTable.addColumn(columnTitle);
    }

    // Copy rows from startRowIndex to the end of the table
    for (int row = startRowIndex; row < table.getRowCount(); row++) {
        TableRow originalRow = table.getRow(row);
        TableRow newRow = croppedTable.addRow();
        
        for (int col = 0; col < table.getColumnCount(); col++) {
        newRow.setFloat(col, originalRow.getFloat(col));
        }
    }

    return croppedTable;
    }
}
