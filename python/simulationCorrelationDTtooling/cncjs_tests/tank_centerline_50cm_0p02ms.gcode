; ASMR CNCjs standalone tank test
; Tank: 750.0 x 690.0 mm
; Path: X 225.0 -> 525.0 mm at Y 0.0 mm
; Max speed: 0.0100 m/s (600.0 mm/min)
$X
$H
G21
G90
G94
G0 X225.000 Y0.000 Z0.000
G1 X525.000 Y0.000 Z0.000 F600.000
M2
