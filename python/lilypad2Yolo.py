output_folder = "C:/Users/blagn771/Desktop/nacaDataset/rawLabel"
input_file = "C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/saved/pressure.txt"
n, m = 2**6, 2**6

output = []

with open(input_file, 'r') as pressure:
    for line in pressure:
        output.append(line)

output = output[4:]
if n == m :
    outputYolo = [[float(y)/n for y in x.split()] for x in output]
    for elmt in outputYolo:
        elmt[0] = 0
else:
    print("Need to implement this part")

count = 1
for elmt in outputYolo:
    strYolo = ' '.join(map(str, elmt))
    if count < 10000:
        with open(output_folder+'/frame-'+str(10000+count)[1:]+'.txt', 'w') as writeFile:
            writeFile.write(strYolo)
    else:
        with open(output_folder+'/frame-'+str(count)+'.txt', 'w') as writeFile:
            writeFile.write(strYolo)
    count += 1