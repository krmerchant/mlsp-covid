xs = [4.35,2.17,9.01]
type(xs[1])

data = open("data1.txt",'w')
for x in xs:
    #data.write(str(x) + "\n")
    data.write(str(x) + " ")
data.write("\n")
data.close()

#line.split('\n')
data  = open("data1.txt",'r')
#lines = data.readlines()
lines = data.readline().strip() # Read the first line and strip trailing spaces/newlines
data.close()

ys = [0.0 for i in range(len(lines))]

for i in range(len(ys)):
    #ys[i] = float(lines[i][:-1]) #get rid of '\n
    ys = [float(value) for value in lines.split()]  # Split by spaces and convert to floats

print(f"xs = {xs}")
print(f"ys = {ys}")
print(f"lines = {lines}")
print(f"lines[1] = {lines[1]}")


#features and labels 2 types covid and dieseased vs nondisesased
#co-occurance matrix

#dictionary
features_data = { "image1": {"prop1": xs[0], "prop2": xs[1], "prop3": xs[2]}}
'''
s = 'car lrerw wrw' 
s.split(' ')
headers, features
["label" ]
'''
