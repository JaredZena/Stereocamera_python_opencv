# Playing with files
"""

file = open("testfile.txt","w")

M = [[3,4],[3,7],[4,8],[1,2]]
 
file.write("Hello World\n") 
file.write("This is our new text file") 
file.write("and this is another line.") 
file.write(str(M)) 
 
file.close() """

Intrinsic_mtx_1_file = open("Intrinsic_mtx_1_file.txt","r")
dist_1_file = open("dist_1_file.txt","r")
Intrinsic_mtx_2_file = open("Intrinsic_mtx_2_file.txt","r")
dist_2_file = open("dist_2_file.txt","r")
R_file = open("R_file.txt","r")
T_file = open("T_file.txt","r")
E_file = open("E_file.txt","r")
F_file = open("F_file.txt","r")

M1 = Intrinsic_mtx_1_file.read()
d1 = dist_1_file.read()
M2 = Intrinsic_mtx_2_file.read()
d2 = dist_2_file.read()
R = R_file.read()
T = T_file.read()
E = E_file.read()
F = F_file.read()

Intrinsic_mtx_1_file.close()
dist_1_file.close()
Intrinsic_mtx_2_file.close()
dist_1_file.close()
R_file.close()
T_file.close()
E_file.close()
F_file.close()

print("Intrinsic_mtx_1\n" + M1 + "\n\n")
print("dist_1\n" + d1 + "\n\n")
print("Intrinsic_mtx_2\n" + M2 + "\n\n")
print("dist_2\n" + d2 + "\n\n")
print("R\n" + R + "\n\n")
print("T\n" + T + "\n\n")
print("E\n" + F + "\n\n")
print("F\n" + E)
