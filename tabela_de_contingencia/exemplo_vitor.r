# 0  0  0  0  41
# 0  0  0  1  59
# 0  0  1  0  75  
# 0  0  1  1  25  
# 0  1  0  0  28
# 0  1  0  1  72
# 0  1  1  0  69
# 0  1  1  1  31
# 1  0  0  0  68
# 1  0  0  1  32
# 1  0  1  0  18
# 1  0  1  1  82
# 1  1  0  0  63
# 1  1  0  1  37
# 1  1  1  0  31
# 1  1  1  1  69

dados  = c(41, 59, 75, 25, 28, 72, 69, 31, 68, 32, 18, 82, 63, 37, 31, 69)
orient = c( 0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1)
ilumin = c( 0,  0,  0,  0,  1,  1,  1,  1,  0,  0,  0,  0,  1,  1,  1,  1)
transp = c( 0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  1,  1)
escol  = c( 0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1)

library(MASS)
library(gmodels)
mytable <- xtabs(dados ~ orient + ilumin + transp + escol)

# # 1. Mutual Independence: A, B, and C are pairwise independent.
print ("Mutual Independence")
loglm(~orient + ilumin + transp + escol, mytable)
#model.matrix(loglm(~orient + ilumin + transp + escol, mytable))

# # 2. Partial Independences:
print ("Orientacao e parcialmente independente do resto")
loglm(~orient + ilumin + transp + escol + ilumin*transp*escol, mytable)
#model.matrix(loglm(~orient + ilumin + transp + escol + ilumin*transp*escol, mytable))

print ("Iluminacao e parcialmente independente do resto")
loglm(~orient + ilumin + transp + escol + orient*transp*escol, mytable)

print ("Transporte e parcialmente independente do resto")
loglm(~orient + ilumin + transp + escol + orient*ilumin*escol, mytable)

print ("Escolha e parcialmente independente do resto")
loglm(~orient + ilumin + transp + escol + orient*ilumin*transp, mytable)

# de Loglinear Models http://www.statmethods.net/stats/frequencies.html
