"""
Plots the number of moves/time needed to solve a puzzle
EternaBrain vs Etera human players
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn; seaborn.set()

# players
x = [214,214,214,104,379,392,400,16,16,22,104,150,312,285,105,400,382,361,315,287,72,116]
yt = [219,273,134,123,346,319,512,32,20,23,104,211,280,330,150,527,441,479,396,323,65,110]
yb = [518,554,588,303,1009,1119,1201,10,30,40,257,333,600,506,169,982,907,955,882,767,166,274]
#eternabrain
#xe = [400,389,32,16,149,072,104,165,303,400,287,360,284,213,63,111,116,123]
ye = [518,554,588,303,1009,1119,1201,35,25,40,257,333,600,506,169,982,997,907,955,882,767,166,274]
yet = [518,554,588,303,1009,1119,1201,10,30,40,257,333,600,506,169,982,997,907,955,882,767,166]#,274]
y = [219,273,134,123,346,319,512,32,20,23,104,211,280,330,150,576,555,441,479,396,323,65,110]

# fractastar5, cuboid, gcplacement, cloudbeta, six legd turtle, water strider, stickshift, martian, tarax, chctrac, frac3, trtmoves, adenine, the sun
#x = [400     , 380   , 16         , 63       , 108            , 103          , 39        , 213    , 282  , 116    , 119  , 316     , 175    , 258]
yp = yt#[1224   , 1132   , 51        , 211      , 100            , 130          , 40        , 324    , 656  , 176    , 265  , 703     , 281    , 379]
yb = yet#[945    , 788   , 23         , 45       , 111             , 41          , 56        , 313    , 633  , 76     , 75   , 554     , 211    , 366]

print len(x), len(yp), len(yb)

ypt = [i ** 0.5 for i in yp]
ybt = [j ** 0.5 for j in yb]
mp,bp = np.polyfit(x,yp,1)
me,be = np.polyfit(x,yb,1)
p_eq = np.polyfit(x,yp,1)
b_eq = np.polyfit(x,yb,1)

print p_eq
print b_eq

print bp,'+',mp,'x'
print be,'+',me,'x'
# player_eq = np.poly1d(player_line)
# brain_line = np.poly1d(brain_line)


def player_eq_list(x, degree):
    ylist = []
    for i in x:
        #ylist.append(bp + mp * i)
        if degree == 2:
            ylist.append(p_eq[2] + (p_eq[1] * i) + (p_eq[0] * (i**2)))
        elif degree == 1:
            ylist.append(p_eq[1] + (p_eq[0] * i))
    return ylist


def brain_eq_list(x, degree):
    ylist = []
    for i in x:
        #ylist.append(be + me * i)
        if degree == 2:
            ylist.append(b_eq[2] + (b_eq[1] * i) + (b_eq[0] * (i**2)))
        elif degree == 1:
            ylist.append(b_eq[1] + (b_eq[0] * i))
    return ylist

plt.scatter(x,yp,label='Top Eterna Players')
plt.scatter(x,yb,label='EternaBrain')

plt.plot(range(401),player_eq_list(range(401), 1),label='Line of best fit for Players')
plt.plot(range(401),brain_eq_list(range(401), 1),label='Line of best fit for EternaBrain')

plt.title("Moves needed to solve a puzzle of a given length")
plt.xlabel("Length of the puzzle (in number of nucleotides)")
plt.ylabel("Number of moves")
plt.ylim([-50,1300])

plt.legend()
plt.show()
