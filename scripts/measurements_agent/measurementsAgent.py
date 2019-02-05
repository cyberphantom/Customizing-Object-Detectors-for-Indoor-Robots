#!/usr/bin/env python


def findDistance(initA=None, initD=None,  fstA=None, fstD=None, secA=None, secD=None):


    ppcm = 104.2

    delta_A1 = fstA - initA #in inch
    den1 = (fstD * 100) * (delta_A1 / 104.2)

    delta_A2 = secA - initA #in inch
    den2 = (secD * 100) * (delta_A2 / 104.2)


    Dist = ((den2 - den1) / ((delta_A2 - delta_A1) /104.2)) / 100
    #Size =  (initA / 104.2) * (Dist * 100) /  (((569.883) * (569.883)) +  ((568.007) * (568.007)))**(0.5)
    #w =
    return Dist
