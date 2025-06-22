# -*- coding: utf-8 -*-
"""
NMR Spectra Simulator 1.0.0

Copyright (C) 2023 Mestrelab Research S.L. All rights reserved.

Author: Gary Sharman (gary.sharman@mestrelab.com)
"""

from nmrsim import NMRSimulator, SpinSystem
import json
import random

def getRand(x) :
    return (random.random()-0.5)*x

nmrsim = NMRSimulator()
nmrsim.setField(400)
nmrsim.setTD(2**14)
nmrsim.setSWPPM(10)
nmrsim.setOffsetPPM(5)
nmrsim.setT2(2.0)

labels = ['none', '1,4', '1,2,4', '1,2', '1,3,5', '1,2,3,4', '1', '1,2,3']
alipahticLabels = ['none', 'ethyl', 'methyl', 'CHCH2']


for label in labels:
    for alLabel in alipahticLabels:

        for i in range(400) :
        
            ssList = []
            #set up spin systems here
            
            # aliphatic none
            if alLabel=='none':
                pass
                
            # ethyl
            if alLabel=='ethyl':
                r1 = getRand(0.2)
                r2 = getRand(0.2)
                r3 = getRand(0.8)

                ss = SpinSystem()
                ss.addNuc('a',1.0+r1);
                ss.addNuc('b',1.0+r1);
                ss.addNuc('c',1.0+r1);
                ss.addNuc('d',2.4+r2);
                ss.addNuc('e',2.4+r2);
                ss.addJ('a','d',7+r3)
                ss.addJ('b','d',7+r3)
                ss.addJ('c','d',7+r3)
                ss.addJ('a','e',7+r3)
                ss.addJ('b','e',7+r3)
                ss.addJ('c','e',7+r3)
                ss.level=1
                ssList.append(ss)
                
            # methyl
            if alLabel=='methyl':
                r1 = getRand(0.2)

                ss = SpinSystem()
                ss.addNuc('a',2.1+r1);
                ss.addNuc('b',2.1+r1);
                ss.addNuc('c',2.1+r1);
                ss.level=1
                ssList.append(ss)
                
            # CHCH2
            if alLabel=='CHCH2':
                r1 = getRand(0.2)
                r2 = getRand(0.2)
                r3 = getRand(0.2)
                r4 = getRand(0.8)
                r5 = getRand(0.8)
                r6 = getRand(0.8)
                ss = SpinSystem()
                ss.addNuc('a',3.3+r1);
                ss.addNuc('b',2.6+r2);
                ss.addNuc('c',2.5+r3);
                ss.addJ('a','b',8+r4)
                ss.addJ('a','c',3+r5)
                ss.addJ('b','c',13+r6)

                ss.level=1
                ssList.append(ss)
           
            # none
            if label=='none':
                pass
            
            # 1,2,3
            if label=='1,2,3':
                r1 = getRand(0.15)
                r2 = getRand(0.15)
                r3 = getRand(0.15)
                r4 = getRand(0.8)
                r5 = getRand(0.8)
                r6 = getRand(0.8)
                ss = SpinSystem()
                ss.addNuc('a',7.4+r1);
                ss.addNuc('b',7.8+r2);
                ss.addNuc('c',7.1+r3);
                ss.addJ('a','b',7+r4)
                ss.addJ('b','c',7+r5)
                ss.addJ('a','c',2+r6)
                ss.level=1
                ssList.append(ss)
            
            # 1,4
            if label=='1,4':
                r1 = getRand(0.15)
                r2 = getRand(0.15)
                r3 = getRand(0.8)
                r4 = getRand(0.8)
                ss = SpinSystem()
                ss.addNuc('a',7.4+r1);
                ss.addNuc('b',7.8+r2);
                ss.addNuc('c',7.4+r1);
                ss.addNuc('d',7.8+r2);
                ss.addJ('a','b',7+r3)
                ss.addJ('c','d',7+r3)
                ss.addJ('a','c',2+r4)
                ss.addJ('b','d',2+r4)
                ss.level=1
                ssList.append(ss)
            
            
            # 1,2,4
            if label=='1,2,4':
                r1 = getRand(0.15)
                r2 = getRand(0.15)
                r3 = getRand(0.8)
                r4 = getRand(0.8)
                ss = SpinSystem()
                ss.addNuc('a',7.4+r1);
                ss.addNuc('b',7.8+r2);
                ss.addNuc('c',7.2+r4);
                ss.addJ('a','b',7+r3)
                ss.addJ('b','c',7+r3)
                ss.addJ('a','c',2+r4)
                ss.level=1
                ssList.append(ss)
            
            # 1,2
            if label=='1,2':
                r1 = getRand(0.15)
                r2 = getRand(0.15)
                r3 = getRand(0.15)
                r4 = getRand(0.15)
                r5 = getRand(0.8)
                r6 = getRand(0.8)
                r7 = getRand(0.8)
                r8 = getRand(0.8)
                r9 = getRand(0.8)
                ss = SpinSystem()
                ss.addNuc('a',7.0+r1);
                ss.addNuc('b',7.3+r2);
                ss.addNuc('c',7.6+r3);
                ss.addNuc('d',7.9+r4);
                ss.addJ('a','b',7+r5)
                ss.addJ('b','c',7+r6)
                ss.addJ('c','d',7+r7)
                ss.addJ('a','c',2+r8)
                ss.addJ('b','d',2+r9)
                ss.level=1
                ssList.append(ss)
            
            # 1,3,5
            if label=='1,3,5':
                r1 = getRand(0.15)
                r2 = getRand(0.15)
                r3 = getRand(0.15)
                r4 = getRand(0.8)
                r5 = getRand(0.8)
                r6 = getRand(0.8)
                ss = SpinSystem()
                ss.addNuc('a',7.4+r1);
                ss.addNuc('b',7.8+r2);
                ss.addNuc('c',7.2+r3);
                ss.addJ('a','b',2+r4)
                ss.addJ('b','c',2+r5)
                ss.addJ('a','c',2+r6)
                ss.level=1
                ssList.append(ss)
            
            # 1,2,3,4
            if label=='1,2,3,4':
                r1 = getRand(0.15)
                r2 = getRand(0.15)
                r4 = getRand(0.8)
                ss = SpinSystem()
                ss.addNuc('a',7.4+r1);
                ss.addNuc('b',7.8+r2);
                ss.addJ('a','b',7+r4)
                ss.level=1
                ssList.append(ss)
            
            # 1
            if label=='1':
                r1 = getRand(0.15)
                r2 = getRand(0.15)
                r3 = getRand(0.15)
                r4 = getRand(0.8)
                r5 = getRand(0.8)
                r6 = getRand(0.8)
                r7 = getRand(0.8)
                r8 = getRand(0.8)
                ss = SpinSystem()
                ss.addNuc('a',7.1+r1);
                ss.addNuc('b',7.4+r2);
                ss.addNuc('c',7.1+r1);
                ss.addNuc('d',7.4+r2);
                ss.addNuc('e',7.7+r3);
                ss.addJ('a','b',7+r4)
                ss.addJ('c','d',7+r4)
                ss.addJ('b','e',7+r5)
                ss.addJ('d','e',7+r5)
                ss.addJ('a','e',2+r6)
                ss.addJ('c','e',2+r6)
                ss.addJ('a','c',2+r7)
                ss.addJ('b','d',2+r8)
                ss.level=1
                ssList.append(ss)
            

            scale = (0.01 + random.random()*0.99) * 10
            bias = random.random()*0.5 -0.25
            spc, fs = nmrsim.simSpectrum(ssList, scale, bias);
            
            # Serializing json
            outOb = {}
            outOb['datapoints']=[];
            for a in range(len(spc)):
                ob = {}
                ob['y']=spc[a];
                ob['x']=fs[a]
                outOb['datapoints'].append(ob)
                
            outOb['type']=label
            outOb['aliphaticType']=alLabel
            outOb['field']=nmrsim.field
            outOb['size']=nmrsim.td
            outOb['offset']=nmrsim.offset
            outOb['swp']=nmrsim.swp
            outOb['nucleus']=nmrsim.nucleus
            json_object = json.dumps(outOb)
             
            # Writing to sample.json
            with open("C:/nmr-simulation/nmr_"+str(i)+"_"+label+"__"+alLabel+".json", "w") as outfile:
                outfile.write(json_object)