#!/usr/bin/env python
# coding: utf-8


import numpy as np
import time
import coreFunctionsVX

if __name__ ==  '__main__': 
    print("ACCVALMIN\tACCVALMAX\tACCVALMEAN\tPARAMS")
    results=[]
    
    seed = 216504733
    (ACCVALMIN,ACCVALMAX,ACCVALMEAN,ROCAREA,params,graph) = coreFunctionsVX.plot(seed)
    print("%.4f\t%.4f\t%.4f\t%.4f\t%s"%(ACCVALMIN,ACCVALMAX,ACCVALMEAN,ROCAREA,str(params)))
            
    graph.write_png('immunaTree.png')
    import webbrowser
    webbrowser.open('immunaTree.png')



