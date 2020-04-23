#!/usr/bin/env python
# coding: utf-8


import coreFunctionsVX2
import multiprocessing

if __name__ ==  '__main__': 
    print("ACCVALMIN\tACCVALMAX\tACCVALMEAN\tPARAMS")

    
    seed = 190104698#seed = 6770320
    (ACCVALMIN,ACCVALMAX,ACCVALMEAN,ROCAREA,params,graph) = coreFunctionsVX2.plot(seed)
    print("%.4f\t%.4f\t%.4f\t%.4f\t%s"%(ACCVALMIN,ACCVALMAX,ACCVALMEAN,ROCAREA,str(params)))

    graph.write_png('CryoTree.png')
    import webbrowser
    webbrowser.open('CryoTree.png')
    
    



