# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 14:22:37 2022

@author: Dr. Heba
"""

#=============================================================================================
#This file for presenting the objective functions of the three engineering case studies of 
#Many Objective Reliability-Redundancy Allocation Problem
#=============================================================================================
import numpy as np

class EngApps():
    
    def IntializeX(self,case):
       
        case=case
        if case=="cs1":  #Series-parallel system
            m=5 #number of subsystems
            X1 = np.random.uniform(low=0, high=1, size=m) #the reliability of a component ri
            X2 = np.random.randint(low=1, high=5, size=(m,)) #the number of components ni
            X=[] # will be given not calculated here
            for x in X1:
                X.append(x)
            for x in X2:
                X.append(x)
            X=np.array([0.01807539494323529, 0.008120767199493955, 0.10593267029451878, 0.0003551234578614004 ,0.0009120808174647062 ,1,2, 1, 1 ,2])
        elif case=="cs2":#Overspeed protection for gas turbine
              m=4 #number of subsystems
              X1 = np.random.uniform(low=0.5, high=(1-(1/np.power(10,6))), size=m) #the reliability of a component ri
              X2 = np.random.randint(low=1, high=10, size=(m,)) #the number of components ni
              X=[] # will be given not calculated here
              for x in X1:
                  X.append(x)
              for x in X2:
                  X.append(x)
        elif case=="cs3": #Large-scale reliability allocation problem
             m=36 #number of subsystems
             X = np.random.randint(low=1, high=10, size=(m,)) #the number of components ni

        

        print("X after first initialization:")
        print ("X=",X)
        return X
    
    def _evaluate(self, x,case):
        X=x
        case=case
        if case=="cs1":  #Series-parallel system
           print("we are in cs1")
           Fit=[]
           f2=0
           f3=0
           f4=0
           R=[] #system reliability
           m=5 #number of subsystems
           #Calculating System reliability R for each subsystem, needed for f1
           for i in range(m):
               #print("i=",i)
               R.append(1-np.power(1-X[i], X[i+m]))
           print("R=",R)

           # the following paprameters are given in table 2 in the engineering application paper    
           wv=[2,4,5,8,4] #given and =w*power(v,2) in equation 6
           alpha=[2.5,1.45,0.541,0.541,2.1]#given
           beta= [1.5,1.5,1.5,1.5,1.5]#given
           w=[3.5,4,4,3,4.5]#given
           V=[180,180,180,180,180]#given
           C=[175,175,175,175,175]#given
           W=[100,100,100,100,100]#given
           
           f1 = 1-(1-R[0]*R[1])*(1-(R[2]+R[3]-(R[2]*R[3]))*R[4])
           print("f1=",f1)
           Fit.append(f1)
           for i in range(m):
               f2 = f2 + (wv[i]*np.power(X[i+m],2))
           print("f2=",f2)
           Fit.append(f2)
           for i in range(m):
               f3 = f3 + (alpha[i]/100000)*(np.power((-1000/np.log(X[i])),beta[i]))*(X[i+m]+np.exp(0.25*X[i+m]))
           print("f3=",f3)
           Fit.append(f3)
           
           min = w[0]*X[m]*np.exp(0.25*X[m])
           for i in range(m): 
               if i!=0:
                  temp = (w[i]*X[i+m]*np.exp(0.25*X[i+m]))
                  if temp<min:
                     min=temp
           f4=min
           print("f4=",f4)
           Fit.append(f4)
           if (f2-V[0])<=0 and (f3-C[0])<=0 and (f4-W[0])<=0:
              print("case=",case)
              print("Feasible Solution = ",X)
              print("With Fit = ",Fit)
              return Fit
           else:
               print("case=",case)
               print("Not Feasible Solution found in case={} and the algorithm will repair".format(case))
               return None
           
           
        elif case=="cs2":#Overspeed protection for gas turbine
             if case=="cs2":  
                print("we are in cs2") 
                Fit=[]
                f1=1
                f2=0
                f3=0
                f4=0
                m=4 #number of subsystems
                
                # the following paprameters are given in table 2 in the engineering application paper    
                alpha=[1,2.3,0.3,2.3]#given
                beta= [1.5,1.5,1.5,1.5]#given
                v=[1,2,3,2]#given
                w=[6,6,8,7]#given
                V=[250,250,250,250]
                C=[400,400,400,400]#given
                W=[500,500,500,500]#given
                T=[1000,1000,1000,1000]#given
                c=[]
                
                for i in range(m):
                    f1 = f1*(1-(np.power(1-X[i],X[i+m])))
                print("f1=",f1)
                Fit.append(f1)
                for i in range(m):
                    f2 = f2 + (w[i]*np.power(v[i],2)*np.power(X[i+m],2))
                print("f2=",f2)
                Fit.append(f2)
                #Calculating Cost for each subsystem, needed for f3
                for i in range(m):
                    c.append(alpha[i]/100000*(np.power(-1*T[i]/np.log(X[i]),beta[i])))
                
                for i in range(m):
                    f3 = f3 + (c[i]*(X[i+m] + np.exp(0.25*X[i+m])))
                print("f3=",f3)
                Fit.append(f3)
                
                min = w[0]*X[m]*numpy.exp(0.25*X[m])
                for i in range(m): 
                    if i!=0:
                       temp = (w[i]*X[i+m]*np.exp(0.25*X[i+m]))
                       if temp<min:
                          min=temp
                f4=min
                print("f4=",f4)
                Fit.append(f4)
                if (f2-V[0])<=0 and (f3-C[0])<=0 and (f4-W[0])<=0:
                   print("case=",case)
                   print("Feasible Solution = ",X)
                   print("With Fit = ",Fit)
                   return Fit
                else:
                    print("case=",case)
                    print("Not Feasible Solution found in case={} and the algorithm will repair".format(case))
                    return None
        elif case=="cs3": #Large-scale reliability allocation problem
             print("we are in cs3") 
             Fit=[]
             f1=1
             f2=0
             f3=0
             f4=0
             f5=0
             m=36 #number of subsystems
             

             # the following paprameters are given in table 2 in the engineering application paper    
             rev_r=[0.005,0.026,0.035,0.029,0.032,0.003,0.02,0.018,0.004,0.038,0.028,0.021,0.039,0.013,0.038,0.037,0.021,0.023,0.027,0.028,0.03,0.027,0.018,0.013,0.006,0.029,0.022,0.017,0.002,0.031,0.021,0.023,0.03,0.026,0.009,0.019]
             alpha=[8,10,10,6,7,10,9,9,7,6,6,10,9,10,7,10,10,8,10,7,6,6,7,8,9,8,8,9,10,9,7,9,6,7,6,10]#given
             beta= [4,4,4,3,1,4,2,3,4,4,5,3,1,4,4,2,1,3,5,4,2,2,2,5,5,1,3,3,1,2,5,5,3,3,5,5]#given
             gama= [13,16,12,12,13,16,19,15,12,16,14,15,17,20,14,13,15,19,18,13,15,12,20,19,15,18,16,15,18,19,15,11,15,14,15,17]#given
             delta= [26,32,23,24,26,31,38,29,23,31,28,30,34,39,28,25,29,38,36,26,30,24,40,38,39,35,32,29,35,37,28,22,29,27,29,33]#given
             
             
             for i in range(m):
                 f1 = f1*(1-(np.power(rev_r[i],X[i])))
             print("f1=",f1)
             Fit.append(f1)
             
             for i in range(m):
                 f2 = f2 + ((alpha[i]/100000)*np.power(X[i],2))
             print("f2=",f2)
             Fit.append(f2)
             
             for i in range(m):
                 f3 = f3 + ((beta[i]/100000)* np.exp(X[i]/2))
             print("f3=",f3)
             Fit.append(f3)
             
             for i in range(m): 
                 f4 = f4 + ((gama[i]/100000)*X[i])
             print("f4=",f4)
             Fit.append(f4)
             
             for i in range(m): 
                 f5 = f5 + ((delta[i]/100000)*np.sqrt(X[i]))
             print("f5=",f5)
             Fit.append(f5)
             
             if f2<=391 and f3<=257 and f4<=738 and f5<=1454:
                print("case=",case)
                print("Feasible Solution = ",X)
                print("With Fit = ",Fit)
                return Fit
             else:
                 print("case=",case)
                 print("Not Feasible Solution found in case={} and the algorithm will repair".format(case))
                 return None
    

 
case1= EngApps()
result=None
while True:
      #intialization of X depends on the case, each case has different bounds
      Xvector=case1.IntializeX("cs1")
      result=case1._evaluate(Xvector,"cs1")
      pass