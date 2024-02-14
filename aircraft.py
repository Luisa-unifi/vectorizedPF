"""
Python prototype implementation of the vectorized PF algorithm
described in "Feynman-Kac models for universal probabilistic 
programming" ([1]), Section 6.

In this script we consider the aircraft tracking scenario described in [1]:
the aircraft is seen as a point moving on a 2D plane for t discrete time steps.
Through these time steps, the aircraft is tracked by six radars, measuring 
the distance between the aircraft and the radar, in particular:
       - xx: horizontal position of the aircraft.
       - yy: vertical position of the aircraft.
       - tt: time step.

The implementation is based on TensorFlow and autograph:
https://github.com/tensorflow/tensorflow/blob/master/
tensorflow/python/autograph/g3doc/reference/index.md.
"""

import time
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions




@tf.function (input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]*10+[tf.TensorSpec(shape=None, dtype=tf.int32)]) 
def ftrack(xx,yy,tt,SS,W,ZERO,ONE,SEVEN,ZZ,ones,N): 
    print('Warm up')
        
    Obs = tf.constant([                                   
           [2.0, 1.80278, 0.5, 3.0, 4.0, 2.0],
           [1.5, 1.5, 1.5, 3.0, 4.0, 2.0],
           [1.65529, 1.65529, 2.0, 3.0, 4.0, 2.0],
           [2.0, 1.0, 2.0, 2.88617, 4.0, 2.0],
           [2.0, 0.70711, 1.78885, 2.75862, 4.0, 2.0],
           [2.0, 1.78885, 2.0, 3.0, 3.25576, 2.0],
           [2.0, 2.0, 2.0, 2.84429, 2.54951, 1.50333],
           [2.0, 2.0, 2.0, 2.75862, 3.54401, 2.0],
           [0.0]*6                                  
           ])      
        
    def fbody(xx,yy,tt,SS,W):     

        ### RESAMPLING ######
        EW=tf.math.exp(W)
        P=EW/tf.reduce_sum(EW)
        cum_dist = tf.math.cumsum(P[0])
        cum_dist /= cum_dist[-1]  # to account for floating point errors
        unif_samp = tf.random.uniform((N,), 0, 1)
        rs = tf.searchsorted(cum_dist, unif_samp)  # indices for resampling
        state_t = tf.concat([xx,yy,tt,SS],axis=0)    # state tensor  
        state_t = tf.gather(state_t,rs,axis=1) # resampled state tensor
        xx,yy,tt,SS= tuple([state_t[tf.newaxis,j] for j in range(4)])  # sliced state_t
        W=ZZ
        #### END RESAMPLING 1 ##########
        
        
        # MASKS DEFINITIONS
        mask0= (SS==0.0)
        mask1=(SS==1.0)&(tt>=0)&(tt<=7)
        mask2=(SS==1.0)&(tt>7)

        # STATE 0
        xx=tf.where(mask0,tfd.Normal(loc=2*ones, scale=1*ones).sample(),xx)
        yy=tf.where(mask0,tfd.Normal(loc=-1.5*ones, scale=1*ones).sample(),yy)
        SS=tf.where(mask0,ONE,SS) 
        
        SS=tf.where(mask2,SEVEN,SS) # FINAL STATE

        # STATE 1
        xx=tf.where(mask1,tfd.Normal(loc=xx, scale=2*ones).sample(),xx)
        yy=tf.where(mask1,tfd.Normal(loc=yy, scale=2*ones).sample(),yy)
        
        nn0=tfd.TruncatedNormal(loc=ZZ, scale=1, low=0.0, high=radius[0]).sample()
        computed_distance=(xx-rad_x[0])**2+(yy-rad_y[0])**2
        radius2=tf.where(tfd.Bernoulli(probs=0.999*ones).sample()==1,radius[0],radius[0]+0.001*nn0)
        measured_distance = tf.where((computed_distance>radius_sq[0]) ,radius2,tf.sqrt(computed_distance)+0.1*nn0)
        observation=Obs[int(tt[0][0])][0]*ones
        W=tf.where(mask1,tf.math.log(tfd.Normal(loc=measured_distance, scale=.01).prob(observation)),W)
        
        nn1=tfd.TruncatedNormal(loc=ZZ, scale=1, low=0.0, high=radius[1]).sample()
        computed_distance=(xx-rad_x[1])**2+(yy-rad_y[1])**2        
        radius2=tf.where(tfd.Bernoulli(probs=0.999*ones).sample()==1,radius[1],radius[1]+0.001*nn1)       
        measured_distance = tf.where((computed_distance>radius_sq[1]),radius2,tf.sqrt(computed_distance)+0.1*nn1)   
        observation=Obs[int(tt[0][0])][1]*ones
        W=tf.where(mask1,W+tf.math.log(tfd.Normal(loc=measured_distance, scale=.01).prob(observation)),W)
        
        nn2=tfd.TruncatedNormal(loc=ZZ, scale=1, low=0.0, high=radius[2]).sample()   
        computed_distance=(xx-rad_x[2])**2+(yy-rad_y[2])**2
        radius2=tf.where(tfd.Bernoulli(probs=0.999*ones).sample()==1,radius[2],radius[2]+0.001*nn2)
        measured_distance = tf.where((computed_distance>radius_sq[2]),radius2,tf.sqrt(computed_distance)+0.1*nn2)
        observation=Obs[int(tt[0][0])][2]*ones
        W=tf.where(mask1,W+tf.math.log(tfd.Normal(loc=measured_distance, scale=.01).prob(observation)),W)
           
        nn3=tfd.TruncatedNormal(loc=ZZ, scale=1, low=0.0, high=radius[3]).sample()  
        computed_distance=(xx-rad_x[3])**2+(yy-rad_y[3])**2
        radius2=tf.where(tfd.Bernoulli(probs=0.999*ones).sample()==1,radius[3],radius[3]+0.001*nn3)
        measured_distance = tf.where((computed_distance>radius_sq[3]),radius2,tf.sqrt(computed_distance)+0.1*nn3)
        observation=Obs[int(tt[0][0])][3]*ones
        W=tf.where(mask1,W+tf.math.log(tfd.Normal(loc=measured_distance, scale=.01).prob(observation)),W)
        
        nn4=tfd.TruncatedNormal(loc=ZZ, scale=1, low=0.0, high=radius[4]).sample()  
        computed_distance=(xx-rad_x[4])**2+(yy-rad_y[4])**2
        radius2=tf.where(tfd.Bernoulli(probs=0.999*ones).sample()==1,radius[4],radius[4]+0.001*nn4)
        measured_distance = tf.where((computed_distance>radius_sq[4]),radius2,tf.sqrt(computed_distance)+0.1*nn4)
        observation=Obs[int(tt[0][0])][4]*ones
        W=tf.where(mask1,W+tf.math.log(tfd.Normal(loc=measured_distance, scale=.01).prob(observation)),W)
        
        nn5=tfd.TruncatedNormal(loc=ZZ, scale=1, low=0.0, high=radius[5]).sample()  
        computed_distance=(xx-rad_x[5])**2+(yy-rad_y[5])**2
        radius2=tf.where(tfd.Bernoulli(probs=0.999*ones).sample()==1,radius[5],radius[5]+0.001*nn5)
        measured_distance = tf.where((computed_distance>radius_sq[5]),radius2,tf.sqrt(computed_distance)+0.1*nn5)
        observation=Obs[int(tt[0][0])][5]*ones
        W=tf.where(mask1,W+tf.math.log(tfd.Normal(loc=measured_distance, scale=.01).prob(observation)),W)
        
        tt=tf.where(mask1,tt+1,tt)
        
        SS=tf.where(mask1,ONE,SS)
                     
        return  (xx,yy,tt,SS,W) 

    def fcond(xx,yy,tt,ss,W):
        return True 
    
    xx,yy,tt,SS,W=tf.while_loop(fcond, fbody, (xx,yy,tt,SS,W), maximum_iterations=10,parallel_iterations=5,back_prop=True)  #maximum_iterations = fiter !
    return xx,yy,tt,SS,W


#number of particles
N=10**3

# radar position     
rad_x =  [0.0+tf.zeros((1,N)), 3.0+tf.zeros((1,N)), 1.5+tf.zeros((1,N)), 5.0+tf.zeros((1,N)), 6.0+tf.zeros((1,N)), 5.6+tf.zeros((1,N))]      
rad_y =  [0.0+tf.zeros((1,N)), 0.0+tf.zeros((1,N)), -1.5+tf.zeros((1,N)), 1.3+tf.zeros((1,N)), -4.0+tf.zeros((1,N)), -3.0+tf.zeros((1,N))]

# radar radius
radius = [2.0+tf.zeros((1,N)), 2.0+tf.zeros((1,N)), 2.0+tf.zeros((1,N)), 3.0+tf.zeros((1,N)), 4.0+tf.zeros((1,N)), 2.0+tf.zeros((1,N))] 
radius_sq = [4.0+tf.zeros((1,N)), 4.0+tf.zeros((1,N)), 4.0+tf.zeros((1,N)), 9.0+tf.zeros((1,N)), 16.0+tf.zeros((1,N)), 4.0+tf.zeros((1,N))]


ZZ=tf.zeros((1,N))
start_time=time.time()
res=ftrack(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,1+ZZ,7+ZZ,ZZ,1+ZZ,N)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M elems  %s seconds -------        " % final_time)



#weights
W=res[-1]
#final state
S=res[-2]
#output function
R=res[0]   

# lower bound computation
EW=tf.math.exp(W)
P=EW/tf.reduce_sum(EW)  
l_L = tf.reduce_sum(R*P)    

# effective sample size for W
ess = tf.reduce_sum(EW)**2/(tf.reduce_sum(EW**2)) 

# upper bound computation
nil_n = 7.0  
Term = tf.where(S==nil_n,1.0, 0.0)     
Wu = W*Term     
EWu=tf.math.exp(Wu)
Pu=EWu/tf.reduce_sum(EWu) 
l_U = tf.reduce_sum(R*Pu) + (1/tf.reduce_sum(Term*Pu)-1)

# effective sample size for Wu   
ess_u = tf.reduce_sum(EWu)**2/(tf.reduce_sum(EWu**2)) 

#normalizing constant
norm = EW.numpy().sum()/len(EW.numpy()[0])
