"""
Python prototype implementation of the vectorized PF algorithm
described in "Feynman-Kac models for universal probabilistic 
programming" ([1]), Section 6.

In this script we consider a variation on the classical drunk man’s
walk random process, in which a drunk man and a mouse perform 
independent random walks, in particular:
       - xx: man position.
       - yy: mouse position.
       - dd: man variance.
       - rr: mouse variance.

The implementation is based on TensorFlow and autograph:
https://github.com/tensorflow/tensorflow/blob/master/
tensorflow/python/autograph/g3doc/reference/index.md.
"""

import time
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions



@tf.function (input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]*12) 
def running_example(xx,yy,dd,rr,tt,SS,W,ZERO,ONE,SEVEN,ZZ,ones):  #with resampling
    print('Warm up')
    def fbody(xx,yy,dd,rr,tt,SS,W):
        ### RESAMPLING 1 ######
        EW=tf.math.exp(W)
        P=EW/tf.reduce_sum(EW)
        cum_dist = tf.math.cumsum(P[0])
        cum_dist /= cum_dist[-1]  # to account for floating point errors
        unif_samp = tf.random.uniform((N,), 0, 1)
        rs = tf.searchsorted(cum_dist, unif_samp)  # indices for resampling
        state_t = tf.concat([xx,yy,dd,rr,tt,SS],axis=0)    # state tensor  
        state_t = tf.gather(state_t,rs,axis=1) # resampled state tensor
        xx,yy,dd,rr,tt,SS= tuple([state_t[tf.newaxis,j] for j in range(6)])  # sliced state_t
        #W=ZZ
        #### END RESAMPLING 1 ##########
        
    
        # MASKS DEFINITIONS
        mask0= (SS==0.0)
        mask1=(SS==1.0)& (abs(xx-yy)>=1/10)
        mask2=(SS==1.0)& (abs(xx-yy)<1/10)

        # STAE 0
        dd=tf.where(mask0,tfd.Uniform(0*ones, 2*ones).sample(),dd)
        rr=tf.where(mask0,tfd.Uniform(0*ones, ones).sample(),rr)        
        xx=tf.where(mask0,-1*ones,xx)
        yy=tf.where(mask0,ones,yy)
        SS=tf.where(mask0,ONE,SS) 


        SS=tf.where(mask2,SEVEN,SS) # FINAL STATE
        
        # STATE 1
        xx=tf.where(mask1,tfd.Normal(loc=xx, scale=dd).sample(),xx)
        yy=tf.where(mask1,tfd.Normal(loc=yy, scale=rr).sample(),yy)
        W=tf.where(mask1,W+tf.math.log(tf.cast(abs(xx-yy) <= 3,tf.float32)),W)
        SS=tf.where(mask1,ONE,SS)
                     
        return  (xx,yy,dd,rr,tt,SS,W) 

    def fcond(xx,yy,dd,rr,tt,ss,W):
        return True 
    
    xx,yy,dd,rr,tt,SS,W=tf.while_loop(fcond, fbody, (xx,yy,dd,rr,tt,SS,W), maximum_iterations=100,parallel_iterations=5,back_prop=True)  #maximum_iterations = fiter !
    return xx,yy,dd,rr,tt,SS,W



#number of particles
N=10**5

ZZ=tf.zeros((1,N))
start_time=time.time()
res=running_example(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,1+ZZ,7+ZZ,ZZ,1+ZZ)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M elems  %s seconds -------        " % final_time)


#weights
W=res[-1]
#final state
S=res[-2]
#output function
R=res[2]   


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
EWu=tf.where(tf.math.is_nan(EWu),0,EWu)

Pu=EWu/tf.reduce_sum(EWu) #
l_U = tf.reduce_sum(R*Pu) + 2*(1/tf.reduce_sum(Term*Pu)-1)    

# effective sample size for Wu.
ess_u = tf.reduce_sum(EWu)**2/(tf.reduce_sum(EWu**2)) 

#normalizing constant
norm = EW.numpy().sum()/len(EW.numpy()[0])

#note: values given in the article due to a clerical error are slightly different.
