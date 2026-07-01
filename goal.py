"""
© 2021 This work is licensed under a CC-BY-NC-SA license.
Title:
**Authors:** Cristiano Capone
"""

import numpy as np
import utils as ut
from tqdm import trange

#import _pickle as cPickle
import traceback


def sigm ( x, dv ):

	if dv < 1 / 30:
		return x > 0;
	y = x / dv;
    #y = x / 10.;

	out = 1.5*(1. / (1. + np.exp (-y*3. )) - .5)

	return out;

def gaussian(x, mu, sig):
	return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

class GOAL_GRADED:

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape']
        net_shape = (self.N, self.T)

        self.dt = par['dt']#1. / self.T;
        self.itau_m = self.dt / par['tau_m']
        self.tau_m = par['tau_m']
        self.tau_m = np.logspace( np.log(par['tau_m']*.5) ,np.log(par['tau_m']*2.),self.N)
        self.num_g = par['num_g']

        self.itau_s = np.exp (-self.dt / par['tau_s'])
        self.itau_ro = np.exp (-self.dt / par['tau_ro'])

        self.dv = par['dv']

        # This is the network connectivity matrix
        self.J = np.random.normal (0., par['sigma_rec'], size = (self.N, self.N));#np.zeros ((self.N, self.N));

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I));
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O));
        self.Jout = np.random.normal (0.0, par['sigma_output'], size = (self.O,self.N));#np.zeros ((self.O, self.N));
        self.Jout_g = np.random.normal (0.0, par['sigma_output'], size = (self.num_g,self.N));
        self.JoutV = np.random.normal (0.0, par['sigma_output'], size = (1,self.N));#np.zeros ((self.O, self.N));
        self.Jsigmaout = np.random.normal (0., par['sigma_output'], size = (self.O,self.N))#np.zeros ((self.O, self.N));

        print('hello')

        self.dJfilt = np.zeros(np.shape(self.J))
        self.dJfilt_out = np.zeros(np.shape(self.Jout))
        self.dJfilt_out_g = np.zeros(np.shape(self.Jout_g))
        
        self.dJfilt_sigma_out = np.zeros(np.shape(self.Jsigmaout))

        self.dJoutV_filt = np.zeros(np.shape(self.JoutV))
        self.dJoutV_aggregate = np.zeros(np.shape(self.JoutV))


        self.dJout_aggregate = np.zeros(np.shape(self.Jout))
        self.dJout_g_aggregate = np.zeros(np.shape(self.Jout_g))
        self.dJout_sigma_aggregate = np.zeros(np.shape(self.Jsigmaout))
        self.dJ_aggregate = np.zeros(np.shape(self.J))
        self.dh_aggregate = np.zeros(self.N,)

        self.S_filt_tot = np.zeros(self.N,)

        self.value=0
        self.r=0
        self.r_old=0

        # Remove self-connections
        np.fill_diagonal (self.J, 0.);

        self.name = 'model'

        # Impose reset after spike
        self.s_inh = -par['s_inh'];
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh)

        # This is the external field
        h = par['h'];

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h;

        # Membrane potential
        self.H = np.ones (self.N) * par['Vo'];

        self.Vo = par['Vo'];

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N);
    
        self.S_hat = np.zeros (self.N);
        self.dH = np.zeros (self.N);

        # This is the single-time output buffer
        self.state_out = np.zeros (self.N)
        self.state_out_p = np.zeros (self.N);

        # Here we save the params dictionary
        self.par = par

    def step_ (self, inp, if_spk = False,if_tanh = False):
        itau_m = self.dt/self.tau_m
        exp_tau_s = np.exp(-self.dt/self.tau_m)
        
        if if_spk:
        
            self.S_hat   = self.S_hat   * itau_s + self.S   * (1. - itau_s)

        else:
            self.S_hat   = self.S_hat   * exp_tau_s + self.S   * (1. - exp_tau_s)#np.copy(self.S)#) (self.S_hat   * itau_s + self.S   * (1. - itau_s))

        
        self.dH   = self.dH  * np.exp( - itau_m) + np.tanh( (1.-np.exp( -itau_m)) * self.S_hat ) 
        
                                                          #+ self.Jreset @ self.S  ;
        if if_spk:
            self.H   = self.H   *  np.exp( - itau_m) + (1.-np.exp( -itau_m)) * (self.J @ self.S_hat   + self.Jin @ inp + self.h)\
                                                          + self.Jreset @ self.S
            self.S   = self._sigm (self.H  , dv = self.dv) - 0.5 > 0.
        else:
            if if_tanh:
                self.H   = self.H   * np.exp( - itau_m) + (1.-np.exp( -itau_m)) *  np.tanh(self.J @ self.S   + self.Jin @ inp)#\
            else:
                self.H   = self.H   * np.exp( - itau_m) + (1.-np.exp( -itau_m)) * (self.J @ self.S   + self.Jin @ inp + self.h)
                                                        

     
            self.S   = np.copy(self.H)#self._sigm (self.H  , dv = self.dv)# - 0.5 > 0.
        self.y = self.Jout@self.S        
        self.sigma = np.sqrt( np.clip(self.Jsigmaout @self.S ,0.05**2,0.5**2))


    def step_g (self, inp, if_spk = False,if_tanh = False):
        itau_m = self.dt/self.tau_m
        exp_tau_s = np.exp(-self.dt/self.tau_m)
        
        if if_spk:
            self.S_hat   = self.S_hat   * itau_s + self.S   * (1. - itau_s)

        else:
            self.S_hat   = self.S_hat   * exp_tau_s + self.S   * (1. - exp_tau_s)#np.copy(self.S)#) (self.S_hat   * itau_s + self.S   * (1. - itau_s))
        
        self.dH   = self.dH  * np.exp( - itau_m) + np.tanh( (1.-np.exp( -itau_m)) * self.S_hat ) 
        
                                                          #+ self.Jreset @ self.S  ;
        if if_spk:
            self.H   = self.H   *  np.exp( - itau_m) + (1.-np.exp( -itau_m)) * (self.J @ self.S_hat   + self.Jin @ inp + self.h)\
                                                          + self.Jreset @ self.S
            self.S   = self._sigm (self.H  , dv = self.dv) - 0.5 > 0.
        else:
            if if_tanh:
                self.H   = self.H   * np.exp( - itau_m) + (1.-np.exp( -itau_m)) *  np.tanh(self.J @ self.S   + self.Jin @ inp)#\
            else:
                self.H   = self.H   * np.exp( - itau_m) + (1.-np.exp( -itau_m)) * (self.J @ self.S   + self.Jin @ inp + self.h)
                
            self.S   = np.copy(self.H)#self._sigm (self.H  , dv = self.dv)# - 0.5 > 0.
            
        self.y = self.Jout@self.S
        self.y_g = self.Jout_g@self.S_hat        
        self.sigma = np.sqrt( np.clip(self.Jsigmaout @self.S ,0.05**2,0.5**2))

    def value_step (self, r):
        self.r_old = self.r
        self.r = r
        self.a_old = self.r_old - self.value_old + self.gamma*self.value
        self.value_old = self.value
        self.value = self.JoutV @ self.state_out

    def policy_ (self,if_derive_var=False):        
        self.mu = np.tanh(self.y)
        if if_derive_var:
            self.act_variance = np.copy(self.sigma)
        self.action = self.mu + np.random.normal (0., self.act_variance, size = ( self.O,))
        
    def policy_hier (self):
            self.g[0] = (np.tanh(self.y_g)+1)/2
            self.g[1] = 1.-self.g[0]
            self.mu = np.tanh(self.y[0:self.out_dim]*self.g[0] + self.y[self.out_dim:2*self.out_dim]*self.g[1])
            #self.mu = np.tanh(self.y[0:self.out_dim]*0. + self.y[self.out_dim:2*self.out_dim]*1.)            
            #self.mu = np.tanh(self.y[0:self.out_dim])# + self.y[self.out_dim:2*self.out_dim]*self.g[1])
            self.musub1 = np.tanh(self.y[0:self.out_dim])
            self.musub2 = np.tanh(self.y[self.out_dim:2*self.out_dim])
            self.action = self.mu + np.random.normal (0., self.act_variance, size = ( self.out_dim,))
            #self.action = np.copy(np.clip(self.action, -1., 1.))
            #return  mu + np.random.normal (0., self.act_variance, size = ( self.O,)), mu

    def policy_hier_CPG (self,time):
            
            self.g[0] = (np.tanh(self.y[-1])+1)/2#(np.tanh(  2*np.cos(time * 2 * np.pi/16 )   )+1)/2
            self.g[1] = 1.-self.g[0]

            self.dg[0] =  (np.tanh(  2*np.cos(time * 2 * np.pi/16 )   )+1)/2 - self.g[0]
            self.dg[1] =  np.copy(self.dg[0])#self.g[1] - ( 1 - (np.tanh(self.y[-1])+1)/2

            self.mu = np.tanh(self.y[0:self.out_dim]*self.g[0] + self.y[self.out_dim:2*self.out_dim]*self.g[1])
            self.musub1 = np.tanh(self.y[0:self.out_dim])
            self.musub2 = np.tanh(self.y[self.out_dim:2*self.out_dim])
            self.action = self.mu + np.random.normal (0., self.act_variance, size = ( self.out_dim,))
    
    def policy_multi_hier (self):
            self.g = self.y[-self.num_g:]
            self.g = self.g*3.#self.y_g*3.
            self.g = np.exp(self.g) / np.sum(np.exp(self.g))
            out=0.
            for gi in range(self.num_g):
                out = out + self.y[self.out_dim*gi :self.out_dim*gi + self.out_dim]*self.g[gi]
            self.mu = np.tanh(out)
            self.action = self.mu + np.random.normal (0., self.act_variance, size = ( self.out_dim,))

    def reset (self, init = None):

        self.S   = init if init else np.zeros (self.N)
        self.S_hat   = self.S   * self.itau_s if init else np.zeros (self.N)
        self.state_out  *= 0
        self.state_out_p  *= 0
        self.H  *= 0.
        self.H  += self.Vo
        self.dJfilt *= 0
        self.dJfilt_out *= 0
        self.dJfilt_out_g *= 0
        
        self.value=0
        self.value_old=0

        self.dJoutV_filt *= 0
        self.dJoutV_aggregate *= 0

        self.dJfilt_sigma_out *= 0#self.dJfilt_sigma_out*(1-alpha_J) + dJ_sigma_out*alpha_J
        #self.dJ_aggregate *= 0#r*self.dJfilt
        self.dJout_aggregate *= 0
        #self.dJout_sigma_aggregate *= 0#r*self.dJfilt_sigma_out

    def learn_error (self, r, learn_rec=False):

        alpha_J = 1-self.gamma#.002#0.002
        if learn_rec:
            f_avg_target=0.1
            dJ = np.outer (self.Jout.T@(self.action - self.mu)*self._dsigm (self.H, dv = 1.), self.dH)
            self.dh = (f_avg_target - self.S)

        dmu = self.action - self.mu
        dJ_out =  np.outer( dmu*(1.-np.tanh(self.y)**2) , self.S.T)#/self.sigma**2
        dJ_sigma_out =  np.outer( ( (dmu)**2 -  self.sigma**2 ) , self.state_out.T)#/self.sigma**3

        if learn_rec:
            self.dJfilt = self.dJfilt*(1-alpha_J) + dJ*alpha_J
        self.dJfilt_out = self.dJfilt_out*(1-alpha_J) + dJ_out*alpha_J
        self.dJfilt_sigma_out = self.dJfilt_sigma_out*(1-alpha_J) + dJ_sigma_out*alpha_J

        if learn_rec:
            self.dJ_aggregate += r*self.dJfilt
            self.dh_aggregate += self.dh
            self.S_filt_tot += self.dH
        self.dJout_aggregate += r*self.dJfilt_out
        
        #print(dJ_out)
        self.dJout_sigma_aggregate += r*self.dJfilt_sigma_out

    def learn_w_sample (self, r, learn_rec=False):

        alpha_J = 1-self.gamma#.002#0.002
        self.dJfilt_out = self.dJfilt_out*(1-alpha_J) + self.dJ_out*alpha_J
        self.dJout_aggregate += r*self.dJfilt_out
        self.dJout_sigma_aggregate += r*self.dJfilt_sigma_out

    

    def update_J (self,learn_rec=False,l2_out=0.001):
        #print(learn_rec)
        if learn_rec:
            #print('here')
            self.dJ_aggregate = self.dJ_aggregate +  0.001*np.outer (self.dh*self._dsigm (self.H, dv = 1.), self.S_filt_tot )
            self.J = self.adam_rec.step (self.J, self.dJ_aggregate)# - 0.001*self.J)#
            #self.h = self.h + 0.00001* self.dh_aggregate
            self.h = np.minimum(self.h,-1.)
            self.Vo = np.copy(self.h)# + 0.000001* self.dh_aggregate
            
        #self.J = self.J + self.par["alpha"]*self.dJ_aggregate*0
        np.fill_diagonal (self.J, 0.)

        self.Jout = self.adam_out.step (self.Jout, self.dJout_aggregate - l2_out*self.Jout )
        #self.Jout_g = self.adam_out_g.step (self.Jout_g, self.dJout_g_aggregate  )
        #self.Jsigmaout = self.adam_sigma_out.step (self.Jsigmaout, self.dJout_sigma_aggregate)

        self.dJ_aggregate=0
        self.dh_aggregate=0
        self.dJout_aggregate=0
        self.dJout_g_aggregate=0
        self.dJout_sigma_aggregate=0
        self.S_filt_tot=0


    def learn_error_hpg (self, r ,   g_coeff):
            
            alpha_J = 1-self.gamma#.002#0.002
            dmu = (self.action - self.mu)*(1.-self.mu**2)

            self.dJout_aggregate += r*self.dJfilt_out
            
            dJ_out = np.zeros((self.O,self.N))
            #dJ_ent_out = np.zeros((self.O,self.N))

            #dJ_ent_out[0:3,:] = - g[0]*np.outer (  self.prob*np.log(self.prob ) + self.entropy , self.state_out.T)
            #dJ_ent_out[3:6,:] = - g[1]*np.outer (  self.prob*np.log(self.prob ) + self.entropy , self.state_out.T)

            out_dim=self.out_dim
            dJ_out[0:out_dim,:] =  self.g[0]*np.outer(dmu, self.S.T)#+dJ_ent_out_1#np.outer (self.Jout.T@(ac_vector - self.out)*self._dsigm (self.H, dv = 1.), self.dH)
            dJ_out[out_dim:out_dim*2,:] =  self.g[1]*np.outer(dmu, self.S.T)#+dJ_ent_out_2#np.outer (self.Jout.T@(ac_vector - self.out)*self._dsigm (self.H, dv = 1.), self.dH)
            dJ_out[out_dim*2,:] =  np.sum(np.outer(dmu*(self.y[0:out_dim]-self.y[out_dim:2*out_dim]), self.S_hat.T) *g_coeff,axis=0)#*(1-np.tanh(self.y[-1])**2)#(2*self.g[0]-1.)*
            
            #if g_coeff==0:
            #    self.Jout[out_dim*2,:] = np.copy( self.Jout[out_dim*2,:] * 0)
            self.dJfilt_out = self.dJfilt_out*(1-alpha_J) + dJ_out
            self.grad = np.copy(r*self.dJfilt_out)
            self.dJout_aggregate += (r*self.dJfilt_out) #+ ch*dJ_ent_out

            self.dh = 0#(f_avg_target - self.S)

    def learn_error_hpg_target (self, r ,   g_coeff):
            
            alpha_J = 1-self.gamma#.002#0.002
           

            dmu = (self.action - self.mu)*(1.-self.mu**2)

            self.dJout_aggregate += r*self.dJfilt_out
            
            dJ_out = np.zeros((self.O,self.N))

            out_dim=self.out_dim
            dJ_out[0:out_dim,:] =  self.g[0]*np.outer(dmu, self.S.T)#+dJ_ent_out_1#np.outer (self.Jout.T@(ac_vector - self.out)*self._dsigm (self.H, dv = 1.), self.dH)
            dJ_out[out_dim:out_dim*2,:] =  self.g[1]*np.outer(dmu, self.S.T)#+dJ_ent_out_2#np.outer (self.Jout.T@(ac_vector - self.out)*self._dsigm (self.H, dv = 1.), self.dH)
            dJ_out[out_dim*2,:] =  np.sum(np.outer( self.dg , self.S.T ) *g_coeff,axis=0)#*(1-np.tanh(self.y[-1])**2)#(2*self.g[0]-1.)*
          
            self.dJfilt_out = self.dJfilt_out*(1-alpha_J) + dJ_out
            self.grad = np.copy(r*self.dJfilt_out)
            self.dJout_aggregate += (r*self.dJfilt_out) #+ ch*dJ_ent_out

            self.dh = 0#(f_avg_target - self.S)

    def learn_error_ppo (self, r):

        self.dh=0.

        alpha_J = 1-self.gamma#.002#0.002
        eps = self.eps#0.1
        #print(np.shape(self.action - self.mu))
        #print(self.Jout.T@(self.action - self.mu))
        dmu = self.action_old - self.mu_old

        p_new = gaussian(self.action_old,self.mu,self.act_variance)
        p_old = gaussian(self.action_old,self.mu_old,self.act_variance)
        p_frac = p_new/p_old
        #print(p_frac)
        #rho = np.maximum(p_frac[action], 1-eps )
        #rho = np.minimum(rho, 1+eps )
        #rho = p_frac
        #print(p_frac)
        #rho = np.clip(rho,1-eps,1+eps)

        #self.p_filt = self.p_filt*alpha_J + rho
        ppo_mask = np.ones(self.O,)

        ppo_mask[p_frac > 1+eps] = 0
        ppo_mask[p_frac < 1-eps] = 0

        #dJ = np.outer (self.Jout.T@(dmu*p_frac*ppo_mask) * self._dsigm (self.H, dv = 1.), self.dH)
        dJ_out =  np.outer( dmu*p_frac*ppo_mask , self.S.T)#/self.sigma**2  * (p_frac)*ppo_mask
        #dJ_sigma_out =  np.outer( ( (dmu)**2 -  self.sigma**2 ) , self.state_out.T)#/self.sigma**3

        #self.dJfilt = self.dJfilt*(1-alpha_J) + dJ*alpha_J
        self.dJfilt_out = self.dJfilt_out*(1-alpha_J) + dJ_out*alpha_J
        #self.dJfilt_sigma_out = self.dJfilt_sigma_out*(1-alpha_J) + dJ_sigma_out*alpha_J

        #self.dJ_aggregate += r*self.dJfilt
        self.dJout_aggregate += r*self.dJfilt_out
        #self.dJout_sigma_aggregate += r*self.dJfilt_sigma_out

    def learn_error_multi_hpg (self, r ,   g_coeff):
                
        alpha_J = 1-self.gamma#.002#0.002

        dmu = (self.action - self.mu)*(1.-self.mu**2)
        
        out_dim=self.out_dim

        dJ_out = np.zeros((self.O,self.N))
        dJ_out_g = np.zeros((self.num_g,self.N))

        for gi in range(self.num_g):
            dJ_out[out_dim*gi :out_dim*gi + out_dim,:] =  self.g[gi]*np.outer(dmu, self.S.T)#+dJ_ent_out_1#np.outer (self.Jout.T@(ac_vector - self.out)*self._dsigm (self.H, dv = 1.), self.dH)
            dJ_out[ out_dim *self.num_g + gi ,:] =  np.sum(np.outer(dmu*(self.y[out_dim*gi :out_dim*gi + out_dim]), self.S_hat.T* self.g[gi]*(1.-self.g[gi])  ) *g_coeff,axis=0)#*(1-np.tanh(self.y[-1])**2)

        #dJ_out[0:out_dim,:] =  self.g[0]*np.outer(dmu, self.S.T)#+dJ_ent_out_1#np.outer (self.Jout.T@(ac_vector - self.out)*self._dsigm (self.H, dv = 1.), self.dH)
        #dJ_out[out_dim:out_dim*2,:] =  self.g[1]*np.outer(dmu, self.S.T)#+dJ_ent_out_2#np.outer (self.Jout.T@(ac_vector - self.out)*self._dsigm (self.H, dv = 1.), self.dH)
        #dJ_out[out_dim*2,:] =  np.sum(np.outer(dmu*(self.y[0:out_dim]-self.y[out_dim:2*out_dim]), self.S.T) *g_coeff,axis=0)#*(1-np.tanh(self.y[-1])**2)#(2*self.g[0]-1.)*
        
        if g_coeff==0:
            self.Jout[out_dim*self.num_g,:] = np.copy( self.Jout[out_dim*self.num_g,:] * 0)

        self.dJfilt_out = self.dJfilt_out*(1-alpha_J) + dJ_out
        self.dJfilt_out_g = self.dJfilt_out_g*(1-alpha_J) + dJ_out_g

        
        self.dJout_aggregate += (r*self.dJfilt_out) #+ ch*dJ_ent_out
        self.dJout_g_aggregate += (r*self.dJfilt_out_g) #+ ch*dJ_ent_out

        self.dh = 0#(f_avg_target - self.S)

    def learn_error_multi_hpg_wsample (self, r ,   g_coeff):
                
        alpha_J = 1-self.gamma#.002#0.002
        #dJ = np.outer (self.Jout.T@(self.action - self.mu)*self._dsigm (self.H, dv = 1.), self.dH)
        #print(np.shape(self.action - self.mu))
        #print(self.Jout.T@(self.action - self.mu))

        #dJ_ent_out = np.zeros((self.O,self.N))

        #dJ_ent_out[0:3,:] = - g[0]*np.outer (  self.prob*np.log(self.prob ) + self.entropy , self.state_out.T)
        #dJ_ent_out[3:6,:] = - g[1]*np.outer (  self.prob*np.log(self.prob ) + self.entropy , self.state_out.T)


        self.dJfilt_out = self.dJfilt_out*(1-alpha_J) + self.dJ_out
        self.grad = np.copy(r*self.dJfilt_out)
        self.dJout_aggregate += (r*self.dJfilt_out) #+ ch*dJ_ent_out

        self.dh = 0#(f_avg_target - self.S)
