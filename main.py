"""
safeopt with uncertain contexts on the OpenAI Gym Mountain Car
"""
import GPy
import numpy as np 
import safeopt  
import gym 
from sklearn.metrics.pairwise import rbf_kernel as rbf
import copy

class safe_opt:
    def __init__(self, initial_param=np.array([[0.3]]), noise_var=1, bounds_param=[(-1, 1), [-1, 1]], threshold=1e-4, num_contexts=1, disc_contexts=2, lambd=1e-4, delta=0.05, bound_rkhs=1):
        self.param, self.noise_var, self.bounds_param, self.threshold, self.num_contexts, self.disc_contexts, self.lambd, self.delta, self.bound_rkhs = initial_param, noise_var, bounds_param, threshold, num_contexts, disc_contexts, lambd, delta, bound_rkhs
        # Define Kernel
        # works on the first column of X, index=0
        k_parameters = GPy.kern.Matern32(input_dim=1, variance=1, lengthscale=0.1, ARD=True, active_dims=[0])
        # works on the second column of X, index=1
        if self.num_contexts > 0:
            k_context = GPy.kern.RBF(input_dim=num_contexts, variance=2., lengthscale=0.5, active_dims=[1], name='context')
        if self.num_contexts > 0:
            self.kernel = k_parameters * k_context
        else:
            self.kernel = k_parameters
        self.k_opt = copy.deepcopy(initial_param)
        self.k_scale = 20
        self.parameter_set = safeopt.linearly_spaced_combinations([self.bounds_param[0]], 2000)
        self.context = np.zeros(num_contexts).reshape(1, -1)
        self.opt_done = False
        self.systems = [gym.make('MountainCarContinuous-v0'), gym.make('MountainCarContinuous-v0')]
        self.systems[1].power = 0.1

    def power_func(self, meas, n, k, K):
        arg = rbf(meas, meas) - k@np.linalg.solve(K+n*self.lambd*np.eye(K.shape[0]), k.T)
        return np.sqrt(arg)

    def context_conf_int(self, meas, gamma=2):
        try:
            K = rbf(self.meas, self.meas, gamma=gamma)
            k = rbf(meas, self.meas, gamma=gamma)
            prob = self.meas_cont@np.linalg.solve(K + self.meas.shape[0]*self.lambd*np.eye(self.meas.shape[0]), k.T)
            cg = 1/4*np.sqrt(np.log(np.linalg.det(K + np.max([1, self.meas.shape[0]*self.lambd])*np.eye(self.meas.shape[0]))) - 2*np.log(self.delta))
            epsilon = self.power_func(meas, self.meas.shape[0], k, K)*(np.sqrt(self.bound_rkhs) + cg/np.sqrt(self.meas.shape[0]*self.lambd))
            val = np.max(prob)
            prob_cont = np.where(prob == np.amax(prob))[0][0]
            return prob_cont, np.max([0, (val - epsilon)[0, 0]])
        except:
            return 0, 0

    def update_context_conf_int(self, meas, cont):
        try:
            self.meas = np.vstack((self.meas, meas))
            self.meas_cont = np.hstack((self.meas_cont, np.zeros(self.meas_cont.shape[0]).reshape(-1, 1)))
            self.meas_cont[int(cont), -1] = 1
        except:
            self.meas = meas
            self.meas_cont = np.zeros((self.disc_contexts, 1))
            self.meas_cont[int(cont), -1] = 1

    def sample_env(self, min_temp=-6, max_temp=7):
        T = np.random.uniform(0, 1)
        T_scaled = (max_temp - min_temp)*T + min_temp
        prob = 1/(1 + np.exp(-T_scaled + 1))
        if np.random.random() > prob:
            self.sys = self.systems[1]
        else:
            self.sys = self.systems[0]
        return np.array([[T]])

    def setup_optimization(self):
        self.sample_env()
        # The statistical model of our objective function
        if self.num_contexts > 0:
            param = np.hstack([self.param.reshape(1, -1), self.context])
        else:
            param = self.param.reshape(1, -1)
        self.gp = GPy.models.GPRegression(param.reshape(1,-1), self.obj_fun(param.reshape(1,-1))[:,0,None], self.kernel, noise_var=self.noise_var)
        self.opt = safeopt.SafeOpt(self.gp, self.parameter_set, [-np.inf, 0], num_contexts=self.num_contexts, threshold=self.threshold)
        self.opt.context = self.context
        perf_init = [None]*self.disc_contexts
        for cont in range(self.disc_contexts):
            self.sys = self.systems[cont]
            param = np.hstack((self.param.reshape(1, -1), np.array([[cont]])))
            y_meas = self.obj_fun(param).reshape(1, -1)
            perf_init[cont] = y_meas[0, 0]
            if self.num_contexts > 0:
                self.opt.add_new_data_point(self.param.reshape(1, -1), y_meas, context=np.array([[cont]]))
            else:
                self.opt.add_new_data_point(self.param.reshape(1, -1), y_meas)
        return perf_init


    def optimization(self, num_it=20):
        cont_identified = 0 
        cont_identified_wrong = 0
        for idx in range(num_it):
            # To reduce computation time, only run SafeOpt for first 1000 iterations
            if idx > 1000:
                self.opt_done = True
            meas = self.sample_env()
            if self.num_contexts > 0:
                prob_cont, prob = self.context_conf_int(meas)
                if prob > 0.8:
                    context = prob_cont
                    cont_identified += 1
                    print('identified context')
                    print('confidence')
                    print(prob)
                    if context != self.context:
                        cont_identified_wrong += 1
                        print('wrong context')
                    else:
                        print('correct context')
                else:
                    context = self.context
                    self.update_context_conf_int(meas, self.context)
            if not self.opt_done:
                if self.num_contexts > 0:
                    x_next = self.opt.optimize(context) 
                else:
                    x_next = self.opt.optimize()
                if self.num_contexts > 0:
                    y_meas = self.obj_fun(np.hstack((x_next.reshape(1, -1), self.context)).reshape(1, -1))
                    self.opt.add_new_data_point(x_next, y_meas, context=context)
                else:
                    y_meas = self.obj_fun(x_next.reshape(1, -1))
                    self.opt.add_new_data_point(x_next, y_meas)

        if self.num_contexts > 0:
            with open('results.txt', 'w') as file:
                file.write('identified contexts')
                file.write('\n')
                file.write(str(cont_identified))
                file.write('\n')
                file.write('wrongly identified contexts')
                file.write('\n')
                file.write(str(cont_identified_wrong))
                file.write('\n')

    def obj_fun(self, param):
        F = self.k_scale*param[0, 0]
        state = self.sys.reset()
        action = 1
        reward = 100
        for i in range(100):
            vel = state[1]
            if not np.isscalar(vel):
                vel = vel[0]
            action = F*vel
            state, rew, _, _ = self.sys.step(np.array([[action]]))
            reward += rew
        return np.array([[reward]])

def safeopt_context(num_it=10000):
    safe = safe_opt(num_contexts=1)
    safe.setup_optimization()
    safe.optimization(num_it=num_it)

if __name__ == '__main__':
    safeopt_context()