import numpy as np
from pywaterflood import CRM as _CRM
from utils.CRM_module import CRMP as _CRMP
from utils.CRMIP_module import CRMIP as _CRMIP
class CRM:
    def __init__(self, tau_selection='per-pair', constraints='up-to one', N_inj = None, I_P=False):
        if N_inj is None:
            self.crm = _CRM(tau_selection=tau_selection, constraints=constraints)
            self.crmp = None
            self.crmip = None
        elif I_P:
            self.crm = None
            self.crmp = None

            tau0 = np.ones((N_inj, 1)) /10      # tau guess
            gain0 = np.ones((N_inj, 1)) / N_inj   # uniform allocation
            q00 = np.ones(1)     # initial production
            self.init_guess = [tau0, gain0, q00]
            self.crmip = _CRMIP(self.init_guess, include_press=False)
        else:
            self.crm = None

            tau = np.ones(1)
            gain_mat = np.ones([N_inj, 1])
            gain_mat = gain_mat / np.sum(gain_mat, axis=0, keepdims=True)
            qp0 = np.ones(1)
            J = np.ones(1) / 10
            self.init_guess = [tau, gain_mat, qp0, J]
            self.crmp = _CRMP(self.init_guess, include_press=False)

            self.crmip = None

    def fit(self, production=None, injection=None, time=None):
        if self.crm:
            return self.crm.fit(production, injection, time)
        if self.crmp:
            return self.crmp.fit_model([time, injection], production, self.init_guess)
        if self.crmip:
            return self.crmip.fit_model([time, injection], production, self.init_guess)
        raise Exception("CRM instance is not initialized")

    def predict(self, injection=None, time=None):
        if self.crm:
            if time is None and injection is None:
                return self.crm.predict()
            return self.crm.predict(injection=injection, time=time)
        if self.crmp:
            return self.crmp.prod_pred([time, injection])
        if self.crmip:
            return self.crmip.prod_pred([time, injection])
            
