from pywaterflood import CRM as _CRM
class CRM:
    def __init__(self, tau_selection='per-pair', constraints='up-to one'):
        self.crm = _CRM(tau_selection=tau_selection, constraints=constraints)
    def fit(self, production=None, injection=None, time=None):
        return self.crm.fit(production, injection, time)
    def predict(self, injection=None, time=None):
        if time is None and injection is None:
            return self.crm.predict()
        return self.crm.predict(injection=injection, time=time)
