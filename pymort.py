"""
Pymort: mortgage simulator
Author: Alon Diament (alondmnt.com)
Date: January 2019
License: MIT
"""

from collections import Iterable
from copy import deepcopy
from itertools import product
from time import time
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class Plan(object):
    def __init__(self, principal, years, interest, track_inflation=False, description=''):
        self.principal = float(principal)
        self.set_years(years)
        self.set_interest(interest)
        self.track_inflation = bool(track_inflation)
        self.description = str(description)
        self.current_balance = float(principal)

    def __str__(self):
        return '\nPlan{}'.format({k: ('{:.4f}'.format(v)
                                    if type(v) is not str else v)
                                for k, v in self.__dict__.items()})

    def __repr__(self):
        return self.__str__()

    def set_years(self, years):
        self.years = float(years)

    def set_interest(self, interest):
        self.interest = float(interest)  # percentage
        self.monthly_interest = float(interest) / 1200.0  # fraction
        self.monthly_pay = self.calc_monthly()

    def calc_monthly(self):
        d = 1 - (1 / (1 + self.monthly_interest)) ** (12 * self.years)
        return self.principal * self.monthly_interest / d

    def reset(self):
        self.current_balance = self.principal
        self.monthly_pay = self.calc_monthly()

    def pay(self, inflation):
        if self.track_inflation:
            self.monthly_pay *= (1 + inflation)
            self.current_balance *= (1 + inflation)

        interest_pay = self.monthly_interest * self.current_balance
        fund_pay = min(self.monthly_pay - interest_pay, self.current_balance)
        self.current_balance -= fund_pay

        return fund_pay, interest_pay


class Mortgage(object):
    def __init__(self):
        self.inflation = 0.0  # percentage
        self.monthly_inflation = 0.0  # fraction
        self.plans = []

    def __str__(self):
        return '\nMortgage{}'.format(self.__dict__)

    def __repr__(self):
        return self.__str__()

    def add_plan(self, principal, years, interest, track_inflation, description):
        # cleanup
        self.plans = [p for p in self.plans if p.current_balance > 1e-2]
        self.plans.append(Plan(principal, years, interest, track_inflation, description))

    def reset_plans(self):
        for p in self.plans:
            p.reset()

    def set_inflation(self, inflation):
        self.inflation = float(inflation)  # percentage
        # self.monthly_inflation = inflation / 1200.0
        self.monthly_inflation = (1 + float(inflation) / 100) ** (1/12) - 1  # fraction

    def get_principal(self):
        return sum([p.principal for p in self.plans])

    def get_balance(self):
        return sum([p.current_balance for p in self.plans])

    def get_monthly(self):
        return sum([p.monthly_pay for p in self.plans])

    def simulate(self, inflation=None):
        """ taking inflation into account. """
        self.reset_plans()
        if inflation is not None:
            self.set_inflation(inflation)
        total = []
        while self.get_balance() > 1e-2:
            monthly = np.zeros(2)
            for plan in self.plans:
                monthly += plan.pay(self.monthly_inflation)

            total.append(monthly)

        return pd.DataFrame(total, columns=['fund', 'interest'])

    def plot(self, inflation=None):
        payments = self.simulate(inflation)
        payments.plot(kind='bar', stacked=True, width=1.0)
        plt.xticks(np.arange(0, len(payments)+1, 12))
        plt.title('mortgage monthly payment\n(return_ratio={:.2f}, inflate={:.2f}%)'.format(
                  payments.sum().sum()/self.get_principal(), self.inflation))
        plt.xlabel('year')
        plt.ylabel('amount')


class MortgageSim(object):
    def __init__(self):
        self.morts = []

    def add_mortgage(self, morts):
        if not isinstance(morts, Iterable):
            morts = [morts]
        assert all([type(m) is Mortgage for m in morts]), 'mortgage must be a collection of Mortgage objects'
        self.morts += [m for m in morts]
        print('{} offers added to simulator\ntotal offers: {}'.format(len(morts), len(self.morts)))

    def add_mortgage_from_file(self, mort_fn):
        """
        CSV file columns structure:
            [mortgage_id (int), plan_principal (float), plan_year (int),
             plan_interest (float), plan_tracks_inflation (bool)]
        """
        df = pd.read_csv(mort_fn)
        count = 0
        for _, plans in  df.groupby('mortgage_id'):
            count += 1
            m = Mortgage()
            for _, x in plans.iterrows():
                m.add_plan(x['plan_principal'], x['plan_years'],
                        x['plan_interest'], x['plan_tracks_inflation'],
                        x['description'])
            self.morts.append(m)

        print('{} offers added to simulator\ntotal offers: {}'.format(count, len(self.morts)))


class MultiSim(MortgageSim):
    """ takes a defined set of plans and provides statistics from
    multiple simulations in order to select the best one. """
    def __init__(self):
        MortgageSim.__init__(self)

    def simulate(self, inflation=np.arange(1, 3, 0.25),
                 max_pay=None, max_first_pay=None):
        if not isinstance(inflation, Iterable):
            inflation = [inflation]
        M = len(self.morts)
        I = len(inflation)
        first_pay = np.zeros((M, I))
        highest_pay = np.zeros((M, I))
        return_ratio = np.zeros((M, I))

        for m in range(M):
            for i in range(I):
                payments = self.morts[m].simulate(inflation[i]).sum(axis=1)
                first_pay[m, i] = payments.iloc[0]
                highest_pay[m, i] = payments.max()
                return_ratio[m, i] = payments.sum() / self.morts[m].get_principal()

        # filter by constraints
        if max_pay is not None:
            return_ratio[highest_pay > max_pay] = np.nan
        if max_first_pay is not None:
            return_ratio[first_pay > max_first_pay] = np.nan

        return_ratio = pd.DataFrame._from_arrays(return_ratio.T,
                                                 [('inflat_%.2f' % i).replace('.', '_') for i in inflation],
                                                 range(len(self.morts)))
        return_ratio.index.name = 'mortgage'

        highest_pay = pd.DataFrame._from_arrays(highest_pay.T,
                                                [('inflat_%.2f' % i).replace('.', '_') for i in inflation],
                                                range(len(self.morts)))
        highest_pay.index.name = 'mortgage'

        return return_ratio, highest_pay


class LinearSim(MortgageSim):
    """ takes a defined set of plans and tries to build a new plan based on
    a prediction function f_interest(x_years) which uses linear
    interpolation for points (x_years) between the given plans.

    note, that if a mortrage consists of several plans (e.g., 3 components),
    all the given mortgages must have the same number of plans, in the same
    order. """
    def __init__(self):
        MortgageSim.__init__(self)

    def build_model(self, years, global_model=False):
        # build a linear model for different lengths of years
        P = len(self.morts[0].plans)
        self.years = years
        self.plans = []
        for p in range(P):
            plan_train_data = [[m.plans[p].years, m.plans[p].interest]
                               for m in self.morts]
            plan_train_data = np.array(sorted(plan_train_data, key=lambda x:x[0]))
            if global_model:
                # use all points for a single linear model
                # otherwise, inrepolate
                linmodel = LinearRegression()
                linmodel.fit(plan_train_data[:, 0].reshape(-1, 1),
                             plan_train_data[:, 1].reshape(-1, 1))
            plan_set = []
            for y in years:
                plan_set.append(deepcopy(self.morts[0].plans[p]))
                plan_set[-1].set_years(y)  # must be set before interest
                if global_model:  # global linear regression
                    plan_set[-1].set_interest(float(linmodel.predict(y.reshape(-1, 1))))
                else:  # local linear interpolation
                    plan_set[-1].set_interest(np.interp(y, plan_train_data[:, 0],
                                                        plan_train_data[:, 1]))
            self.plans.append(plan_set)
        self.plans = np.array(self.plans)

        self.years2ind = np.zeros(self.years.max() + 1, dtype=int)
        self.years2ind[self.years] = np.arange(len(self.years))

    def get_plan(self, *years):
        """ run build_model() or simulate() first. """
        P = len(self.morts[0].plans)
        assert len(years) == P
        mort = Mortgage()
        mort.plans = self.plans[np.arange(P), self.years2ind[list(years)]].tolist()
        return mort

    def simulate(self, years=np.arange(10, 21), inflation=np.arange(1, 3, 0.25),
                 max_pay=None, max_first_pay=None, global_model=False):
        t0 = time()
        if not isinstance(inflation, Iterable):
            inflation = [inflation]
        if max_pay is None:
            print('max_pay is undefined, will probably converge to a trivial solution')

        P = len(self.morts[0].plans)
        Y = len(years)
        I = len(inflation)
        highest_pay = np.zeros(P*(Y,) + (I,))
        first_pay = np.zeros_like(highest_pay)
        return_ratio = np.zeros_like(highest_pay)

        self.build_model(years, global_model=global_model)

        # evaluate all possible plans for a range of inflation rates
        dummy_mort = deepcopy(self.morts[0])
        year_grid = np.array(P*list(range(Y))).reshape(P, -1).tolist()  # (Y)^P

        for y_vec in product(*year_grid):
            dummy_mort.plans = [self.plans[p][y_vec[p]] for p, y in enumerate(y_vec)]
            for i in range(I):
                payments = dummy_mort.simulate(inflation[i]).sum(axis=1)
                highest_pay[y_vec + (i,)] = payments.max()
                first_pay[y_vec + (i,)] = payments.iloc[0]
                return_ratio[y_vec + (i,)] = payments.sum() / dummy_mort.get_principal()

        # filter by constraints
        if max_pay is not None:
            return_ratio[highest_pay > max_pay] = np.nan
        if max_first_pay is not None:
            return_ratio[first_pay > max_first_pay] = np.nan

        # anounce the winners
        ibest = []
        for i in range(I):
            if np.isnan(return_ratio[..., i]).all():
                ibest.append(P*(np.nan,))
                continue
            ibest.append(np.unravel_index(np.nanargmin(return_ratio[..., i]), P*(Y,)))

        # tidy up
        return_ratio = pd.DataFrame.from_dict({tuple(years[list(i)]): return_ratio[i]
                        for i in ibest if np.isfinite(i).all()}).T
        highest_pay = pd.DataFrame.from_dict({tuple(years[list(i)]): highest_pay[i]
                        for i in ibest if np.isfinite(i).all()}).T

        if not len(return_ratio):
            print('no matching plans found')
            return return_ratio, highest_pay

        return_ratio.index = return_ratio.index.set_names(['plan_%d' % (p+1) for p in range(P)])
        return_ratio.columns = [('inflat_%.2f' % i).replace('.', '_') for i in inflation]

        highest_pay.index =  highest_pay.index.set_names(['plan_%d' % (p+1) for p in range(P)])
        highest_pay.columns = [('inflat_%.2f' % i).replace('.', '_') for i in inflation]

        print('%.1fsec' % (time()-t0))
        return return_ratio, highest_pay
