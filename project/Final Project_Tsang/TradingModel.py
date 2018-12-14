# utf - 8
# Author: Lishuo Pan Data: Dec 12 2018
# load packages
# import pandas as pd
import numpy as np
# from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import resample

# import GPy
# construct packages
class TradingModel:
    def __init__(self, X_tr, y_bid, y_ask, money):
        self.y_tr_bid = y_bid
        self.y_tr_ask = y_ask
        self.X_tr = X_tr
        self.init_money = money
        self.money = money
        self.train_size = 200
        self.pred_size = 20
        self.split = 3
        self.safe_lock = 0
        self.stock = list()
        self.short_sell = list()
        self.inventory_Max = 50
        self.hold_max = np.inf
        self.resampling = 2
        self.rng = 0
        self.cautious_stock = 0.999
        self.cautious_short_sell = 1
        print("Money at the Begining of the day:", self.money)

    ##########################################################
    # Regression Part
    ##########################################################

    def WLS_regression(self, X_tr, y_tr_bid, y_tr_ask, count_down=0.9, p=3):
        # weight matrix generation
        def weight_matrix(dim,count_down):
            G = np.eye(dim)
            for i in np.arange(dim):
                G[i,i] = np.power(count_down, dim-(i+1))
            return G

        G = weight_matrix(self.train_size, count_down)
        # because G is diag matrix, so this is valid
        G_half = np.sqrt(G)
        weighted_X_tr = np.dot(G_half,X_tr)
        weighted_y_bid = np.dot(G_half,y_tr_bid)
        weighted_y_ask = np.dot(G_half, y_tr_ask)

        # model construction
        model_bid = Pipeline([('poly', PolynomialFeatures(degree=p)),
                              ('linear', LinearRegression(fit_intercept=True))])
        model_ask = Pipeline([('poly', PolynomialFeatures(degree=p)),
                              ('linear', LinearRegression(fit_intercept=True))])
        model_bid.fit(weighted_X_tr, weighted_y_bid)
        model_ask.fit(weighted_X_tr, weighted_y_ask)

        return model_bid, model_ask
    def AdaBoostReg_fit(self, X_tr, y_tr):
        Ada_reg_model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2,random_state=self.rng),
                                            n_estimators=300,random_state=self.rng,loss='square')
        Ada_reg_model.fit(X_tr, y_tr)
        return Ada_reg_model
    def GradientBoosting(self, X_tr, y_tr):
        GB_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                             max_depth = 1, random_state = self.rng, loss = 'square')
        GB_model.fit(X_tr, y_tr)
        return GB_model
    ##########################################################
    # Model behavior Part
    ##########################################################
    def FindHL(self, pred):
        High = np.max(pred)
        Low = np.min(pred)
        return High, Low
    def buy_stock(self,Ask_t, inven):
        if len(self.stock) < self.inventory_Max:
            price = Ask_t
            share = inven/Ask_t
            hold_time = 0
            self.stock.append([price, share, hold_time])
            self.money -= inven
        else:
            pass

    def buy_short_sell(self,Bid_t, inven):
        if len(self.short_sell) < self.inventory_Max:
            price = Bid_t
            share = inven/Bid_t
            hold_time = 0
            self.short_sell.append([price, share, hold_time])
            self.money += inven
        else:
            pass

    def sell_stock(self,y_bid):
        for index, eval in enumerate(self.stock):
            price = eval[0]
            share = eval[1]
            eval[2] += 1
            earn = y_bid - price
            # exercise once have profit
            if earn > 0:
                self.money += y_bid * share
                self.stock.pop(index)

            elif eval[2] > self.hold_max:
                self.money += y_bid * share
                self.stock.pop(index)

    def promise_short_sell(self,y_ask):
        for index, eval in enumerate(self.short_sell):
            price = eval[0]
            share = eval[1]
            eval[2] += 1
            earn = price - y_ask
            # exercise once have profit
            if earn > 0:
                self.money -= y_ask*share
                self.short_sell.pop(index)

            elif eval[2] > self.hold_max:
                self.money -= y_ask * share
                self.short_sell.pop(index)

    def Train_Matrix_t(self,t,X_te,y_bid,y_ask):
        if t+1 >= self.train_size:
            start_tr = t+1-self.train_size
            end_tr = t+1
            index = np.arange(start_tr, end_tr)
            return X_te[index,:], y_bid[index], y_ask[index]
        else:
            n = self.X_tr.shape[0]
            start_tr_1 = t+1-self.train_size
            end_tr_1 = 0
            start_tr_2 = 0
            end_tr_2 = t + 1
            index_1 = np.arange(start_tr_1, end_tr_1)
            index_2 = np.arange(start_tr_2, end_tr_2)
            X_tr_part1 = self.X_tr[index_1, :]
            y_tr_bid_part1 = self.y_tr_bid[index_1]
            y_tr_ask_part1 = self.y_tr_ask[index_1]
            X_tr_part2 = X_te[index_2, :]
            y_tr_bid_part2 = y_bid[index_2]
            y_tr_ask_part2 = y_ask[index_2]
            X_tr = np.vstack((X_tr_part1,X_tr_part2))
            y_tr_bid = np.hstack((y_tr_bid_part1, y_tr_bid_part2))
            y_tr_ask = np.hstack((y_tr_ask_part1, y_tr_ask_part2))
            return X_tr, y_tr_bid, y_tr_ask
    def Test_Matrix_t(self, t, X_te):
        n = len(X_te)
        start = t+1
        if (start+self.pred_size) >= n:
            end = n
        else:
            end = start+self.pred_size
        index = np.arange(start,end)
        return X_te[index,:], index


    def settle_accounts(self,y_bid_end,y_ask_end):
        stock_share = 0
        short_sell_share = 0

        # account the shares in stock and short sell
        for unit in self.stock:
            stock_share += unit[1]
        for unit in self.short_sell:
            short_sell_share += unit[1]
        own_share = stock_share - short_sell_share
        if own_share >= 0:
            self.money += own_share * y_bid_end
        else:
            self.money += own_share * y_ask_end

    def process(self,X_te,y_te_bid,y_te_ask):
        tol_t = len(X_te)
        # For each processing time
        for t in range(tol_t):
            [Bid_t, Ask_t] = [y_te_bid[t],y_te_ask[t]]
            # At the end of the day
            if t == tol_t - 1:
                self.settle_accounts(Bid_t,Ask_t)
                print("At the end of the day:", self.money)


            # In the process
            # When buy in short-sell or stock
            # append a [price, share] to list
            else:
                # prediction
                [X_tr_t, y_tr_bid, y_tr_ask] = \
                    self.Train_Matrix_t(t,X_te,y_te_bid,y_te_ask)

                # resmapling
                sum_pred_bid = 0
                sum_pred_ask = 0
                for re in range(self.resampling):
                    [X_tr_t_B, y_tr_bid_B, y_tr_ask_B] = resample(X_tr_t, y_tr_bid, y_tr_ask,random_state=self.rng)


                    # Fit AdaBoost regression model
                    AdaReg_bid_fit = self.AdaBoostReg_fit(X_tr_t_B, y_tr_bid_B)
                    AdaReg_ask_fit = self.AdaBoostReg_fit(X_tr_t_B, y_tr_ask_B)

                    # AdaReg_bid_fit = self.GradientBoosting(X_tr_t, y_tr_bid)
                    # AdaReg_ask_fit = self.GradientBoosting(X_tr_t, y_tr_ask)

                    # [AdaReg_bid_fit, AdaReg_ask_fit] = \
                    #     self.WLS_regression(X_tr_t, y_tr_bid, y_tr_ask,count_down=1, p = 2)


                    # construct the foresee test data
                    [X_pred,index_te] = self.Test_Matrix_t(t,X_te)
                    # predict the future time of pred_size
                    y_bid_pred_B = AdaReg_bid_fit.predict(X_pred)
                    y_ask_pred_B = AdaReg_ask_fit.predict(X_pred)
                    sum_pred_bid += y_bid_pred_B
                    sum_pred_ask += y_ask_pred_B

                y_bid_pred = sum_pred_bid/self.resampling
                y_ask_pred = sum_pred_ask/self.resampling
                # After the prediction,
                # the process model will decide
                # wether buy in stock or short sell
                [Bid_pred_max, Bid_pred_min] = self.FindHL(y_bid_pred)
                [Ask_pred_max, Ask_pred_min] = self.FindHL(y_ask_pred)
                Buy_stock_indicator = Bid_pred_max*self.cautious_stock > Ask_t
                Buy_short_sell_indicator = Ask_pred_min < Bid_t*self.cautious_short_sell


                # f = plt.figure()
                # ax = f.add_subplot(111)
                # ax.plot(y_bid_pred,label = 'bidp')
                # ax.plot(y_ask_pred,label = 'askp')
                # ax.plot(y_te_bid[index_te],label = 'bid')
                # ax.plot(y_te_ask[index_te],label = 'askp')
                # ax.legend()
                # f.show()
                # plt.close(f)

                # Money you willing to pay at this round
                if self.money>0:
                    if self.safe_lock == 1:
                        inven = np.minimum((self.money/self.split),self.init_money)
                    else:
                        inven = self.money / self.split

                    if Buy_stock_indicator == True:
                        self.buy_stock(Ask_t, inven*0.3)

                    if Buy_short_sell_indicator == True:
                        self.buy_short_sell(Bid_t, inven*0.7)


                # run though the current inventory to see if there is profit
                pred_tr_one_bid = np.hstack(([Bid_t],y_bid_pred))
                pred_tr_one_ask = np.hstack(([Ask_t], y_ask_pred))
                [Bid_pred_max_one_tr, Bid_pred_min_one_tr] = self.FindHL(pred_tr_one_bid)
                [Ask_pred_max_one_tr, Ask_pred_min_one_tr] = self.FindHL(pred_tr_one_ask)

                # if Bid_pred_max_one_tr == Bid_t:
                self.sell_stock(Bid_t)
                # if Ask_pred_min_one_tr == Ask_t:
                self.promise_short_sell(Ask_t)


                print("At time:", t,"you have money:",self.money,
                      "stock:",len(self.stock),"short:",len(self.short_sell))


                # decide if buy in stock or short sell at this time

        pass

