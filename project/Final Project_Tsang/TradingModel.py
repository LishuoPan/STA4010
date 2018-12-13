# utf - 8
# Author: Lishuo Pan Data: Dec 12 2018
# load packages
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# import GPy
# construct packages
class TradingModel:
    def __init__(self, X_tr, y_bid, y_ask, money):
        self.y_tr_bid = y_bid
        self.y_tr_ask = y_ask
        self.X_tr = X_tr
        self.money = money
        self.train_size = 200
        self.pred_size = 10
        self.split = 20
        self.stock = list()
        self.short_sell = list()
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
        Ada_reg_model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                            n_estimators=500)
        Ada_reg_model.fit(X_tr, y_tr)
        return Ada_reg_model


    ##########################################################
    # Model behavior Part
    ##########################################################
    def FindHL(self, pred):
        High = np.max(pred)
        Low = np.min(pred)
        return High, Low
    def buy_stock(self,Ask_t, inven):
        price = Ask_t
        share = inven/Ask_t
        self.stock.append([price, share])
        self.money -= inven

    def buy_short_sell(self,Bid_t, inven):
        price = Bid_t
        share = inven/Bid_t
        self.short_sell.append([price, share])
        self.money += inven


    def sell_stock(self,y_bid):
        for index, eval in enumerate(self.stock):
            price = eval[0]
            share = eval[1]
            earn = y_bid - price
            # exercise once have profit
            if earn > 0:
                self.money += y_bid * share
                self.stock.pop(index)

    def promise_short_sell(self,y_ask):
        for index, eval in enumerate(self.short_sell):
            price = eval[0]
            share = eval[1]
            earn = price - y_ask
            # exercise once have profit
            if earn > 0:
                self.money -= y_ask*share
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
            # At the end of the day
            if t == tol_t - 1:
                self.settle_accounts(y_te_bid[t],y_te_ask[t])
                print("At the end of the day:", self.money)


            # In the process
            # When buy in short-sell or stock
            # append a [price, share] to list
            else:
                # prediction
                [X_tr_t, y_tr_bid, y_tr_ask] = \
                    self.Train_Matrix_t(t,X_te,y_te_bid,y_te_ask)

                # Fit AdaBoost regression model
                AdaReg_bid_fit = self.AdaBoostReg_fit(X_tr_t, y_tr_bid)
                AdaReg_ask_fit = self.AdaBoostReg_fit(X_tr_t, y_tr_ask)

                # construct the foresee test data
                [X_pred,index_te] = self.Test_Matrix_t(t,X_te)
                # predict the future time of pred_size
                y_bid_pred = AdaReg_bid_fit.predict(X_pred)
                y_ask_pred = AdaReg_ask_fit.predict(X_pred)

                # After the prediction,
                # the process model will decide
                # wether buy in stock or short sell
                [Bid_pred_max, Bid_pred_min] = self.FindHL(y_bid_pred)
                [Ask_pred_max, Ask_pred_min] = self.FindHL(y_ask_pred)
                [Bid_t, Ask_t] = [y_te_bid[t],y_te_ask[t]]
                Buy_stock_indicator = Bid_pred_max > Ask_t
                Buy_short_sell_indicator = Ask_pred_min < Bid_t

                # plt.plot(y_bid_pred)
                # plt.plot(y_te_bid[index_te])
                # plt.show()
                # Money you willing to pay at this round
                if self.money>0:
                    inven = self.money/self.split

                    if Buy_stock_indicator == True:
                        self.buy_stock(Ask_t, inven*0.3)

                    if Buy_short_sell_indicator == True:
                        self.buy_short_sell(Bid_t, inven*0.7)


                # run though the current inventory to see if there is profit
                self.sell_stock(y_te_bid[t])
                self.promise_short_sell(y_te_ask[t])


                print("At time:", t,"you have money:",self.money,
                      "stock:",len(self.stock),"short:",len(self.short_sell))


                # decide if buy in stock or short sell at this time

        pass

