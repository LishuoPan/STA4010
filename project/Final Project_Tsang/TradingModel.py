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

# import GPy
# construct packages
class TradingModel:
    def __init__(self, X_tr, y_bid, y_ask, money):
        self.y_tr_bid = y_bid
        self.y_tr_ask = y_ask
        self.X_tr = X_tr
        self.money = money
        self.train_size = 60
        self.pred_size = 20
        self.split = 4
        self.stock = list()
        self.short_sell = list()
        print("Money at the Begining of the day:", self.money)
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


        pass
    # def GaussianProcessRegression(self):
    #     kern1 = GPy.kern.sde_StdPeriodic()

    def sell_stock(self,y_bid):
        for index, eval in self.stock:
            price = eval[0]
            share = eval[1]
            earn = y_bid - price
            # exercise once have profit
            if earn > 0:
                self.money += earn * share
                self.stock.pop(index)

    def promise_short_sell(self,y_ask):
        for index, eval in self.short_sell:
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
        start = t+1
        end = start+self.pred_size
        index = np.arange(start,end)
        return X_te[index,:]


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

    def process(self,X_te,y_bid,y_ask):
        tol_t = len(X_te)
        # For each processing time
        for t in range(tol_t):
            # At the end of the day
            if t == tol_t - 1:
                self.settle_accounts(y_bid[t],y_ask[t])
                print("At the end of the day:", self.money)


            # In the process
            # When buy in short-sell or stock
            # append a [price, share] to list
            else:
                # prediction
                [X_tr_t, y_tr_bid, y_tr_ask] = self.Train_Matrix_t(t,X_te,y_bid,y_ask)
                [WLS_model_bid, WLS_model_ask] = self.WLS_regression(X_tr_t,y_tr_bid,y_tr_ask,p=1)
                X_pred = self.Test_Matrix_t(t,X_te)
                bid_pred = WLS_model_bid.predict(X_pred)
                ask_pred = WLS_model_ask.predict(X_pred)



                # run though the current inventory to see if there is profit
                self.sell_stock(y_bid[t])
                self.promise_short_sell(y_ask[t])

                # decide if buy in stock or short sell at this time

        pass

