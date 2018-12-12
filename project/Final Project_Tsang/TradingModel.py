# utf - 8
# Author: Lishuo Pan Data: Dec 12 2018
# load packages
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

# construct packages
class TradingModel:
    def __init__(self, X_tr, y_bid, y_ask, money):
        self.y_tr_bid = y_bid
        self.y_tr_ask = y_ask
        self.X_tr = X_tr
        self.money = money
        self.window_size = 60
        self.split = 4
        self.stock = list()
        self.short_sell = list()
        print("Money at the Begining of the day:", self.money)
    def WLS_window(self, count_down=0.9, window_size=60):
        pass


    def sell_stock(self,y_bid):
        for index, eval in self.stock:
            price = eval[0]
            share = eval[1]
            earn = y_bid - price
            if earn > 0:
                self.money += earn * share
                self.stock.pop(index)

    def promise_short_sell(self,y_ask):
        for index, eval in self.short_sell:
            price = eval[0]
            share = eval[1]
            earn = price - y_ask
            if earn > 0:
                self.money -= y_ask*share
                self.short_sell.pop(index)


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
                # run though the current inventory to see if there is profit
                self.sell_stock(y_bid[t])
                self.promise_short_sell(y_ask[t])

                # decide if buy in stock or short sell at this time

        pass

