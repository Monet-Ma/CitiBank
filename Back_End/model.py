from Predict import Predict
class Account:
    def __init__(self,account_items):
        self.accout_items = account_items
        self.count = len(account_items)

        self.ssp_dress = account_items[0]
        self.sp_dress = account_items[1]
        self.dress =  account_items[2]
        self.roles =  account_items[3]
        self.retinue =  account_items[4]
        self.ssp_goods =  account_items[5]
        self.sp_goods =  account_items[6]
        self.goods =  account_items[7]
        self.head =  account_items[8]
        self.head_frame =  account_items[9]
        self.graffiti =  account_items[10]
        self.waiting_action =  account_items[11]
        self.pursue_action =  account_items[12]
        self.music =  account_items[13]
        self.score =  account_items[14]
        self.platform =  account_items[15]
        self.price = account_items[16]

        self.predict = Predict(self.accout_items)

    @property
    def get_account(self):
        return self.accout_items

    def get_predict(self):
        return self.predict.prdict()

